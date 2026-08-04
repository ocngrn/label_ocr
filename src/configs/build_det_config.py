"""검출 학습 config 생성 (TASK 6-A / M3).

실행:
    python -m src.configs.build_det_config \\
        --template <PaddleOCR>/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml \\
        --out configs/det_ppocrv4_server.yml \\
        --root /content/label_ocr --weights /content/weights/best_accuracy

## 왜 생성하는가

Colab 노트북에서 yml 을 손으로 고치면 세션이 끊길 때마다 사라지고, 어떤 config 로 돌린
실험인지 추적이 안 된다. 값의 출처는 `src/spec.py` 하나이고(기획서 7장 재현성 규칙),
노트북은 오케스트레이션만 담당한다(지시서 Colab 규약).

## 템플릿 선택

`ch_PP-OCRv4_det_teacher.yml` = PPHGNet_small + LKPAN = **server** 계열.
baseline 을 PP-OCRv5_server_det 로 쟀으므로 server 계열로 맞춘다
(student 는 PPLCNetV3 + RSEFPN 인 mobile).

## 후처리 파라미터는 기본값 유지

M1 스윕에서 `thresh=0.3 / box_thresh=0.6 / unclip_ratio` 기본값을 넘는 조합이 없었다
(`reports/m1_m2_detection.md`). 학습 후 분포가 바뀌면 재스윕한다.

## AMP 를 쓰지 않는다 (`use_amp=False`)

DBLoss 의 Dice 항이 fp16 에서 **조용히 망가진다.** 손실이 정확히 고정되는 것으로 드러난다:

    loss_shrink_maps: 5.000000   loss_binary_maps: 1.000000   loss_cbn: 1.000000

`union = sum(pred * mask)` 는 배치 4 x 640 x 640 = 164만 픽셀의 합이라 fp16 상한
65504 를 넘겨 `inf` 가 된다. 반면 `intersection` 은 텍스트 픽셀(약 2%)만 더하므로
정상 범위다. 따라서 `1 - 2 * intersection / inf` = **정확히 1.0** 이 되고 기울기가 0 이 된다.
`loss_threshold_maps` 만 정상으로 보이는 이유는 그것이 마스크 합으로 나누는 L1 이라
오버플로가 없기 때문이다. PaddleOCR 의 `amp_level` 기본값이 O2(순수 fp16)라 더 잘 터진다.

순전파 출력만 보면 정상이라 발견이 늦는다. 판정은 **손실이 실제로 내려가는가**로 한다.
"""

import argparse
import copy
import json
from pathlib import Path

import yaml

from src import spec


def deep_merge(base: dict, override: dict) -> dict:
    """override 의 값으로 base 를 재귀 갱신한 새 dict 를 반환."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def det_overrides(root, weights, save_dir, checkpoints=None,
                  epochs=50, batch_size=8, num_workers=4, use_amp=False, lr=1e-4):
    """우리 데이터·환경에 맞춘 override 트리.

    `root` 는 Colab 세션 로컬 디스크의 프로젝트 루트(예: `/content/label_ocr`).
    Drive 에서 이미지를 직접 읽으면 DataLoader 가 심하게 느려지므로 로컬 디스크를 쓴다.
    반대로 `save_dir` 는 Drive 로 둬야 세션이 끊겨도 체크포인트가 남는다.
    """
    root = str(root).rstrip("/")
    # 라벨의 이미지 경로가 디렉터리 접두사 없는 파일명이라 data_dir 를 image_set 으로 잡는다
    data_dir = f"{root}/image_set"
    return {
        "Global": {
            "use_gpu": True,
            "epoch_num": epochs,
            "save_model_dir": str(save_dir),
            # Colab 세션 만료에 대비해 자주 저장하고, checkpoints 로 이어서 학습한다
            "save_epoch_step": 5,
            "eval_batch_step": [0, 200],
            "pretrained_model": str(weights),
            "checkpoints": checkpoints,
            "use_amp": use_amp,
            "print_batch_step": 20,
            "use_visualdl": True,
        },
        # 템플릿 기본값 1e-3 은 10만 장급 코퍼스용이다. 우리 학습셋은 1,872장이고
        # 가르칠 것은 "글자를 찾는 법"이 아니라 박스 규약 하나뿐이라 그 값은 과하다.
        # 실측(2026-08-04): 1e-3 으로 6 epoch 학습하면 val hmean 이 0.6370 -> 0.6024 로
        # 오히려 내려간다. 사전학습 특징이 파괴되는 것이다.
        "Optimizer": {"lr": {"learning_rate": lr}},
        "Train": {
            "dataset": {"data_dir": data_dir,
                        "label_file_list": [f"{root}/labels/det_train.txt"],
                        "ratio_list": [1.0]},
            "loader": {"batch_size_per_card": batch_size, "num_workers": num_workers,
                       "shuffle": True, "drop_last": False},
        },
        "Eval": {
            "dataset": {"data_dir": data_dir,
                        "label_file_list": [f"{root}/labels/det_val.txt"]},
            "loader": {"batch_size_per_card": 1, "num_workers": max(num_workers // 2, 1),
                       "shuffle": False, "drop_last": False},
        },
    }


def build(template, out, **kwargs):
    base = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
    merged = deep_merge(base, det_overrides(**kwargs))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(
        yaml.safe_dump(merged, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True, help="PaddleOCR 2.x 의 det yml 경로")
    p.add_argument("--out", default=str(spec.PROJECT_ROOT / "configs" / "det_ppocrv4_server.yml"))
    p.add_argument("--root", default="/content/label_ocr", help="세션 로컬 디스크의 프로젝트 루트")
    p.add_argument("--weights", required=True, help="사전학습 가중치 접두사 (확장자 제외)")
    p.add_argument("--save-dir", required=True, help="체크포인트 저장 경로 (Drive 권장)")
    p.add_argument("--checkpoints", default=None, help="resume 할 체크포인트 접두사")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()

    cfg = build(a.template, a.out, root=a.root, weights=a.weights, save_dir=a.save_dir,
                checkpoints=a.checkpoints, epochs=a.epochs, batch_size=a.batch_size,
                lr=a.lr)
    print(f"생성: {a.out}")
    print(json.dumps({"Global": {k: cfg["Global"][k] for k in
                                 ("use_gpu", "epoch_num", "save_epoch_step",
                                  "pretrained_model", "checkpoints", "use_amp")},
                      "Train.data_dir": cfg["Train"]["dataset"]["data_dir"],
                      "Train.labels": cfg["Train"]["dataset"]["label_file_list"],
                      "Eval.labels": cfg["Eval"]["dataset"]["label_file_list"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
