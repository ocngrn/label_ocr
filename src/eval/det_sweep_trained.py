"""학습한 2.x 검출 모델의 후처리 파라미터 스윕.

실행 (Colab):
    python -m src.eval.det_sweep_trained --weights <exp>/best_accuracy --split val
    python -m src.eval.det_sweep_trained --weights <exp>/best_accuracy --split test \\
        --grid-thresh 0.3 --grid-box-thresh 0.5 --grid-unclip 2.0     # 확정값 확인

## 기존 `det_postproc_sweep.py` 와의 관계

그것은 M1 용이고 PaddleX 예측기 내부(`pre_tfs`/`runner`/`post_op`)에 묶여 있어
2.x dygraph 모델에는 쓸 수 없다. 여기서는 `detect_trained.build_forward` 를 재사용한다.

## 비용 설계

순전파는 비싸고 후처리는 싸다. **이미지당 순전파 1회**로 파라미터 조합 전체를 평가하므로
그리드 크기와 무관하게 비용이 1회 추론과 같다. 조합마다 `evaluate()` 를 부르면
순전파가 조합 수만큼 반복돼 30분으로 끝나지 않는다.

## 왜 val 에서 고르는가

후처리 파라미터를 test 에서 고르면 그 test 점수는 더 이상 정직하지 않다.
M1 때는 개선이 0이라 문제가 드러나지 않았지만, 이번엔 실제로 값을 고른다.
**val 에서 선택 → test 로 1회 확인** 이 순서를 지킨다.

## 판정 기준

우리 시스템 KPI 는 `Top-5 ≈ plane recall × 85.6%` 라 recall 이 지배한다. 그러나 recall 만
최대화하면 `box_thresh` 를 0 에 붙여 예측을 폭증시키는 쓸모없는 설정이 최적으로 뽑힌다.
그래서 **precision 이 현재 운영점 아래로 떨어지지 않는 선에서 recall 최대화**로 고른다.
후보 노출 개수에 대한 UI 비용 모델이 없으므로, 나오기 전까지는 오검출 부담을 늘리지 않는
쪽으로 방어적으로 잡는다.
"""

import argparse
import itertools
import json
from collections import defaultdict

import cv2
import numpy as np

from src import spec
from src.eval import detect_baseline, detect_trained
from src.preprocess import parse_label, split as split_mod

GRID = {
    "thresh": (0.2, 0.3, 0.4),
    "box_thresh": (0.3, 0.4, 0.5, 0.6, 0.7),
    "unclip_ratio": (1.2, 1.5, 2.0, 2.5),
}


def sweep(paddleocr_root, config_path, weights, split="test",
          limit_side_len=736, limit_type="min", grid=None, progress_every=25):
    import sys
    sys.path.insert(0, str(paddleocr_root))
    from ppocr.postprocess import build_post_process

    grid = grid or GRID
    forward, cfg = detect_trained.build_forward(
        paddleocr_root, config_path, weights, limit_side_len, limit_type)

    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    # 후처리기는 생성 비용이 없으므로 조합마다 미리 만들어 둔다
    posts = {c: build_post_process({**cfg["PostProcess"], **dict(zip(keys, c))})
             for c in combos}

    records = split_mod.load_records()
    source = {s.image: s.boxes for s in parse_label.parse()}
    by_image = defaultdict(list)
    for r in records:
        by_image[r["image"]].append(r)
    targets = detect_baseline.targets_for(split, extra_curved=False)

    tp = {c: {"plane": 0, "curved": 0} for c in combos}
    n_pred = {c: 0 for c in combos}
    total_gt = {"plane": 0, "curved": 0}

    for n, name in enumerate(targets, 1):
        img = cv2.imread(str(spec.IMAGE_DIR / name))
        preds, shape = forward(img)                      # 이미지당 1회
        if preds is None:
            continue

        gts = by_image[name]
        kinds = ["curved" if g["is_curved"] else "plane" for g in gts]
        for k in kinds:
            total_gt[k] += 1
        gt_polys = [source[name][g["box_index"]].points for g in gts]

        for c in combos:
            polys = [np.asarray(p) for p in posts[c](preds, shape)[0]["points"]]
            n_pred[c] += len(polys)
            for gi in detect_baseline.match_boxes(gt_polys, polys):
                tp[c][kinds[gi]] += 1

        if n % progress_every == 0:
            print(f"  {n}/{len(targets)} 처리", flush=True)

    total = total_gt["plane"] + total_gt["curved"]
    rows = []
    for c in combos:
        t = tp[c]["plane"] + tp[c]["curved"]
        r = t / total if total else 0.0
        p = t / n_pred[c] if n_pred[c] else 0.0
        rows.append({
            **dict(zip(keys, c)),
            "plane_recall": tp[c]["plane"] / total_gt["plane"] if total_gt["plane"] else 0.0,
            "curved_recall": tp[c]["curved"] / total_gt["curved"] if total_gt["curved"] else 0.0,
            "recall": r, "precision": p,
            "hmean": 2 * r * p / (r + p) if r + p else 0.0,
            "n_pred": n_pred[c],
        })
    return {"split": split, "images": len(targets), "gt": total_gt,
            "resize": {"limit_side_len": limit_side_len, "limit_type": limit_type},
            "grid": {k: list(v) for k, v in grid.items()}, "results": rows}


def pick(rows, min_precision):
    """precision 하한을 지키는 조합 중 plane recall 최대. 없으면 None."""
    ok = [r for r in rows if r["precision"] >= min_precision]
    return max(ok, key=lambda r: r["plane_recall"]) if ok else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paddleocr", default="/content/PaddleOCR")
    p.add_argument("--config", default=str(spec.PROJECT_ROOT / "configs" / "det_ppocrv4_server.yml"))
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--limit-side-len", type=int, default=736)
    p.add_argument("--limit-type", default="min")
    p.add_argument("--min-precision", type=float, default=0.565,
                   help="현재 운영점(원본 해상도 precision). 이 아래로는 고르지 않는다")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    out = sweep(a.paddleocr, a.config, a.weights, a.split,
                a.limit_side_len, a.limit_type)
    rows = sorted(out["results"], key=lambda r: -r["plane_recall"])

    print(f"\nsplit={out['split']} 이미지 {out['images']}장 "
          f"GT plane={out['gt']['plane']} curved={out['gt']['curved']}")
    print(f"{'thresh':>7} {'box_th':>7} {'unclip':>7} {'plane_r':>8} {'recall':>7} "
          f"{'prec':>7} {'hmean':>7} {'n_pred':>7}")
    for r in rows[:15]:
        print(f"{r['thresh']:>7} {r['box_thresh']:>7} {r['unclip_ratio']:>7} "
              f"{r['plane_recall']*100:>7.1f}% {r['recall']*100:>6.1f}% "
              f"{r['precision']*100:>6.1f}% {r['hmean']*100:>6.1f}% {r['n_pred']:>7}")

    best = pick(out["results"], a.min_precision)
    out["min_precision"] = a.min_precision
    out["picked"] = best
    if best:
        print(f"\n선택 (precision >= {a.min_precision*100:.1f}%): "
              f"thresh={best['thresh']} box_thresh={best['box_thresh']} "
              f"unclip_ratio={best['unclip_ratio']}")
        print(f"  plane recall={best['plane_recall']*100:.1f}%  "
              f"precision={best['precision']*100:.1f}%  "
              f"시스템 Top-5 추정={best['plane_recall']*0.856*100:.1f}%")
    else:
        print(f"\n!! precision >= {a.min_precision*100:.1f}% 를 만족하는 조합이 없다")

    path = spec.PROJECT_ROOT / "reports" / (a.out or f"det_sweep_trained_{a.split}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기록: {path}")


if __name__ == "__main__":
    main()
