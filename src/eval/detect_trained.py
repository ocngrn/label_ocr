"""학습한 2.x 검출 모델을 **우리 지표**로 재측정한다 (M3 판정).

실행 (Colab):
    python -m src.eval.detect_trained --weights <exp>/best_accuracy --tag det_v4_finetuned
    python -m src.eval.detect_trained --weights /content/weights/.../best_accuracy \\
        --tag det_v4_pretrained          # 같은 경로로 잰 '학습 전' 값

## 왜 따로 재는가

`tools/eval.py` 의 hmean 은 PaddleOCR 의 `DetMetric` 이 **val 234장**에서 잰 값이고,
`reports/baseline_det_v4.json` 은 `detect_baseline` 이 **test + 곡면 318장**에서 잰 값이다.
대상도 매칭 구현도 달라 직접 비교하면 안 된다. M3 판정은 시스템 Top-5 로 이어지는
우리 지표로만 가능하다.

지표 계산은 `detect_baseline.evaluate` 를 그대로 쓰고 **검출기만 갈아끼운다.**
복제하면 두 수치가 조용히 갈라진다.

## 왜 export 하지 않는가

`tools/export_model.py` 는 paddle 3.3 에서 깨진다:

    TypeError: sigmoid(): argument (position 1) must be Value, but got Variable

2.10.0 의 export 경로가 구 정적그래프(Variable)를 쓰는데 paddle 3.x 는 PIR(Value)로
넘어갔다. `FLAGS_enable_pir_api=0` 으로도 되돌아가지 않는다. 그래서 dygraph 로 직접
추론한다 — 학습·평가와 같은 코드 경로라 오히려 교란이 적다.

## 학습 전/후를 같은 경로로 잰다

기준선(`baseline_det_v4.json`)은 paddleocr 3.7 의 `TextDetection`(PaddleX 기본 960/max)로
쟀는데, 2.x 평가의 `DetResizeForTest` 기본값은 **736/min** 이다. 입력 해상도가 다르면
가중치 효과와 해상도 효과가 섞인다. 그래서 사전학습 가중치도 이 스크립트로 다시 재서
**가중치만 다른 비교**를 만든다.
"""

import argparse
import json
import sys

import numpy as np

from src import spec
from src.eval import detect_baseline


def build_predict(paddleocr_root, config_path, weights,
                  limit_side_len=736, limit_type="min"):
    """dygraph 로 추론하는 `predict(img) -> [폴리곤, ...]` 을 만든다."""
    sys.path.insert(0, str(paddleocr_root))
    import paddle
    import yaml
    from ppocr.data.imaug import create_operators, transform
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    model = build_model(cfg["Architecture"])
    sd = paddle.load(str(weights) + ".pdparams")
    msd = model.state_dict()
    # 사전학습 가중치에는 분류용 fc/last_conv 가 남아 있어 그대로 넣으면 실패한다
    keep = {k: sd[k] for k in msd if k in sd and tuple(msd[k].shape) == tuple(sd[k].shape)}
    assert len(keep) / len(msd) > 0.9, f"가중치 전이율이 낮다: {len(keep)}/{len(msd)}"
    model.set_state_dict(keep)
    model.eval()

    post = build_post_process(cfg["PostProcess"])
    ops = create_operators([
        {"DetResizeForTest": {"limit_side_len": limit_side_len, "limit_type": limit_type}},
        {"NormalizeImage": {"scale": "1./255.",
                            "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225], "order": "hwc"}},
        {"ToCHWImage": None},
        {"KeepKeys": {"keep_keys": ["image", "shape"]}},
    ], None)

    def predict(img):
        data = transform({"image": img}, ops)
        if data is None:
            return []
        image, shape = data
        with paddle.no_grad():
            preds = model(paddle.to_tensor(image[None]))
        result = post(preds, np.array([shape]))
        return [np.asarray(p) for p in result[0]["points"]]

    return predict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--paddleocr", default="/content/PaddleOCR")
    p.add_argument("--config", default=str(spec.PROJECT_ROOT / "configs" / "det_ppocrv4_server.yml"))
    p.add_argument("--weights", required=True, help="확장자 제외한 접두사")
    p.add_argument("--tag", required=True, help="reports 에 기록할 이름")
    p.add_argument("--limit-side-len", type=int, default=736)
    p.add_argument("--limit-type", default="min")
    p.add_argument("--out", default=None, help="reports/ 아래 파일명 (기본 <tag>.json)")
    a = p.parse_args()

    m = detect_baseline.evaluate(predict=build_predict(
        a.paddleocr, a.config, a.weights, a.limit_side_len, a.limit_type))
    m["model"] = a.tag
    m["resize"] = {"limit_side_len": a.limit_side_len, "limit_type": a.limit_type}

    o = m["overall"]
    print(f"\n{a.tag}  이미지 {m['images_evaluated']}장 "
          f"(IoU>={m['iou_threshold']}, resize {a.limit_side_len}/{a.limit_type})")
    print(f"  전체  GT={o['gt']} 예측={o['pred']} TP={o['tp']}  "
          f"recall={o['recall']*100:.1f}%  precision={o['precision']*100:.1f}%  "
          f"Hmean={o['hmean']*100:.1f}%")
    for k in ("plane", "curved"):
        s = m[k]
        print(f"  {k:6s} GT={s['gt']:4d} TP={s['tp']:4d}  recall={s['recall']*100:.1f}%")
    print(f"  시스템 Top-5 추정 = plane recall x 85.6% = "
          f"{m['plane']['recall']*0.856*100:.1f}%")

    path = spec.PROJECT_ROOT / "reports" / (a.out or f"{a.tag}.json")
    path.write_text(json.dumps({a.tag: m}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기록: {path}")


if __name__ == "__main__":
    main()
