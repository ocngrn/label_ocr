"""TASK 4-3 [게이트] — 검출 baseline 및 fine-tuning 필요성 판단.

실행: python -m src.eval.detect_baseline
      python -m src.eval.detect_baseline --model PP-OCRv4_server_det --out baseline_det_v4.json

지시서는 검출 성능을 **평면(4점)/곡면(5점 이상) 서브셋으로 나눠** 재라고 요구한다.
통합 수치만 보면 곡면 실패(1.2%)가 평면 다수(98.8%)에 묻히기 때문이다.

## 평가 대상 이미지

기본은 **test 만** (`extra_curved=False`).

`extra_curved=True` 는 곡면 이미지 93장을 전부 끌어와 붙인다. test 에 배분된 곡면 박스가
9개뿐이라 곡면 판단이 통계적으로 불가능하기 때문인데(예측 1건이 11%p),
**이는 학습하지 않은 모델에만 쓸 수 있다.**

    곡면 이미지 93장 = train 75 / val 9 / test 9
    test + 곡면 전량 = 318장, GT 1022박스 중 236개(23.1%)가 train 출처

학습한 모델을 이 위에서 재면 23%가 학습 데이터가 된다. 곡면 박스만 보면 99개 중 90개(91%)다.
학습 모델의 곡면 성능은 곡면 5-Fold(`split_stats.md` §4)로만 정직하게 잴 수 있다.

## 지표

ICDAR 방식 — 예측 폴리곤과 GT 폴리곤을 IoU >= 0.5 로 그리디 1:1 매칭.
recall 은 GT 기준이라 서브셋별로 명확히 정의된다. precision 은 예측에 유형이 없으므로
**매칭된 GT 의 유형으로 귀속**하고, 어느 GT 와도 매칭되지 않은 예측(FP)은 전역으로만 센다.
따라서 서브셋 Hmean 은 참고치이며, **판단의 1차 근거는 서브셋 recall** 이다.

## CPU 주의

paddle 3.3.1 CPU 에서 검출 모델은 oneDNN 경로가 깨진다
(`ConvertPirAttribute2RuntimeAttribute not support`). `enable_mkldnn=False` 필수.
"""

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np
from shapely.geometry import Polygon

from src import spec
from src.preprocess import parse_label, split as split_mod

IOU_THRESHOLD = 0.5


def _poly(points):
    p = Polygon(np.asarray(points, dtype=float))
    return p if p.is_valid else p.buffer(0)


def match_boxes(gt_polys, pred_polys, thr=IOU_THRESHOLD):
    """IoU 기준 그리디 1:1 매칭 → {gt 인덱스: pred 인덱스}."""
    gts = [_poly(g) for g in gt_polys]
    preds = [_poly(p) for p in pred_polys]
    pairs = []
    for gi, g in enumerate(gts):
        if g.is_empty or g.area <= 0:
            continue
        for pi, p in enumerate(preds):
            if p.is_empty or p.area <= 0:
                continue
            inter = g.intersection(p).area
            if inter <= 0:
                continue
            iou = inter / (g.area + p.area - inter)
            if iou >= thr:
                pairs.append((iou, gi, pi))

    pairs.sort(reverse=True)
    used_g, used_p, matched = set(), set(), {}
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matched[gi] = pi
    return matched


def targets_for(split="test", extra_curved=False):
    """평가 대상 이미지 목록. `extra_curved` 는 학습하지 않은 모델 전용 (모듈 docstring 참조)."""
    records = split_mod.load_records()
    payload = json.loads(
        (spec.PROJECT_ROOT / "splits" / f"split_seed{spec.SPLIT_SEED}.json").read_text(encoding="utf-8"))
    names = set(payload["split"][split])
    if extra_curved:
        names |= {r["image"] for r in records if r["is_curved"]}
    return sorted(names)


def evaluate(model_name="PP-OCRv5_server_det", predict=None,
             split="test", extra_curved=False):
    """`predict` 를 주면 그 검출기로 잰다 — 지표·대상·매칭 규칙은 그대로 둔다.

    학습한 2.x 모델을 재측정할 때 이 함수를 복제하면 두 수치가 조용히 갈라진다.
    `predict(img) -> [폴리곤, ...]` 하나만 갈아끼우면 비교 가능성이 보장된다.

    `extra_curved` 기본값이 False 인 이유는 모듈 docstring 참조 — 학습한 모델에
    True 를 주면 평가 박스의 23%가 학습 데이터가 된다.
    """
    records = split_mod.load_records()

    # 원본 폴리곤은 Label.txt 에서 box_index 로 직접 가져온다
    source = {s.image: s.boxes for s in parse_label.parse()}

    boxes_by_image = defaultdict(list)
    for r in records:
        boxes_by_image[r["image"]].append(r)

    targets = targets_for(split, extra_curved)

    if predict is None:
        from paddleocr import TextDetection
        detector = TextDetection(model_name=model_name, enable_mkldnn=False)

        def predict(img):
            return [np.asarray(p) for p in list(detector.predict([img]))[0]["dt_polys"]]

    tp = {"plane": 0, "curved": 0}
    total_gt = {"plane": 0, "curved": 0}
    matched_pred = {"plane": 0, "curved": 0}
    total_pred = 0

    for n, name in enumerate(targets, 1):
        img = cv2.imread(str(spec.IMAGE_DIR / name))
        preds = predict(img)
        total_pred += len(preds)

        gts = boxes_by_image[name]
        kinds = ["curved" if g["is_curved"] else "plane" for g in gts]
        for k in kinds:
            total_gt[k] += 1

        gt_polys = [source[name][g["box_index"]].points for g in gts]
        for gi in match_boxes(gt_polys, preds):
            tp[kinds[gi]] += 1
            matched_pred[kinds[gi]] += 1

        if n % 25 == 0:
            print(f"  {n}/{len(targets)} 처리", flush=True)

    def prf(subset):
        r = tp[subset] / total_gt[subset] if total_gt[subset] else 0.0
        p = tp[subset] / matched_pred[subset] if matched_pred[subset] else 0.0
        return {"gt": total_gt[subset], "tp": tp[subset], "recall": r, "precision_matched": p}

    total_tp = tp["plane"] + tp["curved"]
    total = total_gt["plane"] + total_gt["curved"]
    recall = total_tp / total if total else 0.0
    precision = total_tp / total_pred if total_pred else 0.0
    hmean = 2 * recall * precision / (recall + precision) if recall + precision else 0.0

    return {
        "model": model_name,
        "split": split,
        "extra_curved": extra_curved,
        "images_evaluated": len(targets),
        "iou_threshold": IOU_THRESHOLD,
        "overall": {"gt": total, "pred": total_pred, "tp": total_tp,
                    "recall": recall, "precision": precision, "hmean": hmean},
        "plane": prf("plane"),
        "curved": prf("curved"),
    }



def main(models=("PP-OCRv5_server_det",), out_name="baseline_det_metrics.json"):
    out = {}
    for name in models:
        m = evaluate(name)
        out[name] = m
        o = m["overall"]
        print(f"\n{name}  이미지 {m['images_evaluated']}장 (IoU>={m['iou_threshold']})")
        print(f"  전체  GT={o['gt']} 예측={o['pred']} TP={o['tp']}  "
              f"recall={o['recall']*100:.1f}%  precision={o['precision']*100:.1f}%  Hmean={o['hmean']*100:.1f}%")
        for k in ("plane", "curved"):
            s = m[k]
            print(f"  {k:6s} GT={s['gt']:4d} TP={s['tp']:4d}  recall={s['recall']*100:.1f}%")

    path = spec.PROJECT_ROOT / "reports" / out_name
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", help="반복 지정 가능")
    p.add_argument("--out", default="baseline_det_metrics.json", help="reports/ 아래 파일명")
    a = p.parse_args()
    main(tuple(a.model or ["PP-OCRv5_server_det"]), a.out)
