"""M1 — 검출 후처리 파라미터 스윕 (재학습 없는 최저비용 개선).

실행: python -m src.eval.det_postproc_sweep

## 왜 후처리부터인가

`docs/metrics_policy.md` 3장 진단: 검출 실패의 **85%가 "텍스트를 못 찾은 것"이 아니라
"박스 경계 불일치"** 다(분할 56% + 부분 30%). 실패 박스의 조각들이 GT 면적의 평균 61%를
덮고 있다. 즉 네트워크는 텍스트를 보고 있고, **DB 후처리가 조각을 하나로 묶지 못하는 것**이다.

`unclip_ratio` 는 이진화된 텍스트 영역을 얼마나 팽창시킬지를 정하므로, 키우면 인접 조각이
병합된다. `thresh` 를 낮추면 이진화 영역 자체가 커진다. 둘 다 **재학습 없이 추론 시점에만
바뀌는 값**이라 지시서 F5 (a)가 지정한 최우선 저비용 경로다.

## 비용 설계

`TextDetRunnerPredictor.process()` 는 `runner(x)`(네트워크 forward, 비쌈)와
`post_op(preds, ...)`(후처리, 쌈)이 분리돼 있다. 따라서 **이미지당 forward 를 1회만 돌리고
파라미터 그리드 전체를 그 위에서 평가**한다. 그리드 크기와 무관하게 비용이 1회 추론과 같다.
"""

import collections
import itertools
import json

import cv2
import numpy as np

from src import spec
from src.eval.detect_baseline import match_boxes
from src.preprocess import parse_label, split as split_mod

MODEL = "PP-OCRv5_server_det"
GRID = {
    "thresh": (0.2, 0.3),
    "box_thresh": (0.3, 0.6),
    "unclip_ratio": (2.0, 2.5, 3.0),
}
# 조합마다 폴리곤 IoU 매칭을 돌리므로 이미지 수에 비례해 shapely 비용이 커진다.
# 방향성 판정에는 부분표본으로 충분하고, 확정 조합은 전체 test 로 재측정한다.
SAMPLE_IMAGES = 60


def _forward(predictor, img):
    """process() 의 전처리+네트워크 부분만 복제해 확률맵을 얻는다."""
    raw = predictor.pre_tfs["Read"](imgs=[img])
    resized, shapes = predictor.pre_tfs["Resize"](
        imgs=raw,
        limit_side_len=predictor.limit_side_len,
        limit_type=predictor.limit_type,
        max_side_limit=predictor.max_side_limit,
    )
    normed = predictor.pre_tfs["Normalize"](imgs=resized)
    chw = predictor.pre_tfs["ToCHW"](imgs=normed)
    return predictor.runner(x=predictor.pre_tfs["ToBatch"](imgs=chw)), shapes


def build(split="test"):
    from paddleocr import TextDetection

    records = split_mod.load_records()
    payload = json.loads(
        (spec.PROJECT_ROOT / "splits" / f"split_seed{spec.SPLIT_SEED}.json").read_text(encoding="utf-8"))
    members = set(payload["split"][split])

    source = {s.image: s.boxes for s in parse_label.parse()}
    by_image = collections.defaultdict(list)
    for r in records:
        if r["image"] in members:
            by_image[r["image"]].append(r)

    names = sorted(by_image)
    if SAMPLE_IMAGES:
        import random
        names = sorted(random.Random(0).sample(names, min(SAMPLE_IMAGES, len(names))))
        by_image = {k: by_image[k] for k in names}

    detector = TextDetection(model_name=MODEL, enable_mkldnn=False)
    predictor = detector.paddlex_predictor

    combos = list(itertools.product(*GRID.values()))
    keys = list(GRID)
    tp = {c: collections.Counter() for c in combos}
    n_pred = {c: 0 for c in combos}
    total = collections.Counter()

    for i, name in enumerate(sorted(by_image), 1):
        img = cv2.imread(str(spec.IMAGE_DIR / name))
        preds, shapes = _forward(predictor, img)   # 이미지당 1회

        gts = by_image[name]
        gt_polys = [source[name][g["box_index"]].points for g in gts]
        # 토큰 수는 metrics_policy 3장에서 검출 실패와 가장 강하게 연관된 축
        kinds = [f"tok{min(len(g['tokens']), 3)}" for g in gts]
        for k in kinds:
            total[k] += 1
        total["all"] += len(gts)

        for combo in combos:
            params = dict(zip(keys, combo))
            polys, _ = predictor.post_op(preds, shapes, **params)
            polys = [np.asarray(p) for p in polys[0]]
            n_pred[combo] += len(polys)
            for gi in match_boxes(gt_polys, polys):
                tp[combo][kinds[gi]] += 1
                tp[combo]["all"] += 1

        if i % 10 == 0:
            print(f"  {i}/{len(by_image)}", flush=True)

    rows = []
    for combo in combos:
        r = tp[combo]["all"] / total["all"]
        p = tp[combo]["all"] / n_pred[combo] if n_pred[combo] else 0.0
        rows.append({
            **dict(zip(keys, combo)),
            "recall": r, "precision": p,
            "hmean": 2 * r * p / (r + p) if r + p else 0.0,
            "n_pred": n_pred[combo],
            "by_tokens": {k: tp[combo][k] / total[k] for k in total if k != "all"},
        })

    rows.sort(key=lambda x: -x["recall"])
    out = {"model": MODEL, "split": split, "grid": GRID,
           "gt_boxes": total["all"], "gt_by_tokens": dict(total), "results": rows}
    (spec.PROJECT_ROOT / "reports" / "det_postproc_sweep.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    base = next(r for r in rows if (r["thresh"], r["box_thresh"], r["unclip_ratio"]) == (0.3, 0.6, 2.0))
    print(f"\n기본값(thresh=0.3, box_thresh=0.6, unclip=2.0): "
          f"recall={base['recall']*100:.1f}% precision={base['precision']*100:.1f}%")
    print(f"\n{'thresh':>7} {'box_th':>7} {'unclip':>7} {'recall':>8} {'prec':>7} {'hmean':>7}  토큰별 recall")
    for r in rows[:12]:
        tk = " ".join(f"{k}:{v*100:.0f}%" for k, v in sorted(r["by_tokens"].items()))
        print(f"{r['thresh']:>7} {r['box_thresh']:>7} {r['unclip_ratio']:>7} "
              f"{r['recall']*100:>7.1f}% {r['precision']*100:>6.1f}% {r['hmean']*100:>6.1f}%  {tk}")
    return out


if __name__ == "__main__":
    build()
