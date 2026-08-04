"""M2 — 검출 조각을 원본 GT 박스 단위로 재결합해 평가 (토큰 단위 전환 검증).

실행: python -m src.eval.token_merge

## 배경

M1(후처리 스윕)은 실패했다. `unclip_ratio` 는 컨투어를 추출한 **뒤** 각 영역을 팽창시키므로
분리된 조각을 합치지 못하고 박스만 키워 IoU 를 떨어뜨린다(2.0→73.8%, 3.0→55.1%).
2토큰 recall 은 12개 조합 전부에서 44% 로 고정이었다.

반면 **단일 토큰 recall 은 84%** 다. 검출기는 토큰을 정확히 찾고 있고, 우리가 다중 토큰을
한 박스로 묶어둔 것(TASK 1-3 (a)안)이 불일치의 원인이다.

## 이 실험이 검증하는 것

라벨을 재생성하지 않고도 전환 효과를 잴 수 있다 — **검출 결과를 GT 박스 단위로 재결합**하면
"검출·인식은 토큰 단위, 조회는 결합 후"와 동등한 파이프라인이 된다.

    기존:  GT 박스 1개 ↔ 검출 1개 (IoU>=0.5 요구) -> 2토큰에서 44%
    제안:  GT 박스 1개 ↔ 검출 N개 (GT 안에 들어온 조각 전부) -> 각각 인식 후 이어붙임

채점 단위는 **원본 GT 박스 그대로**다(사용자 확정 KPI 단위 유지).

## 귀속 규칙

검출 박스의 면적 중 GT 박스와 겹치는 비율이 `INSIDE_RATIO` 이상이면 그 GT 에 귀속시킨다.
IoU 가 아니라 "검출 박스 기준 포함률"을 쓰는 이유는, 조각은 GT 보다 작아서 IoU 가 낮아도
GT 안에 완전히 들어있기 때문이다. 귀속된 조각은 GT 박스의 장축 방향으로 정렬해 이어붙인다.
"""

import collections
import json

import cv2
import numpy as np

from src import spec
from src.eval.detect_baseline import _poly
from src.eval.infer import recognize
from src.matching import archive, build_db, match
from src.preprocess import crop, normalize, parse_label, split as split_mod

DET_MODEL = "PP-OCRv5_server_det"
REC_MODEL = "PP-OCRv5_server_rec"
INSIDE_RATIO = 0.5
LENGTH_BINS = ((1, 2), (3, 4), (5, 8), (9, 14), (15, 40))


def assign_and_order(gt_points, pred_polys):
    """GT 박스에 귀속되는 검출 인덱스를 읽기 순서(장축 방향)로 반환."""
    gt = _poly(gt_points)
    if gt.is_empty or gt.area <= 0:
        return []

    corners = np.asarray(gt_points, dtype=float)
    centroid = corners.mean(axis=0)
    # 장축 = 코너 분산이 가장 큰 방향 (회전된 텍스트 대응)
    _, _, vt = np.linalg.svd(corners - centroid)
    axis = vt[0]

    hits = []
    for i, poly in enumerate(pred_polys):
        p = _poly(poly)
        if p.is_empty or p.area <= 0:
            continue
        if gt.intersection(p).area / p.area >= INSIDE_RATIO:
            center = np.asarray(poly, dtype=float).mean(axis=0)
            hits.append((float(np.dot(center - centroid, axis)), i))
    return [i for _, i in sorted(hits)]


def run(split="test"):
    from paddleocr import TextDetection, TextRecognition

    records = split_mod.load_records()
    payload = json.loads(
        (spec.PROJECT_ROOT / "splits" / f"split_seed{spec.SPLIT_SEED}.json").read_text(encoding="utf-8"))
    members = set(payload["split"][split])
    seen_flags = payload["seen_in_train"][split]

    source = {s.image: s.boxes for s in parse_label.parse()}
    by_image = collections.defaultdict(list)
    for r in records:
        if r["image"] in members:
            by_image[r["image"]].append(r)

    detector = TextDetection(model_name=DET_MODEL, enable_mkldnn=False)
    items, patches = [], []

    for i, name in enumerate(sorted(by_image), 1):
        img = cv2.imread(str(spec.IMAGE_DIR / name))
        preds = [np.asarray(p) for p in list(detector.predict([img]))[0]["dt_polys"]]

        for g in by_image[name]:
            order = assign_and_order(source[name][g["box_index"]].points, preds)
            idx = []
            for j in order:
                patch = crop.crop_box(img, preds[j])
                if patch is not None and patch.size and min(patch.shape[:2]) >= 2:
                    idx.append(len(patches))
                    patches.append(patch)
            items.append({
                "crop": g["crop"], "gt": g["label"], "is_curved": g["is_curved"],
                "seen": seen_flags[g["crop"]], "n_fragments": len(idx), "patch_idx": idx,
            })

        if i % 20 == 0:
            print(f"  검출 {i}/{len(by_image)}", flush=True)

    print(f"  검출 완료: GT {len(items)}박스, 크롭 {len(patches)}개, "
          f"조각 확보된 GT {sum(1 for x in items if x['n_fragments'])}박스", flush=True)

    texts = recognize(TextRecognition(model_name=REC_MODEL), patches, batch_size=8)
    for x in items:
        parts = [texts[j] for j in x["patch_idx"]]
        x["pred"] = normalize.normalize_text(" ".join(t for t, _, _ in parts).strip())
        x["char_confidences"] = [c for _, confs, _ in parts for c in confs]
        x["mean_confidence"] = float(np.mean([s for _, _, s in parts])) if parts else 0.0
    return items


def score(items):
    db = build_db.load()
    matcher = match.Matcher(db["serials"])
    canonical = db["canonical"]

    top5 = matcher.top_k([x["pred"] for x in items], k=5)
    for x, cands in zip(items, top5):
        names = [c.upper() for c, _ in cands]
        truth = canonical[x["gt"]].upper()
        x["hit5"] = bool(x["pred"]) and truth in names
        x["hit1"] = bool(x["pred"]) and truth == names[0]

    archive.save(
        run_id=f"m2_token_merge_{REC_MODEL}_test",
        model={"det": DET_MODEL, "rec": REC_MODEL, "pipeline": "token-merge"},
        split="test",
        predictions=[{k: x[k] for k in
                      ("crop", "gt", "pred", "char_confidences", "mean_confidence")} for x in items],
        db={"tag": db["tag"], "size": db["size"]},
    )

    def agg(rows):
        if not rows:
            return None
        return {
            "n": len(rows),
            "coverage": float(np.mean([r["n_fragments"] > 0 for r in rows])),
            "avg_fragments": float(np.mean([r["n_fragments"] for r in rows])),
            "top1": float(np.mean([r["hit1"] for r in rows])),
            "top5": float(np.mean([r["hit5"] for r in rows])),
        }

    out = {
        "det_model": DET_MODEL, "rec_model": REC_MODEL, "inside_ratio": INSIDE_RATIO,
        "overall": agg(items),
        "plane": agg([r for r in items if not r["is_curved"]]),
        "curved": agg([r for r in items if r["is_curved"]]),
        "unseen": agg([r for r in items if not r["seen"]]),
        "seen": agg([r for r in items if r["seen"]]),
        "by_tokens": {f"tok{k}": agg([r for r in items
                                      if min(len(r["gt"].split()), 3) == k]) for k in (1, 2, 3)},
        "by_length": {f"{lo}-{hi}": agg([r for r in items if lo <= len(r["gt"]) <= hi])
                      for lo, hi in LENGTH_BINS},
    }
    (spec.PROJECT_ROOT / "reports" / "m2_token_merge.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main():
    out = score(run())
    o = out["overall"]
    print(f"\n=== M2 토큰 재결합 (검출 {DET_MODEL} + 인식 {REC_MODEL}) ===")
    print(f"  전체 n={o['n']}  조각 확보율 {o['coverage']*100:.1f}%  "
          f"평균 조각 {o['avg_fragments']:.2f}개  Top-1 {o['top1']*100:.1f}%  Top-5 {o['top5']*100:.1f}%")
    for k in ("plane", "curved", "unseen", "seen"):
        v = out[k]
        if v:
            print(f"  {k:7s} n={v['n']:4d} 확보 {v['coverage']*100:5.1f}% Top-5 {v['top5']*100:5.1f}%")
    print("  토큰수별 Top-5: " + "  ".join(
        f"{k}:{v['top5']*100:.0f}%(n={v['n']})" for k, v in out["by_tokens"].items() if v))
    print("  길이별 Top-5:   " + "  ".join(
        f"{k}:{v['top5']*100:.0f}%" for k, v in out["by_length"].items() if v))
    return out


if __name__ == "__main__":
    main()
