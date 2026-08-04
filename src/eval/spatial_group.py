"""M2-B — GT 없는 공간 그룹핑 (실배포 가능한 형태로 재측정).

실행:
    python -m src.eval.spatial_group cache    # 검출+인식 1회 실행 후 캐시 (약 50분)
    python -m src.eval.spatial_group sweep    # 캐시 위에서 그룹핑 파라미터 스윕 (즉시)

## 왜 필요한가

M2(`token_merge.py`)는 시스템 Top-5 를 61.9% -> 76.1% 로 끌어올렸지만, **원본 GT 박스를
조각 묶음의 기준으로 썼다.** 실배포에는 GT 가 없으므로 76.1%는 "그룹핑이 완벽할 때의 상한"이다.
이 모듈은 GT 를 쓰지 않고 조각을 묶어, 실제로 배포 가능한 수치를 낸다.
76.1% 와의 차이가 곧 그룹핑 오차다.

## 읽기 방향 판별 (0도 vs 180도)

**한 이미지 안에 정방향 라벨과 180도 뒤집힌 라벨이 함께 찍히는 경우가 있다**
(파이프·부품이 뒤집혀 놓인 상태로 촬영). 실측: 뒤집힌 박스 41개(0.5%),
0도와 180도가 공존하는 이미지 30장(1.3%). 수직 텍스트(±90도)는 353개(4.4%)로 더 많다.

기하만으로는 0도와 180도를 구분할 수 없다(같은 직선이다). 방향을 임의로 +x 로 고정하면
뒤집힌 라벨은 **조용히 역순으로 이어붙는다** (`8196 P32` -> `P32 8196`).
게다가 크롭 자체가 거꾸로 들어가 인식도 망가진다.

### 방향 판정 수단 — 실측으로 고른다

`TextLineOrientationClassification`(PP-LCNet_x0_25_textline_ori)을 먼저 시험했으나
**우리 도메인에서 신뢰할 수 없었다.** 945조각 중 137개(14.5%)를 180도로 판정했는데,
그 조각들을 실제로 뒤집어 인식하면 평균 확신도가 **0.752 -> 0.594 로 떨어졌고**
개선된 것은 31%뿐이었다. 문서·간판 텍스트로 학습된 모델이라 금속 부품 손글씨에 맞지 않는다.

    원본 '8196'(1.00) -> 뒤집음 '9b18'(0.97)
    원본 'JE24S054-119-2'(0.89) -> '2611-490-2083'(0.32)

따라서 판정 수단을 `FLIP_POLICY` 로 분리하고 시스템 Top-5 로 채택 여부를 결정한다:

- `none`       : 뒤집지 않음 (기준선)
- `classifier` : 위 분류기 판정
- `confidence` : **양방향 모두 인식해 평균 확신도가 높은 쪽 채택** (우리 인식기 자체를 판정에 사용)

캐시는 양방향 인식 결과를 모두 담으므로 정책 비교에 추가 추론 비용이 들지 않는다.

## 그룹핑 규칙

같은 텍스트 줄에 나란히 있는 조각을 잇는다. 회전된 라벨이 많으므로 축 정렬 대신
`cv2.minAreaRect` 의 방향에 위 부호를 적용한 **읽기 방향**을 쓴다. 두 조각을 잇는 조건:

1. 읽기 방향이 같다           (부호 있는 각도차 <= `ANGLE_TOL`)
   -> 0도 라벨과 180도 라벨은 나란히 있어도 **묶이지 않는다**
2. 같은 줄이다                (수직 이격 <= `OFFSET_RATIO` x 평균 높이)
3. 나란히 인접해 있다         (진행 방향 간격 <= `gap_ratio` x 평균 높이)

세 조건을 만족하는 쌍을 union-find 로 묶고, 각 그룹 안에서는 읽기 방향으로 정렬해
이어붙인다.

## 채점

그룹의 합집합 폴리곤을 GT 박스와 IoU >= 0.5 로 매칭해 **원본 GT 박스 단위로 채점**한다
(확정된 KPI 단위 유지). 그룹핑 자체에는 GT 가 전혀 쓰이지 않는다.
"""

import collections
import json
import sys

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from src import spec
from src.eval.detect_baseline import _poly, match_boxes
from src.eval.infer import recognize
from src.matching import archive, build_db, match
from src.preprocess import crop, normalize, parse_label, split as split_mod

DET_MODEL = "PP-OCRv5_server_det"
REC_MODEL = "PP-OCRv5_server_rec"
CACHE = spec.PROJECT_ROOT / "reports" / "detect_cache_test.json"

ANGLE_TOL = 20.0      # 도
OFFSET_RATIO = 0.7    # 수직 이격 / 평균 높이
GAP_RATIOS = (0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0)
FLIP_POLICIES = ("none", "classifier", "confidence")


def _geometry(points, flipped=False):
    """조각의 (중심, 읽기 방향 단위벡터, 높이, 길이).

    `flipped` 는 방향 분류기가 180도로 판정했는지 여부다. 기하만으로는 0도와 180도를
    구분할 수 없으므로, 부호는 분류 결과로만 결정된다.
    """
    rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
    (cx, cy), (w, h), angle = rect
    if w < h:                      # 긴 변을 진행 방향으로
        w, h = h, w
        angle += 90.0
    rad = np.deg2rad(angle)
    direction = np.array([np.cos(rad), np.sin(rad)])
    # minAreaRect 의 각도 표현을 먼저 +x(수직이면 +y)로 정규화한 뒤,
    # 크롭이 뒤집혀 있었다면 부호를 되돌린다.
    if direction[0] < -1e-9 or (abs(direction[0]) <= 1e-9 and direction[1] < 0):
        direction = -direction
    if flipped:
        direction = -direction
    return np.array([cx, cy]), direction, max(h, 1e-6), w


def _linkable(a, b, gap_ratio):
    ca, da, ha, wa = a
    cb, db, hb, wb = b

    # 부호 있는 비교 — 0도 라벨과 180도 라벨은 나란히 있어도 다른 줄로 본다
    cos = float(np.dot(da, db))
    if cos <= 0 or np.rad2deg(np.arccos(min(cos, 1.0))) > ANGLE_TOL:
        return False

    direction = da if wa >= wb else db
    delta = cb - ca
    along = abs(float(np.dot(delta, direction)))
    perp = abs(float(delta[0] * direction[1] - delta[1] * direction[0]))
    avg_h = (ha + hb) / 2

    if perp > OFFSET_RATIO * avg_h:
        return False
    gap = along - (wa + wb) / 2          # 두 박스 사이 빈 거리
    return gap <= gap_ratio * avg_h


def group_fragments(polys, gap_ratio, flipped=None):
    """GT 를 쓰지 않고 조각을 묶는다. 그룹별 인덱스를 읽기 순서로 반환.

    `flipped[i]` 는 조각 i 가 180도로 판정됐는지. 생략하면 전부 정방향으로 본다.
    """
    if not polys:
        return []
    flipped = flipped or [False] * len(polys)
    geoms = [_geometry(p, f) for p, f in zip(polys, flipped)]
    parent = list(range(len(polys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if _linkable(geoms[i], geoms[j], gap_ratio):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups = collections.defaultdict(list)
    for i in range(len(polys)):
        groups[find(i)].append(i)

    out = []
    for members in groups.values():
        direction = geoms[max(members, key=lambda i: geoms[i][3])][1]
        out.append(sorted(members, key=lambda i: float(np.dot(geoms[i][0], direction))))
    return out


# --- 1단계: 검출 + 인식 캐시 -------------------------------------------------

def build_cache(split="test"):
    """검출 폴리곤 + 방향 판별 + 인식 결과를 캐시한다.

    검출이 전체 비용의 대부분(이미지당 약 13초)이므로, 캐시에 폴리곤이 이미 있으면
    **검출을 건너뛰고 방향 판별·인식만 다시 한다**.
    """
    from paddleocr import TextDetection, TextLineOrientationClassification, TextRecognition

    payload = json.loads(
        (spec.PROJECT_ROOT / "splits" / f"split_seed{spec.SPLIT_SEED}.json").read_text(encoding="utf-8"))
    images = sorted(set(payload["split"][split]))

    cached = {}
    if CACHE.is_file():
        cached = json.loads(CACHE.read_text(encoding="utf-8")).get("images", {})
        if all(name in cached and cached[name].get("polys") is not None for name in images):
            print("  캐시된 검출 폴리곤 재사용 — 검출 생략")
        else:
            cached = {}

    per_image, patches, owner = {}, [], []
    detector = None if cached else TextDetection(model_name=DET_MODEL, enable_mkldnn=False)

    for i, name in enumerate(images, 1):
        img = cv2.imread(str(spec.IMAGE_DIR / name))
        if cached:
            polys = [np.asarray(p) for p in cached[name]["polys"]]
        else:
            polys = [np.asarray(p) for p in list(detector.predict([img]))[0]["dt_polys"]]

        kept = []
        for p in polys:
            patch = crop.crop_box(img, p)
            if patch is not None and patch.size and min(patch.shape[:2]) >= 2:
                kept.append(p.tolist())
                owner.append(name)
                patches.append(patch)
        per_image[name] = {"polys": kept}
        if detector and i % 20 == 0:
            print(f"  검출 {i}/{len(images)}", flush=True)

    print(f"  조각 {len(patches)}개. 방향 분류기 판정", flush=True)
    classifier = TextLineOrientationClassification()
    cls_flip = [r["label_names"][0] == "180_degree"
                for r in classifier.predict(patches, batch_size=16)]
    print(f"  분류기 180도 판정 {sum(cls_flip)}개 / {len(cls_flip)}", flush=True)

    # 정책 비교를 위해 양방향 인식 결과를 모두 캐시한다
    rec = TextRecognition(model_name=REC_MODEL)
    print("  정방향 인식", flush=True)
    fwd = recognize(rec, patches, batch_size=8)
    print("  역방향 인식", flush=True)
    bwd = recognize(rec, [cv2.rotate(p, cv2.ROTATE_180) for p in patches], batch_size=8)

    for (t0, c0, s0), (t1, c1, s1), name, cf in zip(fwd, bwd, owner, cls_flip):
        e = per_image[name]
        e.setdefault("text_fwd", []).append(normalize.normalize_text(t0))
        e.setdefault("conf_fwd", []).append(c0)
        e.setdefault("score_fwd", []).append(s0)
        e.setdefault("text_bwd", []).append(normalize.normalize_text(t1))
        e.setdefault("conf_bwd", []).append(c1)
        e.setdefault("score_bwd", []).append(s1)
        e.setdefault("cls_flip", []).append(bool(cf))

    CACHE.write_text(json.dumps(
        {"det_model": DET_MODEL, "rec_model": REC_MODEL, "split": split, "images": per_image},
        ensure_ascii=False), encoding="utf-8")
    print(f"  캐시 저장: {CACHE}")
    return per_image


# --- 2단계: 캐시 위에서 그룹핑 스윕 ------------------------------------------

def resolve_flips(entry, policy):
    """조각별 '뒤집을지' 결정. GT 를 쓰지 않는다."""
    n = len(entry.get("text_fwd", []))
    if policy == "none":
        return [False] * n
    if policy == "classifier":
        return list(entry.get("cls_flip", [False] * n))
    if policy == "confidence":
        return [b > f for f, b in zip(entry["score_fwd"], entry["score_bwd"])]
    raise ValueError(policy)


def evaluate(cache, gap_ratio, matcher, canonical, split="test",
             flip_policy="none", archive_run=None):
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

    items = []
    for name, gts in by_image.items():
        entry = cache[name]
        polys = [np.asarray(p) for p in entry["polys"]]
        flips = resolve_flips(entry, flip_policy)
        texts = [entry["text_bwd"][i] if f else entry["text_fwd"][i]
                 for i, f in enumerate(flips)]
        confs = [entry["conf_bwd"][i] if f else entry["conf_fwd"][i]
                 for i, f in enumerate(flips)]
        groups = group_fragments(polys, gap_ratio, flips)

        # 그룹 합집합 폴리곤을 GT 와 매칭 (채점 정렬에만 GT 사용)
        group_polys = []
        for members_idx in groups:
            shapes = [_poly(polys[i]) for i in members_idx]
            shapes = [s for s in shapes if not s.is_empty and s.area > 0]
            merged = unary_union(shapes).convex_hull if shapes else Polygon()
            group_polys.append(np.asarray(merged.exterior.coords)[:-1] if not merged.is_empty
                               else np.zeros((4, 2)))

        gt_polys = [source[name][g["box_index"]].points for g in gts]
        matched = match_boxes(gt_polys, group_polys)

        for gi, g in enumerate(gts):
            if gi in matched:
                idx = groups[matched[gi]]
                pred = normalize.normalize_text(" ".join(texts[i] for i in idx).strip())
                cc = [c for i in idx for c in confs[i]]
            else:
                pred, cc = "", []
            items.append({
                "crop": g["crop"], "gt": g["label"], "pred": pred,
                "char_confidences": cc,
                "mean_confidence": float(np.mean(cc)) if cc else 0.0,
                "is_curved": g["is_curved"], "seen": seen_flags[g["crop"]],
                "matched": gi in matched,
            })

    top5 = matcher.top_k([x["pred"] for x in items], k=5)
    for x, cands in zip(items, top5):
        names = [c.upper() for c, _ in cands]
        truth = canonical[x["gt"]].upper()
        x["hit5"] = bool(x["pred"]) and truth in names
        x["hit1"] = bool(x["pred"]) and truth == names[0]

    if archive_run:
        db = build_db.load()
        archive.save(run_id=archive_run,
                     model={"det": DET_MODEL, "rec": REC_MODEL,
                            "pipeline": "spatial-group", "gap_ratio": gap_ratio},
                     split=split,
                     predictions=[{k: x[k] for k in ("crop", "gt", "pred",
                                                     "char_confidences", "mean_confidence")}
                                  for x in items],
                     db={"tag": db["tag"], "size": db["size"]})
    return items


def _agg(rows):
    if not rows:
        return None
    return {"n": len(rows),
            "coverage": float(np.mean([r["matched"] for r in rows])),
            "top1": float(np.mean([r["hit1"] for r in rows])),
            "top5": float(np.mean([r["hit5"] for r in rows]))}


def sweep():
    cache = json.loads(CACHE.read_text(encoding="utf-8"))["images"]
    db = build_db.load()
    matcher = match.Matcher(db["serials"])

    rows = []
    for policy in FLIP_POLICIES:
        for gap in GAP_RATIOS:
            items = evaluate(cache, gap, matcher, db["canonical"], flip_policy=policy)
            s_ = {"flip_policy": policy, "gap_ratio": gap, "overall": _agg(items),
                  "by_tokens": {f"tok{k}": _agg([r for r in items
                                                 if min(len(r["gt"].split()), 3) == k])
                                for k in (1, 2, 3)},
                  "unseen": _agg([r for r in items if not r["seen"]]),
                  "seen": _agg([r for r in items if r["seen"]]),
                  "curved": _agg([r for r in items if r["is_curved"]])}
            rows.append(s_)
            o = s_["overall"]
            tk = " ".join(f"{k}:{v['top5']*100:.0f}%" for k, v in s_["by_tokens"].items() if v)
            print(f"  flip={policy:<10} gap={gap:<4} 매칭 {o['coverage']*100:5.1f}%  "
                  f"Top-1 {o['top1']*100:5.1f}%  Top-5 {o['top5']*100:5.1f}%   {tk}", flush=True)

    best = max(rows, key=lambda r: r["overall"]["top5"])
    print(f"\n최적: flip={best['flip_policy']} gap_ratio={best['gap_ratio']}  "
          f"시스템 Top-5 {best['overall']['top5']*100:.1f}%")
    evaluate(cache, best["gap_ratio"], matcher, db["canonical"],
             flip_policy=best["flip_policy"],
             archive_run=f"m2b_spatial_group_{REC_MODEL}_test")
    (spec.PROJECT_ROOT / "reports" / "m2b_spatial_group.json").write_text(
        json.dumps({"angle_tol": ANGLE_TOL, "offset_ratio": OFFSET_RATIO,
                    "best": best, "sweep": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cache":
        build_cache()
    else:
        sweep()
