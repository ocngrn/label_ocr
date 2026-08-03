"""end-to-end 시스템 KPI 측정 — 검출 → 크롭 → 인식 → DB 매칭.

실행: python -m src.eval.end_to_end

## 왜 필요한가

TASK 4 까지의 인식 지표는 **GT 크롭을 제공한 조건**에서 잰 것이다(Top-5 87.0%).
그러나 기획서 3.3의 Top-K 목표선은 계층이 "시스템"이므로, 검출 실패까지 포함한
end-to-end 로 재야 한다. 검출 recall 71.1% 를 곱하면 실제는 57.4% 수준으로 추정됐고,
이 모듈은 그 추정치를 **실측으로 대체**한다.

## 채점 규칙 (박스 단위)

GT 박스 하나하나가 평가 단위다. GT 박스가

1. 검출되지 않았으면          -> 시스템 실패 (예측을 빈 문자열로 간주)
2. 검출됐으나 인식이 틀렸고
   매칭 Top-K 에도 정답이 없으면 -> 시스템 실패
3. 매칭 Top-K 에 정답이 있으면  -> 성공

따라서 오답 유형을 (a) 검출 실패 (b) 인식+매칭 실패 로 분해할 수 있다
(지시서 TASK 7-5).

## 비용

검출은 로컬 CPU 에서 이미지당 약 11초다(oneDNN 경로가 깨져 `enable_mkldnn=False` 강제).
검출을 **한 번만 돌려 크롭을 캐시**하고, 인식 모델 여러 개를 그 위에서 비교한다.
"""

import collections
import json

import cv2
import numpy as np

from src import spec
from src.eval.detect_baseline import match_boxes
from src.eval.infer import recognize
from src.matching import archive, build_db, match
from src.preprocess import crop, normalize, parse_label, split as split_mod

DET_MODEL = "PP-OCRv5_server_det"
REC_MODELS = ("PP-OCRv4_server_rec", "PP-OCRv5_server_rec")
LENGTH_BINS = ((1, 2), (3, 4), (5, 8), (9, 14), (15, 40))


def run_detection(split="test"):
    """검출을 돌려 GT 박스별 크롭(또는 미검출)을 만든다. 가장 비싼 단계라 결과를 반환해 재사용."""
    from paddleocr import TextDetection

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
    items, n_pred_total = [], 0

    for i, name in enumerate(sorted(by_image), 1):
        img = cv2.imread(str(spec.IMAGE_DIR / name))
        preds = [np.asarray(p) for p in list(detector.predict([img]))[0]["dt_polys"]]
        n_pred_total += len(preds)

        gts = by_image[name]
        matched = match_boxes([source[name][g["box_index"]].points for g in gts], preds)

        for gi, g in enumerate(gts):
            patch = None
            if gi in matched:
                patch = crop.crop_box(img, preds[matched[gi]])
                if patch is None or patch.size == 0 or min(patch.shape[:2]) < 2:
                    patch = None
            items.append({
                "crop": g["crop"], "gt": g["label"], "is_curved": g["is_curved"],
                "seen": seen_flags[g["crop"]], "detected": patch is not None, "patch": patch,
            })

        if i % 20 == 0:
            print(f"  검출 {i}/{len(by_image)}", flush=True)

    print(f"  검출 완료: GT {len(items)}박스, 예측 {n_pred_total}건, "
          f"검출 성공 {sum(x['detected'] for x in items)}박스", flush=True)
    return items


def score(items, rec_model_name, matcher, canonical, split="test"):
    from paddleocr import TextRecognition

    detected = [x for x in items if x["detected"]]
    model = TextRecognition(model_name=rec_model_name)
    results = recognize(model, [x["patch"] for x in detected], batch_size=8)

    for x, (text, confs, mean_conf) in zip(detected, results):
        x["pred"] = normalize.normalize_text(text)
        x["char_confidences"] = confs
        x["mean_confidence"] = mean_conf
    for x in items:
        if not x["detected"]:
            x["pred"], x["char_confidences"], x["mean_confidence"] = "", [], 0.0

    # 미검출 박스는 빈 예측이라 매칭에서 자동으로 실패 처리된다
    top5 = matcher.top_k([x["pred"] for x in items], k=5)
    for x, cands in zip(items, top5):
        names = [c.upper() for c, _ in cands]
        truth = canonical[x["gt"]].upper()
        x["hit5"] = x["detected"] and truth in names
        x["hit1"] = x["detected"] and truth == names[0]

    archive.save(
        run_id=f"e2e_{rec_model_name}_{split}",
        model={"det": DET_MODEL, "rec": rec_model_name, "pipeline": "end-to-end"},
        split=split,
        predictions=[{k: x[k] for k in
                      ("crop", "gt", "pred", "char_confidences", "mean_confidence")} for x in items],
        db={"tag": "serial_db_proxy_v1", "size": len(matcher)},
    )
    return items


def summarize(items):
    def agg(rows):
        if not rows:
            return None
        return {
            "n": len(rows),
            "det_recall": float(np.mean([r["detected"] for r in rows])),
            "top1": float(np.mean([r["hit1"] for r in rows])),
            "top5": float(np.mean([r["hit5"] for r in rows])),
            "fail_detection": sum(1 for r in rows if not r["detected"]),
            "fail_recognition": sum(1 for r in rows if r["detected"] and not r["hit5"]),
        }

    out = {
        "overall": agg(items),
        "plane": agg([r for r in items if not r["is_curved"]]),
        "curved": agg([r for r in items if r["is_curved"]]),
        "unseen": agg([r for r in items if not r["seen"]]),
        "seen": agg([r for r in items if r["seen"]]),
        "by_length": {},
    }
    for lo, hi in LENGTH_BINS:
        out["by_length"][f"{lo}-{hi}"] = agg([r for r in items if lo <= len(r["gt"]) <= hi])
    return out


def main(split="test"):
    db = build_db.load()
    matcher = match.Matcher(db["serials"])

    items = run_detection(split)
    report = {"det_model": DET_MODEL, "db_size": db["size"], "split": split, "rec": {}}

    for rec_name in REC_MODELS:
        scored = score(items, rec_name, matcher, db["canonical"], split)
        s = summarize(scored)
        report["rec"][rec_name] = s
        o = s["overall"]
        print(f"\n{rec_name}  end-to-end (검출 {DET_MODEL})")
        print(f"  전체 n={o['n']}  검출 {o['det_recall']*100:.1f}%  "
              f"시스템 Top-1 {o['top1']*100:.1f}%  Top-5 {o['top5']*100:.1f}%")
        print(f"    오답 분해: 검출 실패 {o['fail_detection']}  인식+매칭 실패 {o['fail_recognition']}")
        for k in ("plane", "curved", "unseen", "seen"):
            v = s[k]
            if v:
                print(f"  {k:7s} n={v['n']:4d} 검출 {v['det_recall']*100:5.1f}% "
                      f"Top-5 {v['top5']*100:5.1f}%")
        print("  길이별 Top-5: " + "  ".join(
            f"{k}:{v['top5']*100:.0f}%(n={v['n']})" for k, v in s["by_length"].items() if v))

    path = spec.PROJECT_ROOT / "reports" / "end_to_end_metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
