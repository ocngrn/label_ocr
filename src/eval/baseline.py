"""TASK 4 — 인식 baseline 실측.

실행: python -m src.eval.baseline

fine-tuning 전 사전학습 모델로 test 셋을 추론해 목표선의 출발점을 확정한다.
예측 원본(텍스트 + 문자별 확신도)은 TASK 3-4 규약대로 아카이빙해, TASK 6-E 에서
매칭 튜닝이 완성된 뒤 Top-K 를 소급 재계산할 수 있게 한다.

지표는 지시서 TASK 4-2 가 요구한 축으로 분해한다:
곡면/평면, 혼동 문자 혼동행렬, Top-K/Top-1. 여기에 Phase 3 에서 추가한
seen/unseen(라벨을 train 에서 봤는지) 축을 더한다 — 전체 평균만 보면
짧은 계열 코드의 암기분 44.4% 만큼 낙관적이기 때문이다.
"""

import collections
import json

import cv2
import numpy as np
from rapidfuzz.distance import Levenshtein

from src import spec
from src.eval.infer import recognize
from src.matching import archive, build_db, match
from src.preprocess import normalize, split as split_mod

CONFUSABLE = {c for pair in spec.CONFUSABLE_PAIRS for c in pair}


def load_test_set(split="test"):
    records = split_mod.load_records()
    payload = json.loads(
        (spec.PROJECT_ROOT / "splits" / f"split_seed{spec.SPLIT_SEED}.json").read_text(encoding="utf-8"))
    members = set(payload["split"][split])
    seen = payload["seen_in_train"][split]
    return [r for r in records if r["image"] in members], seen


def confusion_pairs(pred, truth):
    """혼동 문자 치환을 (정답문자 -> 예측문자) 로 집계.

    편집거리 정렬(opcodes)로 치환 위치를 찾아, 양쪽 모두 혼동축 문자인 경우만 센다.
    이 오인식은 곧 **다른 실존 부품으로의 오분류**를 뜻하므로 최우선 감시 대상이다.

    한계: 길이가 다르고 편집 경로가 동점이면 정렬이 여러 개 가능해 치환 귀속이
    임의로 갈린다(`AIC` -> `A1BC` 에서 `I->1` 대신 `I->B` 로 잡히는 식).
    전체 집계용 지표이므로 수용하되, 개별 건을 근거로 삼지는 않는다.
    """
    out = []
    for op in Levenshtein.opcodes(truth, pred):
        if op.tag != "replace":
            continue
        for gi, pi in zip(range(op.src_start, op.src_end), range(op.dest_start, op.dest_end)):
            g, p = truth[gi], pred[pi]
            if g in CONFUSABLE and p in CONFUSABLE:
                out.append((g, p))
    return out


def _agg(rows, matcher, canonical):
    """부분집합 지표 — CAR / Exact / Top-1 / Top-5."""
    if not rows:
        return None
    preds = [r["pred"] for r in rows]
    truths = [canonical[r["gt"]] for r in rows]
    return {
        "n": len(rows),
        "CAR": float(np.mean([match.car(r["pred"], r["gt"]) for r in rows])),
        "exact": float(np.mean([r["pred"] == r["gt"] for r in rows])),
        "top1": match.top_k_accuracy(preds, truths, matcher, k=1),
        "top5": match.top_k_accuracy(preds, truths, matcher, k=5),
    }


def evaluate(model_name, split="test"):
    from paddleocr import TextRecognition

    records, seen_flags = load_test_set(split)
    db = build_db.load()
    matcher = match.Matcher(db["serials"])
    canonical = db["canonical"]

    images = [cv2.imread(str(spec.PROJECT_ROOT / "crops" / r["crop"])) for r in records]
    model = TextRecognition(model_name=model_name)
    results = recognize(model, images, batch_size=8)

    rows = []
    for rec, (text, confs, mean_conf) in zip(records, results):
        rows.append({
            "crop": rec["crop"],
            "gt": rec["label"],
            "pred": normalize.normalize_text(text),
            "char_confidences": confs,
            "mean_confidence": mean_conf,
            "is_curved": rec["is_curved"],
            "seen": seen_flags[rec["crop"]],
        })

    archive.save(
        run_id=f"baseline_{model_name}_{split}",
        model={"name": model_name, "source": "paddleocr 3.7.0 official weights"},
        split=split,
        predictions=[{k: r[k] for k in
                      ("crop", "gt", "pred", "char_confidences", "mean_confidence")} for r in rows],
        db={"tag": db["tag"], "size": db["size"]},
    )

    confusions = collections.Counter()
    for r in rows:
        confusions.update(confusion_pairs(r["pred"], r["gt"]))

    return {
        "model": model_name,
        "split": split,
        "db_size": db["size"],
        "overall": _agg(rows, matcher, canonical),
        "plane": _agg([r for r in rows if not r["is_curved"]], matcher, canonical),
        "curved": _agg([r for r in rows if r["is_curved"]], matcher, canonical),
        "unseen": _agg([r for r in rows if not r["seen"]], matcher, canonical),
        "seen": _agg([r for r in rows if r["seen"]], matcher, canonical),
        "confusions": {f"{g}->{p}": n for (g, p), n in confusions.most_common()},
        "confusion_total": sum(confusions.values()),
    }


def main(models=("PP-OCRv4_server_rec", "PP-OCRv5_server_rec", "PP-OCRv6_medium_rec")):
    out = {}
    for name in models:
        m = evaluate(name)
        out[name] = m
        o, c, u = m["overall"], m["curved"], m["unseen"]
        print(f"\n{name}  (|DB|={m['db_size']})")
        print(f"  전체   n={o['n']:3d} CAR={o['CAR']:.4f} Exact={o['exact']*100:5.1f}% "
              f"Top-1={o['top1']*100:5.1f}% Top-5={o['top5']*100:5.1f}%")
        print(f"  평면   n={m['plane']['n']:3d} CAR={m['plane']['CAR']:.4f} Exact={m['plane']['exact']*100:5.1f}%")
        print(f"  곡면   n={c['n']:3d} CAR={c['CAR']:.4f} Exact={c['exact']*100:5.1f}%")
        print(f"  unseen n={u['n']:3d} CAR={u['CAR']:.4f} Exact={u['exact']*100:5.1f}% "
              f"Top-5={u['top5']*100:5.1f}%")
        print(f"  seen   n={m['seen']['n']:3d} CAR={m['seen']['CAR']:.4f} Exact={m['seen']['exact']*100:5.1f}%")
        print(f"  혼동문자 치환 {m['confusion_total']}건: {list(m['confusions'].items())[:8]}")

    path = spec.PROJECT_ROOT / "reports" / "baseline_rec_metrics.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    main()
