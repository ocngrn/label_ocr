"""Phase 2 산출물 — 정규화 라벨, 검출/인식 라벨 파일, 인식용 크롭, label_snapshot_v2.

실행: python -m src.preprocess.build_labels

박스 분할 표준: **(a)안 — 공백 다중 토큰을 한 박스로 유지**하고 후처리로 분리한다
(지시서 TASK 1-3 권장안). 토큰 경계를 메타데이터로 남겨 TASK 7-4 공백 오류율 추적에 대비한다.
"""

import collections
import json
from datetime import datetime, timezone

import cv2
import numpy as np

from src import spec
from src.preprocess import crop, normalize, parse_label

SNAPSHOT_TAG = "label_snapshot_v2"
CROP_DIR = spec.PROJECT_ROOT / "crops"
LABEL_DIR = spec.PROJECT_ROOT / "labels"


def token_spans(text: str):
    """공백 구분 토큰과 각 토큰의 [시작, 끝) 문자 오프셋."""
    spans, i = [], 0
    for token in text.split(" "):
        if token:
            start = text.index(token, i)
            spans.append({"token": token, "start": start, "end": start + len(token)})
            i = start + len(token)
    return spans


def build():
    CROP_DIR.mkdir(exist_ok=True)
    LABEL_DIR.mkdir(exist_ok=True)
    samples = parse_label.parse()

    det_lines, rec_lines, records = [], [], []
    dropped, aspect = [], []

    for s in samples:
        img = None
        det_boxes = []
        for idx, b in enumerate(s.boxes):
            text = normalize.normalize_text(b.transcription)
            if not text.strip():
                # 라벨링 실수로 확인된 빈 transcription — 검출·인식 양쪽에서 제외
                dropped.append({"image": s.image, "box": idx, "reason": "empty transcription"})
                continue

            det_boxes.append({"transcription": text, "points": b.points, "difficult": b.difficult})

            if img is None:
                img = cv2.imread(str(spec.IMAGE_DIR / s.image))
            patch = crop.crop_box(img, b.points)
            if patch is None or patch.size == 0 or min(patch.shape[:2]) < 2:
                dropped.append({"image": s.image, "box": idx, "reason": "degenerate crop"})
                continue

            name = f"{s.image.rsplit('.', 1)[0]}_{idx:02d}.jpg"
            cv2.imwrite(str(CROP_DIR / name), patch)
            rec_lines.append(f"{name}\t{text}")
            aspect.append(patch.shape[1] / patch.shape[0])
            records.append({
                "crop": name,
                "image": s.image,
                "box_index": idx,
                "label": text,
                "label_original": b.transcription,          # 원값 이력 보존
                "n_points": len(b.points),
                "is_curved": b.is_curved,
                "crop_size": [patch.shape[1], patch.shape[0]],
                "tokens": token_spans(text),                 # (a)안 후처리 분리용
            })

        if det_boxes:
            det_lines.append(f"{s.image}\t{json.dumps(det_boxes, ensure_ascii=False)}")

    (LABEL_DIR / "det_all.txt").write_text("\n".join(det_lines) + "\n", encoding="utf-8", newline="\n")
    (LABEL_DIR / "rec_all.txt").write_text("\n".join(rec_lines) + "\n", encoding="utf-8", newline="\n")

    ar = np.array(aspect)
    stats = {
        "images": len(det_lines),
        "boxes": len(records),
        "curved_boxes": sum(1 for r in records if r["is_curved"]),
        "multi_token_boxes": sum(1 for r in records if len(r["tokens"]) > 1),
        "dropped": dropped,
        "max_label_length": max(len(r["label"]) for r in records),
        # Phase 5 합성 크롭이 실사 분포에 맞춰야 하므로 종횡비를 기록한다
        "crop_aspect_ratio": {
            "mean": float(ar.mean()), "p05": float(np.percentile(ar, 5)),
            "p50": float(np.percentile(ar, 50)), "p95": float(np.percentile(ar, 95)),
            "max": float(ar.max()),
        },
    }

    (spec.PROJECT_ROOT / "snapshots" / f"{SNAPSHOT_TAG}.json").write_text(
        json.dumps({
            "tag": SNAPSHOT_TAG,
            "created": datetime.now(timezone.utc).isoformat(),
            "based_on": "label_snapshot_v1",
            "box_split_policy": "a: keep multi-token in one box, split in post-processing",
            "crop_method": {"4pt": "get_rotate_crop_image", "5pt+": "get_minarea_rect_crop"},
            "rec_image_shape": list(spec.REC_IMAGE_SHAPE),
            "stats": stats,
            "records": records,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"images={stats['images']} boxes={stats['boxes']} curved={stats['curved_boxes']} "
          f"multi_token={stats['multi_token_boxes']} dropped={len(dropped)}")
    print(f"max_label_length={stats['max_label_length']} "
          f"aspect p05/p50/p95={ar_fmt(stats['crop_aspect_ratio'])}")
    return stats


def ar_fmt(d):
    return f"{d['p05']:.1f}/{d['p50']:.1f}/{d['p95']:.1f}"


if __name__ == "__main__":
    build()
