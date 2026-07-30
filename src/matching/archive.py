"""예측 원본 아카이빙 규약 (지시서 TASK 3-4).

## 왜 필요한가

매칭 튜닝(신뢰도 가중·미등록 임계값)은 실제 모델의 확신도 분포가 있어야 의미가 있어
TASK 6-E 로 미뤄져 있다. 그런데 Top-K 는 주 KPI 라 TASK 4 baseline 부터 재야 한다.
이 둘이 충돌하지 않으려면 **추론 시점에 텍스트 예측값 + 문자별 확신도를 원본 그대로
저장**해두어야 한다. 그러면 TASK 6-E 에서 매칭이 완성된 뒤 baseline·fine-tuned
양쪽에 **소급 적용해 Top-K 를 재계산**할 수 있다.

따라서 TASK 4(baseline)와 TASK 6(fine-tuned) 추론은 반드시 이 포맷으로 저장한다.

## 포맷

```json
{
  "run_id": "baseline_ppocrv4_test",
  "created": "2026-07-31T...",
  "model": {"name": "PP-OCRv4_server_rec", "weights_sha256": "..."},
  "data": {"snapshot": "label_snapshot_v2", "split": "test", "split_seed": 42},
  "db": {"tag": "serial_db_proxy_v1", "size": 4279},
  "predictions": [
    {
      "crop": "20230909_093111_00.jpg",
      "gt": "10",
      "pred": "1O",
      "char_confidences": [0.99, 0.61],   # 문자별 softmax 확신도
      "mean_confidence": 0.80
    }
  ]
}
```

`char_confidences` 는 미등록 판정 임계값 튜닝(TASK 6-E)의 입력이므로 **평균만 저장하지
말고 문자별 원본을 남긴다** — 평균만 남기면 "한 글자만 확신이 낮은" 경우를 복원할 수 없다.
"""

import json
from datetime import datetime, timezone

from src import spec

ARCHIVE_DIR = spec.PROJECT_ROOT / "predictions"
REQUIRED_FIELDS = ("crop", "gt", "pred", "char_confidences", "mean_confidence")


def save(run_id, model, split, predictions, db=None):
    """예측 원본을 규약대로 저장하고 경로를 반환."""
    missing = [f for f in REQUIRED_FIELDS if predictions and f not in predictions[0]]
    if missing:
        raise ValueError(f"예측 레코드에 필수 필드 누락: {missing}")

    ARCHIVE_DIR.mkdir(exist_ok=True)
    path = ARCHIVE_DIR / f"{run_id}.json"
    path.write_text(json.dumps({
        "run_id": run_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "data": {"snapshot": "label_snapshot_v2", "split": split, "split_seed": spec.SPLIT_SEED},
        "db": db,
        "predictions": predictions,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load(run_id):
    return json.loads((ARCHIVE_DIR / f"{run_id}.json").read_text(encoding="utf-8"))
