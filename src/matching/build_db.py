"""평가용 시리얼 DB 프록시 생성 (지시서 TASK 3-1).

실행: python -m src.matching.build_db

## 구성 원칙

- **정제 후 고유 시리얼 전량**의 합집합. 부품 마스터 목록은 미확보라 라벨 유래값만 쓴다.
- **그룹 분할로 학습에서 제외된 시리얼도 반드시 포함**한다. 폐집합 Top-K 는
  "정답이 DB 에 존재"를 전제하므로, 학습에서 뺐다고 DB 에서 빼면 평가가 성립하지 않는다
  (기획서 3.2 — "학습에서 뺀다 != DB 에서 뺀다").
- |DB| 를 스냅샷에 기록한다. K=5 의 난이도는 후보 공간 크기에 종속되므로
  |DB| 가 바뀌면 Top-K 수치를 직접 비교하지 않는다.

## 공백 정규화 — 표기 흔들림을 하나의 항목으로 병합

라벨에는 **같은 코드가 공백만 다르게 표기된 경우**가 있다. 실측 99군집 207개:

    ' 8196 LUM 02' / '8196 LUM 02' / '8196 LUM02' / '8196LUM 02' / '8196LUM02'
    'EA 8049 P4C-04' / 'EA 8049P4C-04' / 'EA8049 P4C-04' / 'EA8049P4C-04'

매칭 함수는 지시서 TASK 3-2 지침대로 공백을 무시하므로, 이들은 **매칭 관점에서
구별이 불가능한 중복**이다. 그대로 두면 Top-1 이 동점 후보 중 임의로 하나를 골라
모델 성능과 무관하게 최대 4.8%(207/4279)까지 깎인다 — 측정 오류이지 성능이 아니다.
실제 부품 마스터라면 한 부품에 5가지 표기가 있을 리도 없다.

따라서 **공백 제거형을 키로 병합**하고, 대표 표기는 라벨에서 가장 자주 등장한 표기로
정한다. |DB| 4,279 -> **4,171**. 병합 이력은 `variants` 에 보존한다.

주의: 이는 **공백 축만 접는 것**이며 골격 정규화(8<->B 등)와 무관하다.
지시서가 공백 무시 비교를 기본 탑재하라고 한 것과 동일한 근거다.

## 알려진 한계

실제 부품 마스터가 아니라 라벨에 등장한 고유 텍스트 전체다. 따라서 `(1)`, `3`, `902`
같은 수량·순번 표기도 후보 공간에 들어간다. 이는 Top-K 를 실제보다 **어렵게** 만든다
(`901`/`902`/`903`, `(1)`/`(7)` 처럼 편집거리 1 이웃이 조밀해진다). 부품 마스터를
확보하면 `serial_db_proxy_v2` 로 교체하고 Top-K 목표선을 재보정한다.
"""

import collections
import json
from datetime import datetime, timezone

from src import spec
from src.preprocess import split as split_mod

DB_TAG = "serial_db_proxy_v1"
DB_FILE = spec.PROJECT_ROOT / "snapshots" / f"{DB_TAG}.json"


def space_key(label: str) -> str:
    """공백 표기 흔들림을 흡수하는 병합 키."""
    return label.replace(" ", "").upper()


def build():
    records = split_mod.load_records()
    freq = collections.Counter(r["label"] for r in records)

    clusters = collections.defaultdict(list)
    for label in {r["label"] for r in records}:
        clusters[space_key(label)].append(label)

    # 대표 표기 = 라벨에서 가장 자주 등장한 표기 (동률은 사전순으로 결정적 선택)
    canonical = {}
    for key, spellings in clusters.items():
        rep = max(sorted(spellings), key=lambda s: freq[s])
        for s in spellings:
            canonical[s] = rep

    serials = sorted(set(canonical.values()))

    split_file = spec.PROJECT_ROOT / "splits" / f"split_seed{spec.SPLIT_SEED}.json"
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    train_images = set(payload["split"]["train"])
    train_serials = {canonical[r["label"]] for r in records if r["image"] in train_images}

    merged = {k: sorted(v) for k, v in clusters.items() if len(v) > 1}
    db = {
        "tag": DB_TAG,
        "created": datetime.now(timezone.utc).isoformat(),
        "source": "label_snapshot_v2 고유 라벨 (부품 마스터 미확보)",
        "size": len(serials),
        "raw_unique_labels": len(canonical),
        "merged_clusters": len(merged),
        "not_in_train": sorted(set(serials) - train_serials),
        "serials": serials,
        "canonical": canonical,
        "variants": merged,
    }
    DB_FILE.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"|DB| = {len(serials)}  (원 고유 라벨 {len(canonical)} 에서 "
          f"공백 표기 흔들림 {len(merged)}군집 병합)")
    print(f"  학습에 없는 시리얼 {len(db['not_in_train'])}개 포함 (폐집합 성립 조건)")
    return db


def load():
    return json.loads(DB_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    build()
