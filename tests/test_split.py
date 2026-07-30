"""Phase 3 분할 불변 규칙 — 누수 방지·곡면 층화·재현성.

`python -m src.preprocess.split` 를 먼저 실행해야 한다.
"""

import json

import pytest

from src import spec
from src.preprocess import split as split_mod

SPLIT_FILE = spec.PROJECT_ROOT / "splits" / f"split_seed{spec.SPLIT_SEED}.json"

pytestmark = pytest.mark.skipif(not SPLIT_FILE.is_file(), reason="split 미실행")


@pytest.fixture(scope="module")
def payload():
    return json.loads(SPLIT_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records():
    return split_mod.load_records()


def test_splits_are_disjoint_and_complete(payload, records):
    sets = [set(payload["split"][s]) for s in split_mod.SPLITS]
    assert sets[0] & sets[1] == set() and sets[0] & sets[2] == set() and sets[1] & sets[2] == set()
    assert set().union(*sets) == {r["image"] for r in records}


def test_no_identifying_serial_leaks_between_train_and_test(payload):
    """핵심 게이트 — 식별 시리얼이 train/test 에 동시 등장하면 test 성능이 과대평가된다."""
    assert payload["leakage"]["shared_identifying"] == []


def test_shared_labels_are_all_short_family_codes(payload, records):
    """train/test 가 공유하는 라벨은 계열·수량 코드(4자 이하)뿐이어야 한다."""
    train = {r["label"] for r in records if r["image"] in set(payload["split"]["train"])}
    test = {r["label"] for r in records if r["image"] in set(payload["split"]["test"])}
    assert all(len(l) < spec.IDENTIFYING_MIN_LEN for l in train & test)


def test_test_contains_mostly_unseen_labels(payload):
    """그룹 분할의 목적 — test 다수가 학습에서 못 본 코드여야 일반화를 측정한다."""
    lk = payload["leakage"]
    assert lk["test_labels_unseen_in_train"] / lk["test_unique_labels"] > 0.5


def test_nothing_is_excluded_from_the_dataset(payload, records):
    """임계값은 '그룹화 여부'만 정한다 — 짧은 코드도 전부 학습·평가에 들어간다."""
    assert sum(payload["stats"][s]["boxes"] for s in split_mod.SPLITS) == len(records)


def test_short_family_codes_appear_in_both_train_and_test(payload, records):
    """`8166` 같은 계열 코드는 의도적으로 분산시킨다 (도메인 어휘 학습 허용)."""
    where = {i: s for s in split_mod.SPLITS for i in payload["split"][s]}
    per_label = {}
    for r in records:
        per_label.setdefault(r["label"], set()).add(where[r["image"]])
    assert {"train", "test"} <= per_label["8166"]


def test_seen_flags_cover_every_eval_crop(payload, records):
    """지표를 seen/unseen 으로 분해하려면 평가 크롭 전부에 플래그가 있어야 한다."""
    for s in ("val", "test"):
        crops = {r["crop"] for r in records if r["image"] in set(payload["split"][s])}
        assert set(payload["seen_in_train"][s]) == crops


def test_seen_flags_agree_with_leakage_counts(payload):
    flags = payload["seen_in_train"]["test"]
    assert sum(flags.values()) == payload["leakage"]["test_crops_seen_label"]
    assert sum(not v for v in flags.values()) == payload["leakage"]["test_crops_unseen_label"]


def test_every_seen_label_is_a_short_code(payload, records):
    """seen 으로 표시된 크롭의 라벨은 전부 4자 이하여야 한다 (식별 시리얼 누수 0의 따름정리)."""
    by_crop = {r["crop"]: r["label"] for r in records}
    seen = [by_crop[c] for c, v in payload["seen_in_train"]["test"].items() if v]
    assert all(len(l) < spec.IDENTIFYING_MIN_LEN for l in seen)


def test_ratios_are_close_to_target(payload):
    total = sum(payload["stats"][s]["images"] for s in split_mod.SPLITS)
    for s in split_mod.SPLITS:
        actual = payload["stats"][s]["images"] / total
        assert abs(actual - spec.SPLIT_RATIOS[s]) < 0.02


def test_every_split_has_curved_images(payload):
    """train 곡면이 0이면 rectification 학습 신호가 사라진다 (지시서 TASK 2-2)."""
    for s in split_mod.SPLITS:
        assert payload["stats"][s]["curved_images"] > 0


def test_all_curved_boxes_are_distributed(payload):
    assert sum(payload["stats"][s]["curved_boxes"] for s in split_mod.SPLITS) == 99


# --- 곡면 K-Fold (지시서 TASK 2-3 게이트) ------------------------------------

def test_curved_folds_partition_the_curved_images(payload, records):
    folds = payload["curved_folds"]
    flat = [i for f in folds for i in f]
    assert len(flat) == len(set(flat))
    assert set(flat) == {r["image"] for r in records if r["is_curved"]}


def test_curved_folds_are_balanced(payload):
    sizes = [len(f) for f in payload["curved_folds"]]
    assert len(sizes) == spec.CURVED_N_FOLDS
    assert max(sizes) - min(sizes) <= 2


# --- 재현성 (기획서 7장) -----------------------------------------------------

def test_split_is_reproducible_from_seed(records, payload):
    curved = {r["image"] for r in records if r["is_curved"]}
    comps = split_mod.build_components(records)
    again = split_mod.assign(comps, curved)
    assert {s: list(again[s]) for s in split_mod.SPLITS} == payload["split"]


def test_label_files_match_the_split(payload):
    for s in split_mod.SPLITS:
        det = (spec.PROJECT_ROOT / "labels" / f"det_{s}.txt").read_text(encoding="utf-8").splitlines()
        rec = (spec.PROJECT_ROOT / "labels" / f"rec_{s}.txt").read_text(encoding="utf-8").splitlines()
        assert len([l for l in det if l]) == payload["stats"][s]["images"]
        assert len([l for l in rec if l]) == payload["stats"][s]["boxes"]
