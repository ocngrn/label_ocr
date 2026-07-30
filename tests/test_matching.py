"""Phase 4 매칭 불변 규칙 — 골격 정규화 금지, 폐집합 성립, 공백 처리.

`python -m src.matching.build_db` 를 먼저 실행해야 한다.
"""

import json

import pytest

from src import spec
from src.matching import archive, build_db, match
from src.preprocess import split as split_mod

# plan.md Phase 1-1: 원본 대조로 "실존하는 별개 부품 코드"로 확정된 표기쌍.
# 이들이 서로 매칭되면 실존 부품 2개가 붕괴한다.
REAL_CODE_PAIRS = [
    ("8135", "B13S"), ("83", "B3"), ("205", "20S"), ("(5)", "(S)"),
    ("8165", "B16S"), ("501", "S01"), ("925", "92S"), ("P1", "PI"),
    ("L85", "LB5"), ("628", "6Z8"), ("165", "16S"), ("ST035", "ST03S"),
]

pytestmark = pytest.mark.skipif(not build_db.DB_FILE.is_file(), reason="build_db 미실행")


@pytest.fixture(scope="module")
def db():
    return build_db.load()


@pytest.fixture(scope="module")
def matcher(db):
    return match.Matcher(db["serials"])


# --- 규칙 1: 문자 골격 정규화 금지 -------------------------------------------

@pytest.mark.parametrize("a,b", REAL_CODE_PAIRS)
def test_real_code_pairs_are_not_folded(a, b):
    """8135 와 B13S 의 거리가 0 이면 골격 정규화가 들어간 것이다."""
    assert match.car(a, b) < 1.0


@pytest.mark.parametrize("a,b", REAL_CODE_PAIRS)
def test_exact_code_outranks_its_confusable_twin(a, b, matcher, db):
    """예측이 정확할 때 쌍둥이 코드가 1위를 가로채면 안 된다."""
    if a not in set(db["serials"]):
        pytest.skip(f"{a} 는 DB 에 없음")
    top = matcher.top_k([a], k=1)[0]
    assert top[0][0] == a and top[0][1] == pytest.approx(1.0)


def test_confusable_characters_produce_nonzero_distance():
    """0/O, 1/I, 5/S, 8/B, 2/Z 각 축이 매칭에서 구별되어야 한다."""
    for x, y in spec.CONFUSABLE_PAIRS:
        assert match.car(f"A{x}C", f"A{y}C") < 1.0


def test_only_case_is_folded():
    """허용된 정규화는 대소문자 축 하나뿐 (기획서 3.2)."""
    assert match.car("abc-1", "ABC-1") == pytest.approx(1.0)


# --- 공백 처리 (지시서 TASK 3-2) ---------------------------------------------

def test_missing_space_is_not_penalized():
    """다중 토큰 통째 인식 시 공백 탈락은 흔한 실패라 거리 0 으로 흡수한다."""
    m = match.Matcher(["ABC 123", "ABC 124"])
    assert m.top_k(["ABC123"], k=1)[0][0] == ("ABC 123", pytest.approx(1.0))


def test_space_handling_does_not_merge_different_codes():
    """공백 제거 비교가 서로 다른 코드를 같게 만들면 안 된다."""
    m = match.Matcher(["ABC 123", "ABD 123"])
    scores = dict(m.top_k(["ABC123"], k=2)[0])
    assert scores["ABC 123"] > scores["ABD 123"]


# --- 폐집합 성립 조건 (기획서 3.2) -------------------------------------------

def test_db_contains_every_evaluation_label(db):
    """폐집합 Top-K 는 정답이 DB 에 있어야 성립한다 (대표 표기 기준)."""
    records = split_mod.load_records()
    payload = json.loads(
        (spec.PROJECT_ROOT / "splits" / f"split_seed{spec.SPLIT_SEED}.json").read_text(encoding="utf-8"))
    serials, canonical = set(db["serials"]), db["canonical"]
    for s in ("val", "test"):
        members = set(payload["split"][s])
        labels = {r["label"] for r in records if r["image"] in members}
        assert {canonical[l] for l in labels} - serials == set()


def test_every_label_has_a_canonical_form(db):
    records = split_mod.load_records()
    assert {r["label"] for r in records} - set(db["canonical"]) == set()


def test_no_two_db_entries_are_space_equivalent(db):
    """매칭이 공백을 무시하므로, 공백만 다른 항목이 남아 있으면 Top-1 이 임의로 깎인다."""
    keys = [build_db.space_key(s) for s in db["serials"]]
    assert len(keys) == len(set(keys))


def test_whitespace_variants_map_to_one_representative(db):
    """`8196LUM02` 와 `8196 LUM 02` 는 같은 대표 표기로 수렴해야 한다."""
    for spellings in db["variants"].values():
        assert len({db["canonical"][s] for s in spellings}) == 1


def test_canonicalization_only_folds_the_space_axis(db):
    """골격 정규화(8<->B 등)가 병합에 섞여 들어가지 않았는지 확인."""
    for original, rep in db["canonical"].items():
        assert build_db.space_key(original) == build_db.space_key(rep)


def test_db_includes_serials_excluded_from_training(db):
    """'학습에서 뺀다 != DB 에서 뺀다' — 이게 깨지면 폐집합 평가가 무너진다."""
    assert len(db["not_in_train"]) > 0
    assert set(db["not_in_train"]) <= set(db["serials"])


def test_db_size_is_recorded(db):
    """|DB| 가 바뀌면 Top-K 를 직접 비교하면 안 되므로 항상 기록한다."""
    assert db["size"] == len(db["serials"])


# --- 랭킹 동작 ---------------------------------------------------------------

def test_top_k_is_deterministic(matcher):
    a = matcher.top_k(["8I35", "ABC 123"], k=5)
    b = matcher.top_k(["8I35", "ABC 123"], k=5)
    assert a == b


def test_top_k_scores_are_descending(matcher):
    for cands in matcher.top_k(["8135", "IK 351-203", "XXXX"], k=5):
        scores = [s for _, s in cands]
        assert scores == sorted(scores, reverse=True)


def test_top_k_accuracy_counts_hits(matcher, db):
    truths = db["serials"][:20]
    assert match.top_k_accuracy(truths, truths, matcher, k=1) == pytest.approx(1.0)


# --- 예측 아카이빙 규약 (지시서 TASK 3-4) ------------------------------------

def test_archive_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "ARCHIVE_DIR", tmp_path)
    preds = [{"crop": "a.jpg", "gt": "8135", "pred": "8I35",
              "char_confidences": [0.99, 0.42, 0.97, 0.95], "mean_confidence": 0.83}]
    archive.save("unit_test", {"name": "dummy"}, "test", preds)
    got = archive.load("unit_test")
    assert got["predictions"] == preds
    assert got["data"]["split_seed"] == spec.SPLIT_SEED


def test_archive_rejects_records_without_confidences(tmp_path, monkeypatch):
    """확신도를 빠뜨리면 TASK 6-E 의 임계값 튜닝을 소급 적용할 수 없다."""
    monkeypatch.setattr(archive, "ARCHIVE_DIR", tmp_path)
    with pytest.raises(ValueError):
        archive.save("bad", {"name": "d"}, "test", [{"crop": "a.jpg", "gt": "1", "pred": "1"}])
