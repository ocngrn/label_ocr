"""불변 규칙 테스트 (지시서 TASK 0-5) — 이후 상시 실행.

이 테스트들은 "한 번 확인하고 리포트로 남기는 것"이 아니라, 라벨·사전·규격이
나중에 조용히 어긋나는 것을 잡기 위한 회귀 방지 장치다.
"""

import pytest

from src import spec
from src.preprocess import normalize, parse_label

# plan.md Phase 1-1: 자동 탐지에 남았으나 원본 대조로 "실존하는 별개 부품 코드"로 확정된 12쌍.
# 이들이 서로 같아지면 실존 코드가 붕괴한다.
REAL_CODE_PAIRS = [
    ("8135", "B13S"), ("83", "B3"), ("205", "20S"), ("(5)", "(S)"),
    ("8165", "B16S"), ("501", "S01"), ("925", "92S"), ("P1", "PI"),
    ("L85", "LB5"), ("628", "6Z8"), ("165", "16S"), ("ST035", "ST03S"),
]


@pytest.fixture(scope="module")
def samples():
    return parse_label.parse()


@pytest.fixture(scope="module")
def texts(samples):
    return [b.transcription for _, b in parse_label.iter_boxes(samples)]


@pytest.fixture(scope="module")
def charset(texts):
    return normalize.build_charset(texts)


# --- 1. 문자 사전 ------------------------------------------------------------

def test_dict_has_no_lowercase(charset):
    """문자 사전은 0-9 + A-Z + 기호. 소문자 a-z 는 포함하지 않는다 (plan.md 3-2)."""
    assert [c for c in charset if c.islower()] == []


def test_dict_has_no_space(charset):
    """공백은 use_space_char=True 가 자동 추가하므로 사전에 넣으면 인덱스가 중복된다."""
    assert " " not in charset


def test_dict_covers_every_observed_char(texts, charset):
    """정규화 후 등장하는 모든 문자가 사전에 있어야 한다.

    빠진 문자가 있으면 해당 박스는 구조적으로 정답이 불가능해진다
    (실제로 아포스트로피 `'` 36회가 지시서 문자셋에서 누락돼 있었다).
    """
    observed = {c for t in texts for c in normalize.normalize_text(t)} - {" "}
    assert observed - set(charset) == set()


def test_dict_file_roundtrips(charset, tmp_path):
    """디스크에 쓴 사전이 그대로 읽혀야 한다 (개행·인코딩 사고 방지)."""
    path = tmp_path / "dict.txt"
    normalize.write_dict(charset, path)
    assert normalize.read_dict(path) == charset


# --- 2. 골격 정규화 금지 (규칙 1) --------------------------------------------

@pytest.mark.parametrize("a,b", REAL_CODE_PAIRS)
def test_normalize_does_not_fold_real_code_pairs(a, b):
    """8135 와 B13S 는 정규화 후에도 서로 달라야 한다."""
    assert normalize.normalize_text(a) != normalize.normalize_text(b)


@pytest.mark.parametrize("x,y", spec.CONFUSABLE_PAIRS)
def test_confusable_characters_stay_distinct(x, y):
    """8/B, 5/S, 0/O, 1/I, 2/Z 는 각각 별개 문자로 보존된다."""
    assert normalize.normalize_text(x) != normalize.normalize_text(y)


def test_confusable_characters_all_present_in_dict(charset):
    """구별축 문자가 사전에서 빠지면 구별 자체가 불가능해진다."""
    for x, y in spec.CONFUSABLE_PAIRS:
        assert x in charset and y in charset


# --- 3. 대문자 정규화 --------------------------------------------------------

def test_only_known_lowercase_labels_are_changed(texts):
    """정규화로 값이 바뀌는 라벨은 확정된 소문자 3건뿐이어야 한다."""
    changed = {t for t in texts if normalize.normalize_text(t) != t}
    assert changed == set(spec.KNOWN_LOWERCASE_LABELS)


def test_original_value_is_preserved(texts):
    """정규화는 순수 함수 — 원본 리스트를 변형하지 않는다 (원값 이력 보존 전제)."""
    before = list(texts)
    [normalize.normalize_text(t) for t in texts]
    assert texts == before


def test_normalize_is_idempotent(texts):
    once = [normalize.normalize_text(t) for t in texts]
    assert [normalize.normalize_text(t) for t in once] == once


# --- 4. 라벨 ↔ 이미지 정합성 -------------------------------------------------

def test_every_label_path_resolves_to_a_file(samples):
    """라벨의 모든 이미지 경로가 실제 파일로 해석되어야 한다 (크롭·학습이 조용히 깨짐)."""
    missing = [s.image for s in samples if not (spec.IMAGE_DIR / s.image).is_file()]
    assert missing == []


def test_no_unlabeled_images(samples):
    labeled = {s.image for s in samples}
    on_disk = {p.name for p in spec.IMAGE_DIR.glob("*.jpg")}
    assert on_disk - labeled == set()


# --- 5. 데이터 ↔ 인식기 규격 (Phase 0 §5) ------------------------------------

def test_max_label_length_fits_sequence_length(texts):
    """CTC 는 타임스텝 >= 라벨 길이여야 한다.

    짧으면 에러 없이 손실만 계산되어, 긴 코드만 영원히 못 맞히는 상태로 수렴한다.
    """
    longest = max(len(normalize.normalize_text(t)) for t in texts)
    assert longest <= spec.REC_SEQ_LEN
    assert longest <= spec.MAX_TEXT_LENGTH


def test_sequence_length_matches_input_width():
    """Phase 0 실측 관계: CTC 타임스텝 = 입력 폭 // 4."""
    assert spec.REC_SEQ_LEN == spec.REC_IMAGE_SHAPE[2] // 4
