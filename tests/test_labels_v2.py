"""Phase 2 산출물 불변 규칙 — 정규화 라벨·크롭·토큰 경계.

`python -m src.preprocess.build_labels` 를 먼저 실행해야 한다.
"""

import json

import pytest

from src import spec
from src.preprocess import build_labels, normalize

LABEL_DIR = spec.PROJECT_ROOT / "labels"
SNAPSHOT = spec.PROJECT_ROOT / "snapshots" / "label_snapshot_v2.json"

pytestmark = pytest.mark.skipif(not SNAPSHOT.is_file(), reason="build_labels 미실행")


@pytest.fixture(scope="module")
def snap():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rec_lines():
    return [l.split("\t") for l in
            (LABEL_DIR / "rec_all.txt").read_text(encoding="utf-8").splitlines() if l]


def test_no_empty_labels(rec_lines):
    """빈 정답은 학습 타깃이 될 수 없다 (라벨링 실수 1건은 제외 처리됨)."""
    assert [n for n, t in rec_lines if not t.strip()] == []


def test_every_crop_file_exists(rec_lines):
    missing = [n for n, _ in rec_lines if not (spec.PROJECT_ROOT / "crops" / n).is_file()]
    assert missing == []


def test_every_label_char_is_in_dict(rec_lines):
    """사전에 없는 문자가 정답에 있으면 그 박스는 영원히 못 맞힌다."""
    charset = set(normalize.read_dict()) | {" "}
    assert {c for _, t in rec_lines for c in t} - charset == set()


def test_labels_are_normalized(rec_lines):
    assert [t for _, t in rec_lines if normalize.normalize_text(t) != t] == []


def test_original_values_are_preserved(snap):
    """대문자 정규화된 소문자 3건의 원값이 이력으로 남아야 한다."""
    originals = {r["label_original"] for r in snap["records"]
                 if r["label_original"] != r["label"]}
    assert originals == set(spec.KNOWN_LOWERCASE_LABELS)


def test_token_spans_reconstruct_the_label(snap):
    """(a)안 후처리 분리의 전제 — 토큰 경계로 원문 토큰을 복원할 수 있어야 한다."""
    for r in snap["records"]:
        for t in r["tokens"]:
            assert r["label"][t["start"]:t["end"]] == t["token"]
        assert [t["token"] for t in r["tokens"]] == r["label"].split()


def test_all_curved_boxes_survived(snap):
    """곡면 99박스는 실사용 대상이므로 한 건도 유실되면 안 된다."""
    assert snap["stats"]["curved_boxes"] == 99


def test_max_label_length_fits_spec(snap):
    assert snap["stats"]["max_label_length"] <= spec.MAX_TEXT_LENGTH
    assert snap["stats"]["max_label_length"] <= spec.REC_SEQ_LEN


def test_only_the_known_labeling_error_was_dropped(snap):
    assert [d["reason"] for d in snap["stats"]["dropped"]] == ["empty transcription"]


def test_token_spans_helper():
    assert build_labels.token_spans("8196 P32") == [
        {"token": "8196", "start": 0, "end": 4},
        {"token": "P32", "start": 5, "end": 8},
    ]
