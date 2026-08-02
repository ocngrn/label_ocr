"""Phase 5 평가 로직 불변 규칙 — 혼동 문자 집계, 검출 매칭, 아카이빙 충실도."""

import json

import numpy as np
import pytest

from src import spec
from src.eval import baseline, detect_baseline

ARCHIVE = spec.PROJECT_ROOT / "predictions"


# --- 혼동 문자 집계 (지시서 TASK 4-2 최우선 감시) ----------------------------

def test_confusion_detects_skeleton_substitution():
    """8135 -> B13S 는 8->B, 5->S 두 축의 오인식이다."""
    assert sorted(baseline.confusion_pairs("B13S", "8135")) == [("5", "S"), ("8", "B")]


def test_confusion_ignores_non_confusable_substitutions():
    """혼동축이 아닌 치환(C->D)은 이 지표에 섞이면 안 된다."""
    assert baseline.confusion_pairs("ADC", "ACC") == []


def test_confusion_handles_length_mismatch():
    """삽입·삭제가 섞여도 치환 위치만 골라내야 한다."""
    assert baseline.confusion_pairs("IABC", "1AB") == [("1", "I")]


def test_confusion_ignores_pure_insertion():
    assert baseline.confusion_pairs("A1CX", "A1C") == []


def test_confusion_is_directional():
    """0->O 와 O->0 은 다른 오류다 (혼동행렬이므로 방향 보존)."""
    assert baseline.confusion_pairs("O", "0") == [("0", "O")]
    assert baseline.confusion_pairs("0", "O") == [("O", "0")]


# --- 검출 IoU 매칭 -----------------------------------------------------------

def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_identical_boxes_match():
    assert detect_baseline.match_boxes([_box(0, 0, 10, 10)], [_box(0, 0, 10, 10)]) == {0: 0}


def test_low_overlap_does_not_match():
    """IoU 0.5 미만이면 검출 실패로 센다."""
    assert detect_baseline.match_boxes([_box(0, 0, 10, 10)], [_box(7, 0, 17, 10)]) == {}


def test_matching_is_one_to_one():
    """예측 하나가 GT 두 개를 동시에 만족시킬 수 없다."""
    gts = [_box(0, 0, 10, 10), _box(0, 0, 11, 11)]
    matched = detect_baseline.match_boxes(gts, [_box(0, 0, 10, 10)])
    assert len(matched) == 1 and len(set(matched.values())) == 1


def test_best_iou_wins_when_contested():
    gts = [_box(0, 0, 10, 10)]
    preds = [_box(0, 0, 8, 10), _box(0, 0, 10, 10)]
    assert detect_baseline.match_boxes(gts, preds) == {0: 1}


def test_self_intersecting_polygon_does_not_crash():
    """곡면 폴리곤에는 자기교차가 있을 수 있다 (기획서 5장 3항)."""
    bowtie = [[0, 0], [10, 10], [10, 0], [0, 10]]
    detect_baseline.match_boxes([bowtie], [_box(0, 0, 10, 10)])


# --- 아카이빙 충실도 (지시서 TASK 3-4) ---------------------------------------

@pytest.mark.skipif(not ARCHIVE.exists(), reason="baseline 미실행")
@pytest.mark.parametrize("run", ["baseline_PP-OCRv5_server_rec_test"])
def test_archived_predictions_carry_per_char_confidences(run):
    """평균만 남기면 TASK 6-E 의 임계값 튜닝을 소급 적용할 수 없다."""
    path = ARCHIVE / f"{run}.json"
    if not path.is_file():
        pytest.skip(f"{run} 미생성")
    preds = json.loads(path.read_text(encoding="utf-8"))["predictions"]
    assert preds
    for p in preds:
        assert len(p["char_confidences"]) == len(p["pred"])
        assert p["mean_confidence"] == pytest.approx(
            float(np.mean(p["char_confidences"])) if p["char_confidences"] else 0.0, abs=1e-3)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="baseline 미실행")
def test_archive_records_the_db_it_was_scored_against():
    """|DB| 가 바뀌면 Top-K 를 직접 비교하면 안 되므로 함께 남긴다 (기획서 3.2)."""
    path = ARCHIVE / "baseline_PP-OCRv5_server_rec_test.json"
    if not path.is_file():
        pytest.skip("baseline 미생성")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["db"]["tag"] and data["db"]["size"] > 0
    assert data["data"]["split_seed"] == spec.SPLIT_SEED
