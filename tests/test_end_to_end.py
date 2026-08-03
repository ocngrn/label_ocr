"""end-to-end 채점 규칙 불변 테스트 — 모델 없이 로직만 검증."""

import pytest

from src.eval import end_to_end


def _item(gt, pred, detected=True, curved=False, seen=False, hit1=False, hit5=False):
    return {"crop": f"{gt}.jpg", "gt": gt, "pred": pred, "detected": detected,
            "is_curved": curved, "seen": seen, "hit1": hit1, "hit5": hit5}


def test_undetected_box_counts_as_system_failure():
    """검출이 놓친 GT 박스는 인식이 아무리 좋아도 시스템 실패다."""
    s = end_to_end.summarize([_item("ABC", "", detected=False)])
    assert s["overall"]["top5"] == 0.0
    assert s["overall"]["fail_detection"] == 1
    assert s["overall"]["fail_recognition"] == 0


def test_failure_decomposition_is_exclusive():
    """오답은 검출 실패 / 인식+매칭 실패 중 하나로만 귀속된다 (지시서 TASK 7-5)."""
    items = [_item("A", "", detected=False),
             _item("B", "X", detected=True, hit5=False),
             _item("C", "C", detected=True, hit5=True)]
    o = end_to_end.summarize(items)["overall"]
    assert o["fail_detection"] + o["fail_recognition"] + round(o["top5"] * o["n"]) == o["n"]


def test_detection_recall_is_measured_over_gt_boxes():
    items = [_item("A", "A", detected=True, hit5=True), _item("B", "", detected=False)]
    assert end_to_end.summarize(items)["overall"]["det_recall"] == 0.5


def test_curved_is_reported_separately(): 
    """지시서 규칙 4 — 곡면을 통합 지표에 합산하지 않고 분리 보고."""
    items = [_item("A", "A", hit5=True), _item("B", "B", curved=True, hit5=False)]
    s = end_to_end.summarize(items)
    assert s["plane"]["n"] == 1 and s["curved"]["n"] == 1
    assert s["plane"]["top5"] == 1.0 and s["curved"]["top5"] == 0.0


def test_seen_unseen_split_covers_all_items():
    items = [_item("A", "A", seen=True), _item("B", "B", seen=False)]
    s = end_to_end.summarize(items)
    assert s["seen"]["n"] + s["unseen"]["n"] == s["overall"]["n"]


def test_length_buckets_partition_every_box():
    items = [_item("A", "A"), _item("ABCD", "ABCD"), _item("A" * 12, "A" * 12)]
    s = end_to_end.summarize(items)
    assert sum(v["n"] for v in s["by_length"].values() if v) == len(items)


def test_empty_subset_reports_none_not_zero():
    """표본이 없는 구간을 0%로 보고하면 성능 저하로 오독된다."""
    s = end_to_end.summarize([_item("A", "A", hit5=True)])
    assert s["curved"] is None
