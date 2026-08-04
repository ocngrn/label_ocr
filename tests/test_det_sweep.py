"""후처리 스윕의 누수 방지·선택 규칙 검증 (GPU 없이 가능한 부분)."""

import pytest

from src.eval import detect_baseline
from src.eval.det_sweep_trained import pick
from src.preprocess import split as split_mod


def test_default_target_set_is_test_only():
    """학습한 모델을 곡면 증강 대상에서 재면 평가 박스의 23%가 학습 데이터가 된다."""
    payload_test = set(detect_baseline.targets_for("test"))
    records = split_mod.load_records()
    curved = {r["image"] for r in records if r["is_curved"]}
    assert not (payload_test & (curved - payload_test)), "test 기본 대상에 외부 곡면이 섞였다"

    augmented = set(detect_baseline.targets_for("test", extra_curved=True))
    assert augmented > payload_test, "extra_curved=True 는 곡면을 더 끌어와야 한다"


def test_extra_curved_pulls_in_training_images():
    """곡면 증강이 왜 학습 모델에 위험한지를 수치로 고정한다."""
    records = split_mod.load_records()
    train = set(detect_baseline.targets_for("train"))
    augmented = set(detect_baseline.targets_for("test", extra_curved=True))
    leaked = augmented & train
    assert leaked, "곡면 증강은 실제로 train 이미지를 끌어온다 — 이 사실이 바뀌면 문서를 고칠 것"
    assert len(leaked) > 50, f"train 유입이 {len(leaked)}장 — 기록된 75장과 크게 다르다"


@pytest.mark.parametrize("floor,expected", [
    (0.50, 0.80),   # 하한이 낮으면 recall 최대 조합
    (0.60, 0.70),   # 하한을 올리면 precision 을 지키는 것 중 최대
])
def test_pick_respects_precision_floor(floor, expected):
    rows = [
        {"plane_recall": 0.80, "precision": 0.52},
        {"plane_recall": 0.70, "precision": 0.61},
        {"plane_recall": 0.60, "precision": 0.90},
    ]
    assert pick(rows, floor)["plane_recall"] == expected


def test_pick_returns_none_when_floor_unreachable():
    """recall 만 보면 항상 답이 나오지만, 하한을 못 지키면 '없음'이 정직한 답이다."""
    assert pick([{"plane_recall": 0.9, "precision": 0.4}], 0.9) is None
