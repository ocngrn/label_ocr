"""M2 토큰 재결합 — 조각 귀속·읽기 순서 불변 규칙."""

import numpy as np

from src.eval.token_merge import assign_and_order


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


GT_WIDE = [[0, 0], [300, 0], [300, 50], [0, 50]]


def test_fragments_are_returned_in_reading_order():
    """검출 순서와 무관하게 GT 장축 방향(좌->우)으로 정렬되어야 한다."""
    preds = [_q(210, 5, 290, 45), _q(10, 5, 90, 45), _q(110, 5, 190, 45)]
    assert assign_and_order(GT_WIDE, preds) == [1, 2, 0]


def test_fragment_outside_gt_is_excluded():
    assert assign_and_order(GT_WIDE, [_q(400, 5, 480, 45)]) == []


def test_fragment_straddling_the_border_is_excluded():
    """조각 면적의 절반 미만만 GT 안에 있으면 다른 박스의 것으로 본다."""
    assert assign_and_order(GT_WIDE, [_q(280, 5, 380, 45)]) == []


def test_fragment_fully_inside_is_kept_even_with_low_iou():
    """조각은 GT 보다 작아 IoU 가 낮다 — 그래서 IoU 가 아니라 포함률로 귀속한다."""
    tiny = _q(10, 20, 40, 30)          # GT 면적의 2% (IoU 0.02)
    assert assign_and_order(GT_WIDE, [tiny]) == [0]


def test_degenerate_gt_returns_nothing():
    assert assign_and_order([[0, 0], [0, 0], [0, 0], [0, 0]], [_q(0, 0, 10, 10)]) == []


def test_vertical_gt_orders_along_its_long_axis():
    """세로로 긴 박스는 위->아래가 읽기 순서다."""
    gt = [[0, 0], [50, 0], [50, 300], [0, 300]]
    preds = [_q(5, 210, 45, 290), _q(5, 10, 45, 90)]
    assert assign_and_order(gt, preds) == [1, 0]
