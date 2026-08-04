"""M2-B GT 없는 공간 그룹핑 — 묶음 조건과 읽기 순서."""

import cv2
import numpy as np

from src.eval.spatial_group import group_fragments


def _q(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


SAME_LINE = [_q(0, 0, 100, 40), _q(120, 0, 220, 40)]   # 높이 40, 간격 20px


def test_adjacent_fragments_on_one_line_are_merged():
    assert group_fragments(SAME_LINE, 0.6) == [[0, 1]]


def test_zero_gap_ratio_keeps_fragments_separate():
    """gap_ratio=0 은 '재결합 안 함' = 기존 파이프라인과 동등해야 한다."""
    assert sorted(group_fragments(SAME_LINE, 0.0)) == [[0], [1]]


def test_different_lines_are_not_merged():
    """수직으로 떨어진 조각은 간격을 아무리 키워도 묶이면 안 된다."""
    assert sorted(group_fragments([_q(0, 0, 100, 40), _q(0, 100, 100, 140)], 3.0)) == [[0], [1]]


def test_reading_order_is_left_to_right_regardless_of_input_order():
    """검출 순서와 무관하게 좌->우로 이어붙여야 한다."""
    assert group_fragments([_q(120, 0, 220, 40), _q(0, 0, 100, 40)], 1.0) == [[1, 0]]


def test_rotated_label_is_merged_along_its_own_axis():
    """부품 사진은 회전된 라벨이 많아 축 정렬 가정을 쓸 수 없다."""
    def rot(pts, deg):
        m = cv2.getRotationMatrix2D((100, 100), deg, 1.0)
        return cv2.transform(np.array([pts]), m)[0]
    frags = [rot(_q(0, 90, 100, 130), 45), rot(_q(120, 90, 220, 130), 45)]
    assert len(group_fragments(frags, 0.6)) == 1


def test_perpendicular_fragments_are_not_merged():
    """방향이 크게 다르면 나란히 있어도 다른 줄이다."""
    def rot(pts, deg):
        m = cv2.getRotationMatrix2D((160, 20), deg, 1.0)
        return cv2.transform(np.array([pts]), m)[0]
    frags = [_q(0, 0, 100, 40), rot(_q(120, 0, 220, 40), 90)]
    assert len(group_fragments(frags, 1.0)) == 2


def test_empty_input():
    assert group_fragments([], 1.0) == []


# --- 0도 / 180도 공존 (파이프·부품이 뒤집혀 함께 찍힌 경우) ------------------

def test_flipped_fragment_reverses_reading_order():
    """180도 라벨은 이미지상 좌->우가 읽기 순서의 역방향이다."""
    assert group_fragments(SAME_LINE, 0.6, flipped=[True, True]) == [[1, 0]]


def test_upright_and_flipped_are_not_merged():
    """같은 줄에 나란히 있어도 읽기 방향이 반대면 다른 부품의 시리얼이다."""
    groups = group_fragments(SAME_LINE, 3.0, flipped=[False, True])
    assert sorted(groups) == [[0], [1]]


def test_flip_flag_does_not_change_grouping_of_a_uniform_line():
    """전부 뒤집힌 줄은 묶임 자체는 동일하고 순서만 뒤집힌다."""
    up = group_fragments(SAME_LINE, 0.6, flipped=[False, False])
    down = group_fragments(SAME_LINE, 0.6, flipped=[True, True])
    assert len(up) == len(down) == 1
    assert up[0] == list(reversed(down[0]))


def test_missing_flip_flags_default_to_upright():
    assert group_fragments(SAME_LINE, 0.6) == group_fragments(SAME_LINE, 0.6, flipped=[False, False])
