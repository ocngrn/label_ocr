"""DB 유도 병합의 분할 규칙 검증 (GPU 불필요)."""

from src.eval.db_merge import best_partition

CHAIN = [0, 1, 2]
TEXTS = {0: "8196", 1: "P32", 2: "C6B"}


def test_merges_only_what_the_db_supports():
    """`8196 P32` 만 DB 에 있으면 그것만 묶고 `C6B` 는 남긴다.

    기하 규칙으로는 원리적으로 불가능한 구분이다 (m2b_spatial_group.md §1).
    """
    score = lambda t: 1.0 if t == "8196 P32" else 0.5
    assert best_partition(CHAIN, TEXTS, score) == [[0, 1], [2]]


def test_ties_split_rather_than_merge():
    """근거가 동등하면 묶지 않는다.

    M2-B 에서 확신 없이 묶었더니 단일 토큰이 67% -> 29% 로 무너졌다.
    단일 토큰이 전체의 75%라 잘못된 병합의 손실이 이득보다 크다.
    """
    assert best_partition(CHAIN, TEXTS, lambda t: 0.5) == [[0], [1], [2]]


def test_merges_whole_chain_when_that_is_the_best_hit():
    score = lambda t: 1.0 if t == "8196 P32 C6B" else 0.3
    assert best_partition(CHAIN, TEXTS, score) == [[0, 1, 2]]


def test_margin_suppresses_marginal_merges():
    """문턱을 올리면 근거가 약한 병합부터 사라진다."""
    score = lambda t: 0.6 if " " in t else 0.5      # 병합이 근소하게 유리
    assert best_partition(CHAIN, TEXTS, score, margin=0.0) == [[0, 1, 2]]
    assert best_partition(CHAIN, TEXTS, score, margin=0.2) == [[0], [1], [2]]


def test_objective_is_length_weighted():
    """단순 합은 쪼갤수록, 단순 평균은 합칠수록 커진다. 길이 가중이 그 편향을 없앤다.

    모든 구간의 점수가 같으면 어떤 분할이든 목적함수 값이 동일해야 하고,
    그때는 동점 규칙(쪼개기)이 결정한다.
    """
    parts = best_partition(CHAIN, TEXTS, lambda t: 0.7)
    assert parts == [[0], [1], [2]]
