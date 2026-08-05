"""DB 유도 조각 병합 — 기하 대신 시리얼 DB 를 병합 판정자로 쓴다.

실행: python -m src.eval.db_merge

## 왜 이걸 시도하는가

M2-B 는 **기하 규칙**으로 조각을 묶으려다 실패했다(+1.0%p). 실패 원인은 명확했다:

    한 부품 표면의 GT = ['28', 'P32', 'C6B', '8196P32', '15']
    `8196 P32` 는 한 박스인데 바로 옆 `P32` 는 별개다.

간격·정렬로는 이 둘을 구별할 정보가 **원리적으로** 없다. 그러나 **DB 에는 있다.**
`8196P32` 는 시리얼 DB(4,171개)에 있고 `P32C6B` 는 없다. 우리가 이미 가진 자산인데
병합 판정에는 써본 적이 없다.

## 방법

1. 기하로는 **후보 사슬**만 만든다 (`group_fragments` 를 넉넉한 `gap_ratio` 로).
   여기서는 "묶을지"가 아니라 "묶일 수 있는 이웃인지"만 정한다.
2. 사슬을 **연속 구간으로 분할**하는 방법 중 DB 적합도가 가장 높은 것을 고른다.

목적함수는 **글자수 가중 DB 점수**다:

    maximize  Σ_segment  score(segment) × len(segment)

이어붙여도 총 글자수는 변하지 않으므로 분할 방식이 달라도 비교가 성립한다.
단순 합은 조각을 많이 쪼갤수록 커지고(구간마다 ≤1 이 더해짐), 단순 평균은 반대로
합칠수록 커진다. 길이 가중은 **글자 하나하나가 자기 구간의 매칭 품질만큼 기여**하게 해
그 편향을 없앤다.

최적 분할은 사슬 길이에 대한 O(n²) DP 로 정확히 구한다. 사슬은 대개 5조각 이하다.

## 비용

후보 문자열 전체를 모아 `Matcher.distances` **1회 호출**로 점수화한다.
조합마다 부르면 4,171개 시리얼에 대한 편집거리 계산이 반복돼 느려진다.
"""

import json

import numpy as np

from src import spec
from src.eval import spatial_group
from src.matching import build_db, match
from src.preprocess import normalize

# 후보 사슬을 만들 때만 쓰는 값. "묶는다"가 아니라 "이웃 후보"를 정하는 용도라
# M2-B 의 최적값(0.0)보다 넉넉하게 잡는다 — 실제 병합 여부는 DB 가 정한다.
CANDIDATE_GAP = 1.5


def _segments(chain):
    """사슬의 모든 연속 구간 (i, j) — 반열림 [i, j)."""
    return [(i, j) for i in range(len(chain)) for j in range(i + 1, len(chain) + 1)]


def _text_of(texts, chain, i, j):
    return normalize.normalize_text(" ".join(texts[k] for k in chain[i:j]).strip())


def best_partition(chain, texts, score_of, margin=0.0):
    """글자수 가중 DB 점수를 최대화하는 연속 분할. 구간 리스트를 읽기 순서로 반환.

    **동점이면 더 잘게 쪼갠다** (`>=` 로 뒤쪽 i 가 이긴다). M2-B 에서 확신 없이 묶었더니
    단일 토큰 정확도가 67% → 29% 로 무너졌다. 근거가 동등하면 묶지 않는 쪽이 안전하다.
    """
    n = len(chain)
    best = [0.0] * (n + 1)
    cut = [0] * (n + 1)
    for j in range(1, n + 1):
        best[j] = -1.0
        for i in range(j):
            t = _text_of(texts, chain, i, j)
            # 조각을 둘 이상 묶는 구간에만 `margin` 만큼의 문턱을 물린다. 근거가 어중간할 때
            # 묶어서 잃는 쪽(단일 토큰)이 얻는 쪽(다중 토큰)보다 3배 많기 때문이다.
            penalty = margin if j - i > 1 else 0.0
            gain = best[i] + (score_of(t) - penalty) * len(t.replace(" ", ""))
            # 엡실론 없이 `>=` 만 쓰면 부동소수점이 동점 규칙을 뒤집는다:
            #   0.7*10 = 7.000000000000001  vs  0.7*4 + 0.7*6 = 7.0
            # 수학적으로 같은데 병합 쪽이 미세하게 이겨 "동점이면 쪼갠다"가 무력화된다.
            if gain >= best[j] - 1e-9:
                best[j] = gain
                cut[j] = i
    out, j = [], n
    while j > 0:
        i = cut[j]
        out.append(chain[i:j])
        j = i
    return out[::-1]


def make_grouper(matcher, candidate_gap=CANDIDATE_GAP, margin=0.0):
    """`evaluate(grouper=...)` 에 넣을 그룹퍼. 후보 점수는 이미지마다 1회 배치 조회."""

    def grouper(polys, texts, flips):
        chains = spatial_group.group_fragments(polys, candidate_gap, flips)

        cand = []
        for chain in chains:
            for i, j in _segments(chain):
                cand.append(_text_of(texts, chain, i, j))
        uniq = sorted(set(cand))
        if uniq:
            dist = matcher.distances(uniq)
            score = {t: float(1.0 - dist[n].min()) for n, t in enumerate(uniq)}
        else:
            score = {}

        out = []
        for chain in chains:
            out.extend(best_partition(chain, texts, lambda t: score.get(t, 0.0), margin))
        return out

    return grouper


def run(split="test"):
    cache = json.loads(spatial_group.CACHE.read_text(encoding="utf-8"))["images"]
    db = build_db.load()
    matcher = match.Matcher(db["serials"])
    canonical = db["canonical"]

    rows = []
    for tag, kwargs in (
        ("기하 gap=0.0 (M2-B 최적)", {"gap_ratio": 0.0}),
        ("기하 gap=1.5", {"gap_ratio": CANDIDATE_GAP}),
        ("DB 유도 병합", {"gap_ratio": 0.0, "grouper": make_grouper(matcher)}),
    ):
        items = spatial_group.evaluate(cache, matcher=matcher, canonical=canonical,
                                       split=split, flip_policy="none", **kwargs)
        agg = spatial_group._agg(items)
        by_tok = {f"tok{k}": spatial_group._agg(
            [r for r in items if min(len(r["gt"].split()), 3) == k]) for k in (1, 2, 3)}
        rows.append({"tag": tag, "overall": agg, "by_tokens": by_tok,
                     "unseen": spatial_group._agg([r for r in items if not r["seen"]]),
                     "seen": spatial_group._agg([r for r in items if r["seen"]])})
        tk = " ".join(f"{k}:{v['top5']*100:.0f}%" for k, v in by_tok.items() if v)
        print(f"  {tag:24s} 매칭 {agg['coverage']*100:5.1f}%  "
              f"Top-1 {agg['top1']*100:5.1f}%  Top-5 {agg['top5']*100:5.1f}%   {tk}", flush=True)

    out = {"split": split, "candidate_gap": CANDIDATE_GAP,
           "db": {"tag": db["tag"], "size": db["size"]}, "results": rows}
    (spec.PROJECT_ROOT / "reports" / "db_merge.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    base = rows[0]["overall"]["top5"]
    got = rows[-1]["overall"]["top5"]
    print(f"\nDB 유도 병합: {got*100:.1f}%  (기하 최적 대비 {(got-base)*100:+.1f}%p)")
    print("참고 — M2-B 기록: 기하 62.9% / GT 로 묶은 상한 76.1%")
    return out


if __name__ == "__main__":
    run()
