"""시리얼 DB 매칭 — 정규화 편집거리 기반 Top-K 랭킹 (지시서 TASK 3-2).

## 절대 규칙: 문자 골격 정규화 금지

`8<->B, 5<->S, 0<->O, 1<->I, 2<->Z` 를 **매칭 함수에 넣지 않는다.**
`8135` 와 `B13S`, `83` 과 `B3` 가 둘 다 실존 부품 코드로 확정됐으므로
(plan.md Phase 1-1, 원본 이미지 전수 대조), 이들을 접으면 실존 코드 2개가
서로의 최상위 오답으로 강제 매칭되어 Top-1 이 붕괴한다.

허용되는 정규화는 **대소문자 축 하나뿐**이다.

## 공백 처리

다중 토큰을 통째로 인식하면(TASK 1 (a)안) CTC/attention 모두 공백 경계를 놓치기 쉽다
(`ABC 123` -> `ABC123`). 따라서 **원문 비교와 공백 제거 비교를 모두 계산해 더 짧은
편집거리를 채택**한다(지시서 TASK 3-2 기본 탑재 옵션). TASK 7-4 의 공백 경계
오류율이 임계를 넘으면 TASK 1 (b)안(개별 박스 분할)으로 전환한다.

## 동점 처리 (잠정)

편집거리가 같은 후보는 현재 **후보 문자열 사전순**으로 결정적 정렬한다.
신뢰도 가중은 실제 모델의 확신도 분포가 있어야 의미가 있으므로 TASK 6-E 로 미룬다
(지시서 TASK 3 하단 "미룬 항목"). 그때 이 함수의 tie-break 만 교체한다.
"""

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist


def _strip(s: str) -> str:
    return s.replace(" ", "")


class Matcher:
    """DB 를 한 번 전처리해두고 여러 예측을 배치로 매칭한다."""

    def __init__(self, serials):
        self.serials = list(serials)
        self._upper = [s.upper() for s in self.serials]
        self._stripped = [_strip(s) for s in self._upper]

    def __len__(self):
        return len(self.serials)

    def distances(self, predictions) -> np.ndarray:
        """(예측 수 x |DB|) 정규화 편집거리 행렬. 원문/공백제거 중 작은 값."""
        preds = [p.upper() for p in predictions]
        raw = cdist(preds, self._upper, scorer=Levenshtein.normalized_distance, workers=-1)
        nospace = cdist([_strip(p) for p in preds], self._stripped,
                        scorer=Levenshtein.normalized_distance, workers=-1)
        return np.minimum(raw, nospace)

    def top_k(self, predictions, k=5):
        """예측별 상위 k 후보를 [(시리얼, 점수)] 로 반환. 점수 = 1 - 정규화 편집거리."""
        dist = self.distances(predictions)
        k = min(k, len(self.serials))
        out = []
        for row in dist:
            # 거리 오름차순, 동점은 시리얼 사전순 (결정적 재현성)
            idx = sorted(range(len(row)), key=lambda i: (row[i], self._upper[i]))[:k]
            out.append([(self.serials[i], float(1.0 - row[i])) for i in idx])
        return out


def top_k_accuracy(predictions, truths, matcher, k=5):
    """정답이 상위 k 후보에 드는 비율. Top-1 은 k=1."""
    hits = 0
    for cands, truth in zip(matcher.top_k(predictions, k), truths):
        if truth.upper() in {c.upper() for c, _ in cands}:
            hits += 1
    return hits / len(truths) if truths else 0.0


def car(prediction: str, truth: str) -> float:
    """Character Accuracy Rate = 1 - 정규화 편집거리 (기획서 3.1 주 모니터링 지표)."""
    return 1.0 - Levenshtein.normalized_distance(prediction.upper(), truth.upper())
