"""추론 실행 — 문자별 확신도 포함 (지시서 TASK 3-4 아카이빙 규약 충족).

PaddleX 의 `TextRecognition.predict` 는 `rec_score`(문자별 확신도의 평균) 하나만
돌려주고 문자별 원본은 `CTCLabelDecode.decode` 안에서 계산된 뒤 버려진다
(`np.mean(conf_list)` 만 반환). 미등록 판정 임계값 튜닝(TASK 6-E)에는 "한 글자만
확신이 낮은" 경우를 구별할 수 있어야 하므로, 디코더를 가로채 원본을 확보한다.

배치 정렬이 어긋나면 확신도가 엉뚱한 크롭에 붙으므로, 가로챈 값의 평균이 API 가
돌려준 `rec_score` 와 일치하는지 **전 건 검증**한다.
"""

import contextlib

import numpy as np
from paddlex.inference.models.text_recognition.processors import CTCLabelDecode


@contextlib.contextmanager
def _capture_char_confidences():
    """CTC 디코더가 버리는 문자별 확신도를 수집한다."""
    captured = []
    original = CTCLabelDecode.__call__

    def patched(self, pred, return_word_box=False, **kwargs):
        preds = np.array(pred[0])
        idx, prob = preds.argmax(axis=-1), preds.max(axis=-1)
        ignored = self.get_ignored_tokens()
        for b in range(len(idx)):
            # CTCLabelDecode.__call__ 이 쓰는 선택 규칙과 동일 (중복 제거 + blank 제외)
            selection = np.ones(len(idx[b]), dtype=bool)
            selection[1:] = idx[b][1:] != idx[b][:-1]
            for token in ignored:
                selection &= idx[b] != token
            captured.append(prob[b][selection].tolist())
        return original(self, pred, return_word_box, **kwargs)

    CTCLabelDecode.__call__ = patched
    try:
        yield captured
    finally:
        CTCLabelDecode.__call__ = original


def recognize(model, images, batch_size=8):
    """[(text, char_confidences, mean_confidence)] 를 입력 순서대로 반환."""
    with _capture_char_confidences() as captured:
        results = list(model.predict(images, batch_size=batch_size))

    if len(captured) != len(results):
        raise RuntimeError(f"확신도 {len(captured)}건 vs 예측 {len(results)}건 — 개수 불일치")

    out = []
    for i, (res, confs) in enumerate(zip(results, captured)):
        score = float(res["rec_score"])
        mean = float(np.mean(confs)) if confs else 0.0
        if abs(mean - score) > 1e-3:
            raise RuntimeError(
                f"[{i}] 확신도 정렬 불일치: 가로챈 평균 {mean:.6f} != rec_score {score:.6f}")
        out.append((res["rec_text"], [float(c) for c in confs], score))
    return out
