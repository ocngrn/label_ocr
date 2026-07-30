"""Phase 0에서 확정된 고정 규격 (근거: docs/framework_decision.md).

크롭 생성(Phase 2), 합성 데이터(Phase 5), 학습 config(TASK 6)가 모두 이 값에
합의해야 한다. 여기를 바꾸면 크롭을 전부 재생성해야 하므로 단일 진실 공급원으로 둔다.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "image_set"
LABEL_FILE = IMAGE_DIR / "Label.txt"
DICT_FILE = PROJECT_ROOT / "configs" / "dict.txt"

# 인식기 입력 규격. Phase 0 실측: CTC 타임스텝 = 입력 폭 // 4.
# 폭 256 -> 64 타임스텝 = 최대 라벨 길이 32자의 2배(반복 문자용 blank 여유).
REC_IMAGE_SHAPE = (3, 64, 256)
REC_SEQ_LEN = 64
MAX_TEXT_LENGTH = 40

# True 이면 PaddleOCR 이 공백을 문자 목록에 자동 추가한다
# (ppocr/data/imaug/label_ops.py: BaseRecLabelEncode).
# 따라서 dict.txt 에는 공백 줄을 넣지 않는다 — 넣으면 인덱스가 중복된다.
USE_SPACE_CHAR = True

# 규칙 1: 절대 병합 금지. 각 쌍의 양쪽이 모두 실존하는 별개 부품 코드다
# (8135/B13S, 83/B3 등 — plan.md Phase 1-1 에서 원본 이미지 대조로 확정).
CONFUSABLE_PAIRS = (("8", "B"), ("5", "S"), ("0", "O"), ("1", "I"), ("2", "Z"))

# 원본 대조로 실제 소문자임이 확정된 라벨 (plan.md Phase 1-2). 이 3건만 대문자 정규화 대상.
KNOWN_LOWERCASE_LABELS = ("166m84", "Nav-D", "E27C. 50X50X6t")
