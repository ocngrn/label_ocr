"""텍스트 정규화 및 문자 사전 생성.

**접는 축은 대소문자 하나뿐이다.** 골격 정규화(8<->B, 5<->S, 0<->O, 1<->I, 2<->Z)는
금지된다 — `8135` 와 `B13S`, `83` 과 `B3` 가 모두 실존하는 별개 부품 코드이기 때문이다
(plan.md Phase 1-1, 원본 이미지 전수 대조로 확정). 이를 접으면 실존 코드 2개가 붕괴한다.
"""

from src import spec


def normalize_text(text: str) -> str:
    """학습·평가용 정답 표기로 정규화. 대문자화만 수행한다.

    원값은 호출측에서 `label_original` 로 별도 보존할 것.
    """
    return text.upper()


def build_charset(texts) -> list:
    """정규화된 라벨에서 실제 등장한 문자만 수집해 정렬 반환.

    공백은 제외한다 — PaddleOCR 이 `use_space_char=True` 일 때 문자 목록에 자동으로
    덧붙이므로, dict.txt 에도 넣으면 인덱스가 중복된다.
    """
    chars = {c for t in texts for c in normalize_text(t)}
    chars.discard(" ")
    return sorted(chars)


def write_dict(charset, path=None) -> int:
    """dict.txt 를 한 줄에 한 문자로 기록하고 문자 수를 반환."""
    path = path or spec.DICT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    # PaddleOCR 은 개행만 strip 하므로 개행 고정(\n)으로 쓴다.
    with open(path, "w", encoding="utf-8", newline="\n") as fout:
        fout.write("\n".join(charset) + "\n")
    return len(charset)


def read_dict(path=None) -> list:
    path = path or spec.DICT_FILE
    with open(path, encoding="utf-8") as fin:
        return [line.rstrip("\n").rstrip("\r") for line in fin if line.rstrip("\n").rstrip("\r")]


def out_channels(charset) -> int:
    """인식 헤드의 클래스 수 = 사전 + 공백(use_space_char) + blank(CTC)."""
    return len(charset) + (1 if spec.USE_SPACE_CHAR else 0) + 1
