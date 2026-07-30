"""Label.txt (PPOCR 검출 포맷) 파서 및 데이터 프로파일러.

포맷: `이미지경로\\t[{"transcription": ..., "points": [[x,y],...], "difficult": ...}, ...]`
경로는 디렉터리 접두사 없는 파일명이며 IMAGE_DIR 기준으로 해석된다.
"""

import collections
import json
from dataclasses import dataclass

from src import spec


@dataclass(frozen=True)
class Box:
    transcription: str
    points: list
    difficult: bool

    @property
    def is_curved(self) -> bool:
        """5점 이상 다각형 = 곡면 인쇄 텍스트."""
        return len(self.points) > 4


@dataclass(frozen=True)
class Sample:
    image: str
    boxes: list


def parse(label_file=None) -> list:
    """Label.txt 를 Sample 목록으로 읽는다."""
    label_file = label_file or spec.LABEL_FILE
    samples = []
    with open(label_file, encoding="utf-8") as fin:
        for line in fin:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            image, payload = line.split("\t", 1)
            boxes = [
                Box(a["transcription"], a["points"], a.get("difficult", False))
                for a in json.loads(payload)
            ]
            samples.append(Sample(image, boxes))
    return samples


def iter_boxes(samples):
    for s in samples:
        for b in s.boxes:
            yield s, b


def profile(samples) -> dict:
    """데이터 프로파일 집계 — reports/data_profile_v1.md 의 원천."""
    boxes = [b for _, b in iter_boxes(samples)]
    texts = [b.transcription for b in boxes]
    chars = collections.Counter(c for t in texts for c in t)
    n_chars = sum(chars.values())
    curved = [b for b in boxes if b.is_curved]

    return {
        "images": len(samples),
        "boxes": len(boxes),
        "boxes_per_image": len(boxes) / len(samples) if samples else 0,
        "unique_texts": len(set(texts)),
        "point_distribution": dict(sorted(collections.Counter(len(b.points) for b in boxes).items())),
        "curved_boxes": len(curved),
        "curved_images": sum(1 for s in samples if any(b.is_curved for b in s.boxes)),
        "multi_token_boxes": sum(1 for t in texts if " " in t.strip()),
        "empty_boxes": sum(1 for t in texts if not t),
        "charset": "".join(sorted(chars)),
        "char_counts": dict(sorted(chars.items())),
        "digit_ratio": sum(v for k, v in chars.items() if k.isdigit()) / n_chars if n_chars else 0,
        "lowercase_texts": sorted({t for t in texts if any(c.islower() for c in t)}),
        "max_text_length": max((len(t) for t in texts), default=0),
        "length_distribution": dict(sorted(collections.Counter(len(t) for t in texts).items())),
    }
