"""Phase 1 산출물 생성기 — dict.txt, 프로파일/자산 리포트, 데이터 스냅샷.

실행: python -m src.preprocess.build_snapshot
"""

import collections
import hashlib
import json
from datetime import datetime, timezone

from PIL import Image

from src import spec
from src.preprocess import normalize, parse_label

SNAPSHOT_TAG = "label_snapshot_v1"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fin:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_image_assets(samples) -> dict:
    """지시서 TASK 0-2 정합성 게이트: 누락·미라벨·손상·해상도."""
    labeled = {s.image for s in samples}
    on_disk = {p.name for p in spec.IMAGE_DIR.glob("*.jpg")}

    resolutions, corrupt, hashes = collections.Counter(), [], {}
    for name in sorted(on_disk):
        path = spec.IMAGE_DIR / name
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                resolutions[im.size] += 1
        except Exception as exc:  # 손상 파일은 학습을 조용히 깨뜨리므로 전수 검사
            corrupt.append((name, str(exc)))
        hashes[name] = sha256(path)

    return {
        "labeled": len(labeled),
        "on_disk": len(on_disk),
        "missing": sorted(labeled - on_disk),
        "unlabeled": sorted(on_disk - labeled),
        "corrupt": corrupt,
        "resolutions": {f"{w}x{h}": n for (w, h), n in resolutions.most_common()},
        "hashes": hashes,
    }


def _md_table(rows) -> str:
    return "\n".join(f"| {k} | {v} |" for k, v in rows)


def write_reports(prof, assets, charset):
    spec.PROJECT_ROOT.joinpath("reports").mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    gate_ok = not (assets["missing"] or assets["unlabeled"] or assets["corrupt"])
    asset_md = f"""# 이미지 자산 정합성 검증 (image_asset_check_v1)

생성: {stamp} / 스냅샷: `{SNAPSHOT_TAG}`
지시서 `claude_code_prompt_paddleocr_colab.md` TASK 0-2 게이트.

| 항목 | 값 |
|---|---|
{_md_table([
    ("라벨에 등장한 이미지", assets["labeled"]),
    ("디스크의 jpg 파일", assets["on_disk"]),
    ("① 누락 (라벨에 있으나 파일 없음)", len(assets["missing"])),
    ("② 미라벨 (파일은 있으나 라벨 없음)", len(assets["unlabeled"])),
    ("③ 손상 (열리지 않음)", len(assets["corrupt"])),
])}

## ④ 해상도 분포
| 해상도 | 장수 |
|---|---|
{_md_table(assets["resolutions"].items())}

## 판정
**{"통과 — 다음 TASK 진행 가능" if gate_ok else "실패 — 원인 해결 전까지 진행 금지"}**
{"" if gate_ok else chr(10).join(f"- 누락: {p}" for p in assets["missing"][:20])}
"""

    prof_md = f"""# 데이터 프로파일 (data_profile_v1)

생성: {stamp} / 스냅샷: `{SNAPSHOT_TAG}` / 원천: `image_set/Label.txt`

| 항목 | 값 |
|---|---|
{_md_table([
    ("이미지 수", prof["images"]),
    ("텍스트 박스 수", prof["boxes"]),
    ("이미지당 평균 박스", f"{prof['boxes_per_image']:.2f}"),
    ("고유 텍스트 값", prof["unique_texts"]),
    ("곡면 박스 (5점 이상)", prof["curved_boxes"]),
    ("곡면 이미지", prof["curved_images"]),
    ("다중 토큰 박스 (공백 포함)", f"{prof['multi_token_boxes']} ({prof['multi_token_boxes']/prof['boxes']*100:.1f}%)"),
    ("빈 transcription 박스", prof["empty_boxes"]),
    ("숫자 비중", f"{prof['digit_ratio']*100:.1f}%"),
    ("최대 라벨 길이", prof["max_text_length"]),
])}

## 포인트 개수 분포 (4점 vs 5점 이상)
```
{prof["point_distribution"]}
```

## 문자셋 (정규화 전, 등장 빈도순 아님)
```
{prof["charset"]!r}
```
소문자 라벨 {len(prof["lowercase_texts"])}건: {prof["lowercase_texts"]}

## 문자 사전 (`configs/dict.txt`)
- 사전 문자 수: **{len(charset)}** (공백 제외 — `use_space_char=True` 가 자동 추가)
- 인식 헤드 클래스 수 `out_channels`: **{normalize.out_channels(charset)}** (= 사전 + 공백 + CTC blank)
- 내용: `{"".join(charset)!r}`

## 길이 분포
```
{prof["length_distribution"]}
```

> 최대 길이 **{prof["max_text_length"]}**자 대비 인식기 시퀀스 `REC_SEQ_LEN={spec.REC_SEQ_LEN}`
> (입력 `{spec.REC_IMAGE_SHAPE}`), `MAX_TEXT_LENGTH={spec.MAX_TEXT_LENGTH}`. 근거: docs/framework_decision.md §5
"""

    (spec.PROJECT_ROOT / "reports" / "image_asset_check_v1.md").write_text(asset_md, encoding="utf-8")
    (spec.PROJECT_ROOT / "reports" / "data_profile_v1.md").write_text(prof_md, encoding="utf-8")
    return gate_ok


def main():
    samples = parse_label.parse()
    prof = parse_label.profile(samples)
    charset = normalize.build_charset(b.transcription for _, b in parse_label.iter_boxes(samples))
    normalize.write_dict(charset)

    assets = check_image_assets(samples)
    gate_ok = write_reports(prof, assets, charset)

    snap_dir = spec.PROJECT_ROOT / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    (snap_dir / f"{SNAPSHOT_TAG}.json").write_text(
        json.dumps(
            {
                "tag": SNAPSHOT_TAG,
                "created": datetime.now(timezone.utc).isoformat(),
                "label_file_sha256": sha256(spec.LABEL_FILE),
                "images": len(assets["hashes"]),
                "image_sha256": assets["hashes"],
                "profile": {k: v for k, v in prof.items() if k != "char_counts"},
                "dict_size": len(charset),
                "out_channels": normalize.out_channels(charset),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"images={prof['images']} boxes={prof['boxes']} dict={len(charset)} "
          f"out_channels={normalize.out_channels(charset)} max_len={prof['max_text_length']}")
    print(f"asset gate: {'PASS' if gate_ok else 'FAIL'}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
