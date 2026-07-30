"""Phase 3 — 누수 방지 분할 (TASK 2).

실행: python -m src.preprocess.split

## 분할 단위: 이미지 (검출·end-to-end 평가가 이미지 단위이므로)

## 누수 방지 규칙과 그 완화 근거

지시서는 "시리얼 값 단위 그룹 분할 + 이미지 단위 분할 병행"을 요구한다. 그러나
**모든 반복 시리얼로 그룹화하면 이미지의 74.4%(1,740장)가 하나의 연결요소로 묶여**
80/10/10 분할 자체가 불가능하다 (한 이미지에 여러 코드가 있고, 계열 코드가
수십 장에 걸쳐 반복되므로 이미지들이 전부 연결된다).

원인은 `8166`(75장), `8100`(59), `(1)`(53), `P29`(33) 같은 **짧은 계열·수량 코드**다.
이들은 식별 시리얼이 아니라 도메인 어휘이며, 기획서도 모델이 코드 사전확률(prior)을
학습하기를 기대한다. 따라서 **길이 {IDENTIFYING_MIN_LEN}자 이상만 식별 시리얼로 보고 그룹화**한다.

    그룹화 기준            최대 연결요소
    전체 반복 시리얼       1,740 (74.4%)  -> 분할 불가
    길이 >= 5              78 (3.3%)      -> 채택
    길이 >= 8              26 (1.1%)

길이 5 임계에서 제외되는 것은 `8166`, `(1)`, `902`, `E51S` 등 계열/수량/순번 코드이고,
포착되는 것은 `H32C-52`, `EL 8166 P77-04`, `8100LUA.28` 등 실제 부품 식별 코드다.
잔여 누수(계열 코드가 train/test 양쪽 등장)는 리포트에 정량 기록한다.
"""

import collections
import json
import random

from src import spec

SPLITS = ("train", "val", "test")
SNAPSHOT = spec.PROJECT_ROOT / "snapshots" / "label_snapshot_v2.json"
LABEL_DIR = spec.PROJECT_ROOT / "labels"
SPLIT_DIR = spec.PROJECT_ROOT / "splits"


def load_records():
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))["records"]


def build_components(records):
    """식별 시리얼을 공유하는 이미지끼리 묶어 연결요소를 만든다."""
    ser2img = collections.defaultdict(set)
    for r in records:
        ser2img[r["label"]].add(r["image"])

    images = sorted({r["image"] for r in records})
    parent = {i: i for i in images}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for serial, imgs in ser2img.items():
        if len(imgs) < 2 or len(serial) < spec.IDENTIFYING_MIN_LEN:
            continue
        imgs = sorted(imgs)
        for other in imgs[1:]:
            a, b = find(imgs[0]), find(other)
            if a != b:
                parent[a] = b

    groups = collections.defaultdict(list)
    for img in images:
        groups[find(img)].append(img)
    return list(groups.values())


def assign(components, curved_images, seed=None):
    """연결요소를 train/val/test에 배정한다.

    곡면 이미지를 포함한 요소를 먼저 배정해 층화를 보장하고(train 곡면 0 방지),
    나머지를 이미지 수 목표에 맞춰 채운다. 매 단계 목표 대비 가장 모자란 쪽에 넣는
    그리디이며, 시드는 동률 처리에만 쓰인다.
    """
    rng = random.Random(seed if seed is not None else spec.SPLIT_SEED)
    total = sum(len(c) for c in components)
    total_curved = len(curved_images)

    def n_curved(comp):
        return sum(1 for i in comp if i in curved_images)

    assigned = {s: [] for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    curved_counts = {s: 0 for s in SPLITS}

    def place(comps, key, counter, denom):
        """목표 대비 부족분이 가장 큰 split 에 큰 요소부터 배정."""
        for comp in sorted(comps, key=lambda c: (-key(c), rng.random())):
            target = min(SPLITS, key=lambda s: counter[s] - spec.SPLIT_RATIOS[s] * denom)
            assigned[target].append(comp)
            counts[target] += len(comp)
            curved_counts[target] += n_curved(comp)

    curved_comps = [c for c in components if n_curved(c)]
    plain_comps = [c for c in components if not n_curved(c)]
    place(curved_comps, n_curved, curved_counts, total_curved)
    place(plain_comps, len, counts, total)

    return {s: sorted(i for c in assigned[s] for i in c) for s in SPLITS}


def curved_folds(split_images, curved_images, components, n_folds=None):
    """곡면 서브셋 K-Fold 폴드 목록 (지시서 TASK 2-3 게이트).

    곡면 99박스를 8:1:1로 나누면 test 에 10개 내외만 남아 예측 1건이 지표를
    10%p 흔든다. 곡면 평가는 단일 분할 대신 K-Fold 평균±표준편차로 보고한다.
    연결요소는 쪼개지 않는다(폴드 간 누수 방지).
    """
    n_folds = n_folds or spec.CURVED_N_FOLDS
    rng = random.Random(spec.SPLIT_SEED)
    comps = [sorted(set(c) & curved_images) for c in components]
    comps = sorted([c for c in comps if c], key=lambda c: (-len(c), c[0]))

    folds = [[] for _ in range(n_folds)]
    for comp in comps:
        folds.sort(key=lambda f: (len(f), rng.random()))
        folds[0].extend(comp)
    return [sorted(f) for f in folds]


def seen_in_train(records, split):
    """평가용 크롭별 플래그 — 이 크롭의 라벨을 train 에서 본 적 있는가.

    길이 4자 이하 계열 코드(`8166`, `(1)`, `902`)는 그룹화하지 않으므로 train/test
    양쪽에 등장한다(이유는 모듈 docstring 참조). 그만큼 전체 평균 지표는 낙관적이다.
    이 플래그로 지표를 seen/unseen 으로 분해해 **일반화 성능을 따로 보고**한다.
    """
    train_labels = {r["label"] for r in records if r["image"] in set(split["train"])}
    return {
        s: {r["crop"]: (r["label"] in train_labels)
            for r in records if r["image"] in set(split[s])}
        for s in ("val", "test")
    }


def leakage(records, split):
    """분할 후 실제 남은 라벨 중복을 정량화한다."""
    by_split = {s: {r["label"] for r in records if r["image"] in set(split[s])} for s in SPLITS}
    train, test = by_split["train"], by_split["test"]
    shared = train & test
    identifying = {l for l in shared if len(l) >= spec.IDENTIFYING_MIN_LEN}
    test_crops = [r for r in records if r["image"] in set(split["test"])]
    seen = sum(1 for r in test_crops if r["label"] in train)
    return {
        "test_unique_labels": len(test),
        "shared_with_train": len(shared),
        "shared_identifying": sorted(identifying),
        "test_labels_unseen_in_train": len(test - train),
        "test_crops": len(test_crops),
        "test_crops_seen_label": seen,
        "test_crops_unseen_label": len(test_crops) - seen,
    }


def write_label_files(records, split):
    det_src = {}
    for line in (LABEL_DIR / "det_all.txt").read_text(encoding="utf-8").splitlines():
        if line:
            name, payload = line.split("\t", 1)
            det_src[name] = payload

    LABEL_DIR.mkdir(exist_ok=True)
    for s in SPLITS:
        members = set(split[s])
        det = [f"{n}\t{det_src[n]}" for n in split[s] if n in det_src]
        rec = [f"{r['crop']}\t{r['label']}" for r in records if r["image"] in members]
        (LABEL_DIR / f"det_{s}.txt").write_text("\n".join(det) + "\n", encoding="utf-8", newline="\n")
        (LABEL_DIR / f"rec_{s}.txt").write_text("\n".join(rec) + "\n", encoding="utf-8", newline="\n")


def build():
    records = load_records()
    curved_images = {r["image"] for r in records if r["is_curved"]}
    components = build_components(records)
    split = assign(components, curved_images)
    folds = curved_folds(split, curved_images, components)

    write_label_files(records, split)
    SPLIT_DIR.mkdir(exist_ok=True)

    stats = {}
    for s in SPLITS:
        members = set(split[s])
        rs = [r for r in records if r["image"] in members]
        stats[s] = {
            "images": len(members),
            "boxes": len(rs),
            "curved_images": len(members & curved_images),
            "curved_boxes": sum(1 for r in rs if r["is_curved"]),
            "unique_labels": len({r["label"] for r in rs}),
            "multi_token_boxes": sum(1 for r in rs if len(r["tokens"]) > 1),
        }

    payload = {
        "seed": spec.SPLIT_SEED,
        "ratios": spec.SPLIT_RATIOS,
        "identifying_min_len": spec.IDENTIFYING_MIN_LEN,
        "n_components": len(components),
        "max_component": max(len(c) for c in components),
        "split": split,
        "stats": stats,
        "leakage": leakage(records, split),
        "seen_in_train": seen_in_train(records, split),
        "curved_folds": folds,
    }
    (SPLIT_DIR / f"split_seed{spec.SPLIT_SEED}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for s in SPLITS:
        st = stats[s]
        print(f"  {s:5s} images={st['images']:4d} ({st['images']/2340*100:4.1f}%) "
              f"boxes={st['boxes']:4d} curved_img={st['curved_images']:3d} "
              f"curved_box={st['curved_boxes']:3d} uniq={st['unique_labels']:4d}")
    lk = payload["leakage"]
    print(f"  누수: test 고유 라벨 {lk['test_unique_labels']} 중 train 과 공유 {lk['shared_with_train']} "
          f"(식별 시리얼 {len(lk['shared_identifying'])})")
    print(f"  test 크롭 {lk['test_crops']} = unseen {lk['test_crops_unseen_label']} "
          f"+ seen {lk['test_crops_seen_label']} (지표를 이 축으로 분해 보고)")
    print(f"  곡면 폴드 크기: {[len(f) for f in folds]}")
    return payload


if __name__ == "__main__":
    build()
