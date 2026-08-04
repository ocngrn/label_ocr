# 작업 계획 (work_plan.md) — 실측 기반

> **문서 역할**: `claude_code_prompt_paddleocr_colab.md`(실행 지시서)를 **현재 폴더·환경 실측 결과에 맞춰 조정한 착수 계획**이다.
> 지시서와 충돌하는 항목은 이 문서에 근거(실측 명령·결과)를 남겼다. 지시서의 도메인 규칙(문자 골격 정규화 금지 등)은 그대로 유효하다.
>
> 실측 일자: 2026-07-31 / 실측 환경: `o:\Project\label_ocr`, Windows 11, Python 3.13.9

---

## 1. 현재 폴더 구조 (실측)

```
o:\Project\label_ocr\
  .bkit\                    # bkit 플러그인 상태 (작업 산출물 아님)
  docs\
    CLAUDE.md               # 행동 지침
    plan.md                 # 기획서 (의사결정 근거 저장소)
    claude_code_prompt_paddleocr_colab.md   # 실행 지시서 (단일 진실 공급원)
    work_plan.md            # ← 이 문서
  image_set\
    *.jpg          x2,340   # 원본 이미지
    Label.txt               # PPOCR det 포맷 라벨 (2,340행)
    fileState.txt           # PPOCRLabel 산출물 (작업 대상 아님)
    Cache.cach              # PPOCRLabel 산출물 (작업 대상 아님)
```

**없는 것**: `src/`, `configs/`, `experiments/`, `reports/`, `notebooks/`, `splits/`, `tests/` — 코드는 **완전 그린필드**.

**git 저장소 아님** → 기획서 7장 재현성 규칙("코드 Git 태그/커밋 기록")이 현재 강제 불가. Phase 0에서 `git init` 필요.

**`docs/System_Architecture.md` 부재** — `plan.md`가 "신설했다"고 참조하지만 실제로 존재하지 않음. 앱 개발 단계 전까지는 차단 요인 아니므로 이번 범위에서 제외.

---

## 2. 데이터 무결성 검증 결과 — **TASK 0 게이트 통과**

지시서 TASK 0-2의 이미지 자산 정합성 게이트를 실행한 결과:

| 검증 항목 | 결과 | 판정 |
|---|---|---|
| 라벨 행 수 / 고유 경로 | 2,340 / 2,340 | ✅ |
| 디스크 jpg 파일 수 | 2,340 | ✅ |
| 라벨에 있으나 파일 없음(누락) | **0건** | ✅ |
| 파일은 있으나 라벨 없음(미라벨) | **0건** | ✅ |
| 손상 파일 (PIL verify 전수) | **0건** | ✅ |
| 해상도 분포 | **2560×1920 단일 해상도 (2,340장 전부)** | ✅ |
| 라벨 파싱 오류 | 0건 | ✅ |

→ **TASK 0의 차단 게이트는 이미 해소 상태.** 스캐폴딩만 만들면 곧바로 TASK 1로 진행 가능.

### 라벨 경로 형식 주의
`Label.txt`의 경로는 **디렉터리 접두사 없는 순수 파일명**(`20230909_093111.jpg`)이다.
→ PaddleOCR config의 `data_dir`를 `image_set/`로 잡아야 해석된다. 지시서 TASK 0의 트리(`data/`)로 옮길 경우 경로 재작성이 필요하다.

---

## 3. 데이터 프로파일 실측 (지시서 수치 검증)

| 항목 | 지시서 기재 | 실측 | 일치 |
|---|---|---|---|
| 이미지 수 | 2,340 | 2,340 | ✅ |
| 박스 수 | 8,103 | 8,103 | ✅ |
| 4점 박스 | 8,004 (98.8%) | 8,004 | ✅ |
| 5점 이상 박스 | 99 (1.2%) | 99 | ✅ |
| 곡면 이미지 | 93 | 93 | ✅ |
| 다중 토큰 비율 | 21.6% | 1,751 / 8,103 = 21.6% | ✅ |
| 숫자 비중 | 60.6% | 60.6% | ✅ |
| 소문자 라벨 | 3건 | 3건 (`166m84`, `Nav-D`, `E27C. 50X50X6t`) | ✅ |
| 고유 텍스트 | 약 4,281 | 4,280 | ~ |

폴리곤 점 수 분포: `4:8004, 5:28, 6:22, 7:1, 8:11, 9:3, 10:10, 11:5, 12:13, 13:2, 14:2, 15:1, 16:1`

### ⚠️ 발견된 불일치 4건 (지시서 수정 필요)

**① 문자 사전에 아포스트로피 `'` 누락 — 36회 등장**

지시서 확정 문자셋: `0-9 + A-Z + 기호(- / . · ( ) : =` 공백)`
실측 전체 문자셋 (코드포인트 확인):

```
U+0020 ' '  2683    U+002E '.'   380    U+003D '='     2
U+0027 "'"    36 ←  U+002F '/'    59    U+00B7 '·'    35
U+0028 '('   943    U+003A ':'     2    0-9, A-Z
U+0029 ')'   943    U+002D '-'  2777    a,m,t,v (소문자 3라벨)
```

`'`(U+0027)가 **36회** 등장한다(`'10EA'`, `'2EA'`, `8173LCB - 8 - G23S CT410 -'10EA'` 등 수량 표기 관례).
지시서대로 `dict.txt`를 만들면 이 문자가 클래스에서 빠져 **해당 박스들은 구조적으로 절대 정답을 맞힐 수 없다.**

> **채택 결정**: `'`를 `dict.txt`에 **포함**한다. 근거 — 실사 라벨에 36회 실존하고, 제외 시 손실이 확정적인 반면 포함 비용은 클래스 1개다. (`8/B` 구별축과 무관하므로 규칙 1과 충돌 없음.)

**② `max_text_length` 실측 최댓값이 32 — 지시서의 "24자+ 여유"로는 부족**

지시서 TASK 6-B: `Global.max_text_length = 실측 최댓값(24자+ 여유) 이상`
실측 최댓값은 **32자**: `8173LCB - 8 - G23S CT410 -'10EA'`
길이 24자 이상 박스가 29개 존재(24:12, 25:4, 26:3, 27:5, 28:2, 29:2, 32:1).

> **채택 결정**: `max_text_length = 40`. 24로 설정하면 29개 박스가 **조용히 잘려** 학습 타깃이 오염된다.

**③ 빈 transcription 박스 1건** — 길이 0인 박스가 1개 존재. TASK 1에서 제외 또는 재라벨 판정 필요.

**④ 고유 텍스트 4,280** (지시서 4,281). `serial_db_proxy_v1`의 `|DB|` 기록 시 실측값 사용.

---

## 4. 환경 실측 — **지시서의 가장 큰 전제가 어긋남**

### 4.1 로컬에 이미 전체 스택이 설치돼 있음

```
paddlepaddle    3.3.1   (compiled_with_cuda: False → CPU 전용, GPU 0개)
paddleocr       3.7.0
paddlex         3.7.2
opencv-contrib  4.10.0 / opencv-python-headless 4.13.0
numpy 2.3.5 / pillow 12.0 / shapely 2.1.2 / lmdb 1.7.5
albumentations 2.0.8 / RapidFuzz 3.14.5
```

전처리·크롭·분할·매칭·LMDB 패킹에 필요한 라이브러리가 **전부 갖춰져 있다**. GPU만 없다.

### 4.2 🔴 최대 리스크: 지시서가 PaddleOCR **2.x 세대**를 전제로 작성됨

지시서 TASK 6은 다음을 전제한다 — 전부 **2.x dygraph 세대의 인터페이스**다:

- `python tools/train.py -c configs/rec/<our_rec>.yml`
- `tools/infer/utility.py`의 `get_rotate_crop_image`
- `Global.pretrained_model`, `Global.character_dict_path`, `batch_size_per_card`
- RARE/StarNet **v2.0 dygraph** 가중치 (`rec_r34_vd_tps_bilstm_att_v2.0_train.tar`)

그러나 설치된 `paddleocr` 3.7.0은 **추론 전용 래퍼**다. 패키지 export 확인 결과:

```
PaddleOCR, PPStructureV3, DocVLM, PaddleOCRVL, FormulaRecognition, ...
→ 학습 진입점(tools/train.py, configs/) 없음. 학습은 PaddleX로 이관됨.
```

**이것이 TASK 6 착수 시점이 아니라 지금 결정돼야 하는 이유**: 2.x/3.x 선택은 `dict.txt` 포맷, config 스키마, baseline 모델, **그리고 4단계 RARE/TPS 실험의 가능 여부 자체**를 좌우한다. TASK 6에서 발견하면 TASK 1~5 산출물을 재작업해야 한다.

### 4.3 🟢 반대로, TASK 1의 핵심 게이트는 **이미 해소됐을 가능성이 높음**

지시서 TASK 1-4는 이렇게 단언한다:

> **[게이트] 4점 초과 폴리곤 크롭 — PaddleOCR 내장 함수로 불가, 직접 구현 필수**
> `get_rotate_crop_image`는 4점 입력만 받고 `assert len(points) == 4`로 예외 발생 (GitHub Issue #5300)

이 근거는 2022년 PaddleOCR 2.x 기준이다. 설치된 **PaddleX 3.7.2에는 이미 구현이 있다**:

`paddlex/inference/pipelines/components/common/crop_image_regions.py`
```
class CropByPolys(det_box_type="poly")
  ├─ get_rotate_crop_image()      # line 176: assert len(points)==4  ← 4점 전용 (지시서가 지목한 함수)
  ├─ reorder_poly_edge()          # line 230: assert shape[0] >= 4   ← 5점 이상 허용, 상/하단선 분리
  ├─ find_head_tail()             # line 283: if len(points) > 4:    ← 다각형 분기 처리
  └─ sample_points_on_bbox_bp(line, n=50)   # 각 변을 등간격 리샘플링
```

이는 지시서가 "직접 구현하라"고 한 **segment-based unrolling(폴리곤→상/하단선 분리→리샘플링→워핑)과 정확히 동일한 절차**다.
추가로 `paddlex/.../seal_det_warp.py`에 `CurveTextRectifier` / `AutoRectifier`(곡면 텍스트 정류) 완성 구현이 존재한다.

> **판단**: TASK 1의 직접 구현 게이트는 **재검증 대상**. 다만 지시서 자체가 요구한 "육안 확인 포함 구현 검증"은 그대로 수행해야 한다 — 우리 99개 곡면 박스에서 실제로 잘 펴지는지는 실측 전까지 알 수 없다.

---

## 5. 작업 계획

### 실행 환경 분리 (지시서 "전부 Colab" 전제를 조정)

| 구분 | 환경 | 근거 |
|---|---|---|
| TASK 0~3, 5 (전처리·크롭·분할·매칭·합성) | **로컬** `o:\Project\label_ocr` | GPU 불필요. 스택 설치 완료. Drive I/O 병목·세션 휘발 회피 |
| TASK 4 (baseline 추론) | 로컬 CPU 우선 시도 → 느리면 Colab | test셋 234장 규모. CPU로 측정 가능성 있음 |
| TASK 6 (학습) | **Colab GPU** | 로컬 `compiled_with_cuda: False` |
| TASK 7 (평가) | 로컬 | 추론 결과 파일 기반 |

코드는 로컬 git 저장소가 단일 진실 공급원, Colab은 **연산 자원**으로만 사용(Drive에 코드 복제 대신 git clone 또는 동기화).

---

### Phase 0 — 프레임워크 세대 확정 [🔴 최우선 게이트] — ✅ **완료 (2026-07-31)**

**결과 → [`docs/framework_decision.md`](framework_decision.md)**

- **채택: PaddleOCR 2.10.0 (2.x)**. 보조로 `paddleocr` 3.7.0을 참조 baseline 측정 전용(추론만)으로 병행.
- 근거: 지시서는 PP-OCRv5(3.x 전용) baseline과 RARE/TPS 4단계(2.x 전용)를 **동시에** 요구하는데 한 세대로는 불가능. PaddleX 3.7.2에는 **RARE·TPS 구현이 전무**(검색 0건)하므로, 4단계를 유지하려면 2.x 외 선택지가 없음.
- **검증 통과**: RARE/StarNet/SVTR 3종 빌드+forward OK (paddle 3.3.1). RARE 사전학습 가중치 실로드 → 263개 중 **260개 전이, missing 0**, 불일치 3개는 전부 Head(지시서 6-B 예고대로 정상).
- **가중치 생존**: RARE/StarNet 3개 + PP-OCRv4 det·rec 3개 **전부 HTTP 200** → F1 폴백 사전 발동 불필요, 4단계 유지.
- **파생 결정 2건** (§7 참조): 인식 입력 규격 확정, SVTR STN_ON 전제 정정.

---

### Phase 1 — 스캐폴딩 + 재현성 기반 (TASK 0 잔여분) — ✅ **완료 (2026-07-31, `d186178`)**

- **테스트 29건 전부 통과** (`python -m pytest tests/ -q`)
- `configs/dict.txt` **45자** 확정 (`'()-./0123456789:=A-Z·`) → `out_channels=47` (= 45 + 공백 + CTC blank)
- 자산 게이트 재확인: 누락 0 / 미라벨 0 / 손상 0 / 2560×1920 단일
- `snapshots/label_snapshot_v1.json` — 라벨 SHA256 + 이미지 2,340장 개별 SHA256
- **파생 발견**: `use_space_char=True` 가 공백을 문자 목록에 **자동 추가**하므로 `dict.txt` 에 공백 줄을 넣으면 인덱스가 중복된다 → 사전에서 공백 제외 + 회귀 테스트 추가

<details><summary>원래 계획 (참고)</summary>

TASK 0의 데이터 게이트는 §2에서 통과 확인됨. 남은 것은 골격과 재현성 장치뿐이다.

1. `git init` + `.gitignore` (`image_set/*.jpg`, `experiments/`, `*.tar`, `.bkit/` 제외)
   → 기획서 7장 재현성 규칙이 이때부터 강제 가능해짐.
2. 프로젝트 트리 생성: `src/{preprocess,synth,eval,matching}`, `configs/`, `splits/`, `experiments/`, `reports/`, `notebooks/`, `tests/`
3. `src/preprocess/parse_label.py` — Label.txt 파서 (§2·§3 검증 로직을 모듈로 고정)
4. **불변 규칙 테스트** (pytest, 이후 상시 실행) — 지시서 TASK 0-5:
   - `dict.txt`에 소문자 없음
   - 정규화 함수가 `8→B`류 골격 접기를 하지 **않음** (`8135` ≠ `B13S`)
   - 대문자 정규화가 소문자 3건에만 적용 + 원값 이력 보존
   - 라벨의 모든 이미지 경로가 실제 파일로 해석됨
   - **(추가)** `dict.txt`가 실측 문자셋을 전부 포함 (`'` 포함 회귀 방지)
5. `label_snapshot_v1` 태깅 (이미지 SHA256 집계 포함)

- **검증**: `pytest tests/ -v` 전항 통과.
- **산출물**: `reports/data_profile_v1.md`, `reports/image_asset_check_v1.md`(§2 내용), git 초기 커밋

</details>

---

### Phase 2 — 라벨 전처리 + 곡면 크롭 (TASK 1) — ✅ **완료 (2026-07-31)**

- **[게이트] 곡면 크롭 → `minAreaRect` 채택.** 지시서 권장안(segment unrolling)을 **기각**했다.
  PP-OCRv5로 객관 판정: poly **CAR 0.4387**(빈 출력 27/99) vs quad **0.6172**(빈 출력 0/99).
  5점은 동일하나 **6점 이상에서 poly의 기하가 무너진다**(종횡비 3.4→0.97). 점 순서 정규화로도 미해소.
  근거 → [`reports/crop_method_decision.md`](../reports/crop_method_decision.md)
- 4점 박스: `get_rotate_crop_image` 유지 (두 방식 동등 — CAR 0.7922 vs 0.7902, 표본 300)
- 박스 분할 **(a)안** 확정 + 토큰 경계 메타데이터 보존
- 빈 transcription 1건 → **라벨링 실수로 확인, 검출·인식 양쪽에서 제외**
- 산출물: `labels/{det_all,rec_all}.txt`, `crops/` 8,102개, `snapshots/label_snapshot_v2.json`
- **테스트 39건 통과** (Phase 1의 29건 + Phase 2의 10건)
- **계획 정정**: 크롭을 `64×256`으로 미리 리사이즈하지 **않고 원본 해상도로 저장**한다.
  리사이즈는 학습 시 `rec_image_shape`가 담당하므로, 미리 굽면 규격 변경 시 8,102개를 재생성해야 한다.
  대신 Phase 5 합성이 실사 분포에 맞출 수 있도록 **종횡비 통계(p05/p50/p95 = 1.1/2.6/6.8)**를 스냅샷에 기록.

<details><summary>원래 계획 (참고)</summary>

1. 대문자 정규화 (소문자 3건 → 대문자, `label_original` 보존)
2. `configs/dict.txt` 생성 — **§3-① 결정 반영: `'` 포함**, 소문자 제외
3. **[게이트] 박스 분할 표준 확정** — 지시서 권장 (a)안(한 박스 유지 + 후처리 분리) 채택, 토큰 경계 메타데이터 보존
4. 빈 transcription 박스 1건 처리 판정 (§3-③)
5. **[게이트] 곡면 크롭 — §4.3 재검증**
   - 먼저 `PaddleX CropByPolys(det_box_type="poly")`를 99개 곡면 박스에 적용
   - **99건 전수 육안 확인** (지시서가 요구한 검증 절차)
   - 실패 시에만 지시서의 직접 구현(segment unrolling) 또는 `minAreaRect` 폴백으로 이행
6. 4점 박스 8,004개는 `get_rotate_crop_image` 그대로 사용
7. 인식용 크롭 + `경로\t정답` 매핑 파일 생성 — **규격 `64×256` 고정** (Phase 0 §5 결정. Phase 5 합성 크롭도 동일 규격이어야 하며, 뒤에서 바꾸면 양쪽 8,103+α개를 재생성해야 함)

- **검증**: 크롭 8,103개 전수 생성 + 곡면 99건 육안 통과 + Phase 1 pytest 여전히 통과
- **산출물**: `label_snapshot_v2`, `dict.txt`, 검출/인식 라벨 파일, 크롭 방식 결정 문서

</details>

---

### Phase 3 — 분할 (TASK 2) — ✅ **완료 (2026-07-31)**

- 이미지 **80.0/10.0/10.0%** 정확 일치, 곡면 이미지 **75/9/9**(train 곡면 0 방지 충족), 곡면 박스 99개 전량 배분
- **[게이트] 지시서 규칙 완화**: 모든 반복 시리얼로 그룹화하면 이미지의 **74.4%가 한 연결요소**가 되어 분할 불가.
  원인은 `8166`(75장) `(1)`(53) 같은 **계열·수량 코드**. → **길이 ≥ 5만 식별 시리얼로 그룹화**(최대 요소 3.3%).
  **제외되는 데이터는 없다** — 임계값은 "학습에서 뺄지"가 아니라 "같은 split에 묶을지"만 정한다.
- **식별 시리얼 누수 0건.** train/test 공유 라벨 193개는 전원 4자 이하(테스트로 강제)
- **잔여 낙관을 지표로 분해**: test 크롭 762 = unseen 424(55.6%, 주 지표) + seen 338(44.4%).
  크롭별 `seen_in_train` 플래그를 분할 파일에 저장 → TASK 4~7에서 CAR·Exact·Top-K를 이 축으로 분해 보고
- **[게이트] 곡면 표본 부족**: test 곡면 박스가 정확히 **9개**(예측 1건 = 11.1%p) → **5-Fold 채택** `[19,18,18,19,19]`.
  Anchor 방식은 train 곡면 0이 되어 미채택, 발동 조건만 기록
- **테스트 55건 통과**. 근거 → [`reports/split_stats.md`](../reports/split_stats.md)

<details><summary>원래 계획 (참고)</summary>

1. 시리얼 값 단위 그룹 분할 + 이미지 단위 분할 병행 → 80/10/10, 시드 고정
2. 곡면 93 이미지 층화 배분 (train 곡면 0 방지)
3. **[게이트] 곡면 표본 부족 대응** — 지시서 권장 K-Fold(기본) 채택, Anchor 방식은 안전판으로 병행 보고
- **검증**: 동일 시리얼이 train∩test에 0건임을 테스트로 강제 (누수 회귀 방지 테스트 추가)
- **산출물**: `splits/split_seedXX.json`, 분할 통계 리포트, K-Fold 폴드 목록

</details>

---

### Phase 4 — DB 프록시 + 기본 매칭 (TASK 3) — ✅ **완료 (2026-07-31)**

- **`serial_db_proxy_v1`, |DB| = 4,171** (학습에 없는 시리얼 **761개 포함** — 폐집합 성립 조건)
- **🔴 발견**: DB에 **공백만 다른 중복 99군집 207개**(`8196 LUM 02`/`8196LUM02` …).
  매칭이 공백을 무시하므로 구별 불가능한 중복이고, 방치하면 Top-1이 성능과 무관하게 최대 4.8% 깎인다
  (실제로 DB 자기조회 Top-1이 95%로 나와 발각). → 공백 제거형 키로 병합, |DB| 4,279 → **4,171**
- 규칙 1 검증: 실존 코드쌍 11개(DB에 양쪽 존재) 전부 서로 Top-1 가로채지 않음
- **🔴 목표선 보정 근거**: Top-K 응답 곡선 — 완벽 100%/100%, **공백 전부 탈락 100%/100%**(공백 옵션 정상),
  **1글자 오인식 63.4%/86.0%**, 2글자 48.0%/66.5%.
  → **Top-5 ≥95% 달성에는 Exact-match 최소 64%가 필요**. "후단 매칭이 흡수한다"는 1글자 오류에 한해 86%만 성립
- 예측 아카이빙 규약 확정(문자별 확신도 필수 — 누락 시 저장 실패)
- **테스트 95건 통과**. 근거 → [`reports/db_matching_v1.md`](../reports/db_matching_v1.md)

<details><summary>원래 계획 (참고)</summary>

1. `serial_db_proxy_v1` — 고유 시리얼 **4,280**(실측값) 스냅샷, 학습 제외분도 포함
2. `src/matching/match.py` — RapidFuzz 기반 Levenshtein Top-K (설치 완료)
3. 공백 처리 옵션(원문 비교 + 공백 제거 비교 중 짧은 거리 채택) 기본 탑재
4. **예측 원본 아카이빙 포맷 규약** 확정 (텍스트 + 문자별 확신도)
- **검증**: `8135`와 `B13S`가 서로 Top-1로 매칭되지 **않음**을 테스트로 확인
- **산출물**: `matching/` 모듈, `|DB|=4280` 기록, 매칭 테스트

</details>

---

### Phase 5 (= 지시서 TASK 4, Baseline 실측) — ✅ **완료 (2026-08-01)**

**인식 baseline** (test 762크롭, |DB|=4,171) → [`reports/baseline_v1.md`](../reports/baseline_v1.md)

| 모델 | CAR | Exact | Top-1 | Top-5 | 곡면 Exact |
|---|---|---|---|---|---|
| **PP-OCRv4** (공식 baseline, 2.x 계열) | 0.7565 | 35.3% | 70.7% | 82.4% | 11.1% |
| PP-OCRv5 (참조) | 0.8155 | 46.1% | 77.6% | 87.0% | 0.0% |
| PP-OCRv6 (참조) | 0.8458 | 49.7% | 81.8% | 90.4% | 0.0% |

- **seen/unseen 격차 31.4%p** (v6: seen 67.2% vs unseen 35.8%) — 전체 평균은 일반화를 14%p 과대평가
- **🔴 길이별 역설**: 긴 코드는 Exact 13.1%인데 **Top-5 94.1%**, 짧은 코드는 Exact 64.6%인데 Top-5 82.8%.
  DB 이웃 밀도 때문(긴 코드 0.179 vs 짧은 코드 0.370). → **TASK 5 합성 우선순위 재고 필요**
- ~~목표선: Exact ≥64% 승격~~ → **철회**. 인식 단독 기준이라 L3(시스템) 목표선으로 무효.
  지표 3계층 분리와 재정의된 목표선 → [`docs/metrics_policy.md`](metrics_policy.md)
- **🟡 Phase 0 재검토 트리거**: 2.x 선택 비용이 측정됨 — v4 대 v6 = Exact −14.4%p. TASK 6 1단계에서 판단

**검출 게이트** → [`reports/det_finetune_decision.md`](../reports/det_finetune_decision.md)

- 평가 318장(test 234 + **곡면 93 전량** — test 곡면이 9박스뿐이라 판단 불가)
- **평면 recall 71.1%** (임계 95%에 24%p 미달) → **검출 fine-tuning 필요, TASK 6-A 생략 불가**
- **곡면 recall 21.2%** — 99박스 중 78개 유실. **곡면 최대 병목은 인식이 아니라 검출**
- 🟡 착수 전 확인: 배포 config가 `resize_long: 960`으로 2560×1920을 **2.67배 축소**.
  해상도 스윕은 **로컬 CPU 성능 한계로 미완**(1536에서 40분 경과 후에도 표본 20장 미완) → **Colab GPU로 이연**
- precision 52.3%는 미라벨 텍스트 가능성이 있어 목표선 설정 보류. **판단 근거는 recall**

**🔴 Phase 5 후속 — end-to-end 실측으로 전제가 바뀜** → [`docs/metrics_policy.md`](metrics_policy.md)

- **실제 시스템 Top-5 = 61.9%** (검출 72.3% × 조건부 인식 85.6%). 보고해온 87.0%는 인식 단독이었음
- **손실의 73%가 검출.** 그중 **85%는 "못 찾은 것"이 아니라 "박스 경계 불일치"**(분할 56% + 부분 30%)
- 원인은 TASK 1-3의 **(a)안 박스 정책** — 2토큰 박스 검출 recall **46.8%** (단일 토큰 77.6%),
  폭 700px+ 는 59.6%. DBNet이 공백에서 쪼개는데 우리 GT는 합쳐져 있음
- **길이별 순위가 뒤집힘**: GT크롭 최고였던 9–14자(94.1%)가 시스템 최하위(50.3%) — 검출 recall 60.8%
- 자원 배분: 검출 +10%p → **+8.5%p** vs 인식 +10%p → +7.2%p. **검출 1순위 확정**
- 채택 제안: **검출·인식은 토큰 단위, DB 조회는 재결합 후** (게이트 검증 후 적용)
- 마일스톤은 이후 실측으로 재정의됨 → [`metrics_policy.md`](metrics_policy.md) 7장 (이 줄의 옛 값은 폐기)

**테스트 114건 통과** (Phase 5에서 13건 + end-to-end 7건)

---

### Phase 6 — 검출 병목 해소 시도 (M1 / M2 / M2-B) — ✅ **완료 (2026-08-03~04)**

근거 → [`reports/m1_m2_detection.md`](../reports/m1_m2_detection.md), [`reports/m2b_spatial_group.md`](../reports/m2b_spatial_group.md)

| 시도 | 결과 |
|---|---|
| **M1** 검출 후처리 스윕 (`unclip_ratio`/`box_thresh`/`thresh`) | ❌ **개선 0** — unclip 은 컨투어 추출 *뒤* 팽창이라 조각을 못 합침 |
| **M2** GT 로 조각 재결합 | ⚠️ 76.1% — **GT 필요, 배포 불가** (상한 측정치) |
| **M2-B** GT 없는 공간 그룹핑 | ❌ **62.9%** (+1.0%p) — 라벨 단위는 기하로 복원 불가 |

- 핵심 발견: `8196 P32`는 한 박스인데 옆의 `P32`는 다른 박스 —
  **"무엇이 한 박스인가"는 작성자의 의미론적 판단이라 간격·정렬 규칙으로 복원할 수 없다.**
- 0°/180° 공존(파이프 뒤집힘) 반영: 방향 분류기가 우리 도메인에서 신뢰 불가
  (뒤집으면 확신도 0.752→0.594) → `FLIP_POLICY="none"` 채택, 코드에 정책으로 남김
- **결론: 저비용 경로 전부 소진. 검출 fine-tuning 이 필요하고 올바른 도구다**
  (박스 규약은 후처리로는 복원 불가하지만 **학습은 가능**하기 때문)

---

### Phase 7 — Colab 학습 준비 — ✅ **완료 (2026-08-04)**

절차 → [`docs/colab_setup.md`](colab_setup.md) / 노트북 → [`notebooks/train_det.ipynb`](../notebooks/train_det.ipynb)

- GitHub 원격 연결 (`ocngrn/label_ocr`, 코드 3.6MB / 이미지 미추적)
- `src/configs/build_det_config.py` — `spec.py` 기반 config 생성 (실제 2.10.0 템플릿으로 검증)
- `label_ocr_images.tar.gz` 생성 완료 (1,369.4MB, 엔트리 2,344)

---

## 🔴 현재 위치 및 다음 행동

**시스템 Top-5 = 62.9%** (검출 73.1% × 조건부 인식 86.0%) / 목표 ≥95%

### 다음 단 하나의 행동
1. `label_ocr_images.tar.gz` 를 Drive `MyDrive/label_ocr/` 에 업로드 ← **사용자 작업**
2. Colab 에서 `notebooks/train_det.ipynb` 실행 (런타임 GPU)
3. **셀 4 게이트**(GPU 호환성) → **셀 6 게이트**(v4 baseline) → 셀 7 학습

### 열린 스레드 (발동 조건과 함께)

| # | 항목 | 발동 조건 / 시점 | 근거 문서 |
|---|---|---|---|
| 1 | **L3 재측정 어댑터 미구현** | 학습 직후 필수. `end_to_end.py` 는 paddleocr 3.7 API 인데 학습 산출물은 2.x export | `colab_setup.md` |
| 2 | **Phase 0 재검토 (2.x vs 3.x)** | M3 에서 fine-tuned v4 가 무조정 v6(Exact 49.7%)를 못 넘으면 | `baseline_v1.md` §7 |
| 3 | `limit_side_len` 스윕 미완 | GPU 확보 시 (로컬 CPU 로는 20장 40분에도 미완) | `det_finetune_decision.md` §4 |
| 4 | 검출 후처리 재스윕 | 학습 후 출력 분포가 바뀌면 | `build_det_config.py` docstring |
| 5 | `FLIP_POLICY` 재검토 | 도메인 데이터로 방향 분류기를 학습한 뒤 | `m2b_spatial_group.md` §2 |
| 6 | 곡면 5-Fold 미사용 | 최종 후보 1~2개로 좁힌 뒤 확정 검증에만 | `split_stats.md` §4 |
| 7 | 인식 학습 가중 = 매칭 레버리지의 역수 | M4 착수 시 (길이 1–4 상향, 9+ 하향) | `metrics_policy.md` §5-B |
| 8 | TASK 5 합성 우선순위 재조정 | 긴 코드 생성 **하향**(시스템 기여 9%), 짧은 코드 강건성 상향 | `baseline_v1.md` §3 |
| 9 | `serial_db_proxy_v2` | 부품 마스터 확보 시 → Top-K 목표선 재보정 | `db_matching_v1.md` §1 |

### 로컬 실행 규약 (대화에만 있던 것 — 여기 고정)

```bash
PYTHONIOENCODING=utf-8      # 한글 출력 시 cp949 UnicodeEncodeError 방지
KMP_DUPLICATE_LIB_OK=TRUE   # Anaconda OpenMP 중복 로드 (Colab 불필요)
```
- 검출 추론은 `TextDetection(..., enable_mkldnn=False)` 필수 — paddle 3.3.1 CPU 에서 oneDNN 경로가 깨진다
- Windows `tar` 는 후행 슬래시 금지 (`image_set/` ✗ → `image_set` ✓)

### 재생성 비용이 큰 gitignored 산출물

| 산출물 | 재생성 명령 | 비용 |
|---|---|---|
| `crops/` (233MB) | `python -m src.preprocess.build_labels` | 수 분 |
| `reports/detect_cache_test.json` | `python -m src.eval.spatial_group cache` | **약 45분** (검출) |
| `predictions/*.json` | 각 eval 모듈 재실행 | 모듈당 3~50분 |

---

### Phase 8 이후 — TASK 5 → 6(1~5단계) → 7

지시서 순서를 그대로 따른다. 단 Phase 0 결과에 따라 다음 조정 적용:

- **TASK 4**: 로컬 CPU 추론 우선 시도. baseline은 **PP-OCRv4** server det+rec(2.x). 별도로 **PP-OCRv5/v6 참조 수치**를 설치본 `paddleocr` 3.7.0으로 병기(학습에는 미사용).
- **TASK 6 1단계**: `rec_svtrnet.yml` (**STN_ON 내장** — §7-② 참조), `out_char_num=64`.
- **TASK 6 4단계(TPS)**: `rec_r34_vd_tps_bilstm_att.yml` (RARE). 가중치 확보 완료 → **계획 유지 확정**.
- **TASK 6 config 공통**: `max_text_length=40`, `rec_image_shape=[3,64,256]`, `use_space_char=True`, `character_dict_path=configs/dict.txt`.
- **TASK 5**: LMDB 패킹 (lmdb 1.7.5 설치 완료). 크롭 규격 `64×256` 일치 강제. 로컬 생성 후 Drive 압축 전송.
- **Colab**: `paddlepaddle-gpu`는 **3.3.x 계열 핀 고정**. 착수 시 Phase 0 스모크 테스트를 GPU에서 재실행.
- **로컬 실행 시**: `KMP_DUPLICATE_LIB_OK=TRUE` 필요 (Anaconda OpenMP 중복 로드). Colab은 불필요.

---

## 6. 지시서 대비 변경 요약

| # | 지시서 기재 | 실측 결과 | 조치 |
|---|---|---|---|
| 1 | TASK 0 이미지 정합성 게이트 | 누락·미라벨·손상 **전부 0건** | 게이트 통과. TASK 0은 스캐폴딩만 |
| 2 | 문자셋에 `'` 없음 | `'` **36회 실존** | `dict.txt`에 포함 |
| 3 | `max_text_length` 24자+ | 실측 최댓값 **32자** | **40**으로 설정 |
| 4 | 고유 시리얼 4,281 | **4,280** | `\|DB\|` 실측값 사용 |
| 5 | 곡면 크롭 직접 구현 필수 | PaddleX `CropByPolys(poly)` 존재 | 재검증 후 판단 (육안 확인은 유지) |
| 6 | 전 TASK Colab 수행 | 로컬에 스택 완비, GPU만 부재 | 학습(TASK 6)만 Colab |
| 7 | PaddleOCR 2.x 인터페이스 전제 | 설치본은 3.7 추론 전용 + PaddleX | **Phase 0 최우선 게이트로 신설** |
| 8 | (언급 없음) | git 저장소 아님 | Phase 1에서 `git init` |
| 9 | (언급 없음) | 빈 transcription 1건 | Phase 2에서 판정 |
| 10 | `System_Architecture.md` 참조 | 파일 부재 | 이번 범위 외 (앱 단계로 이연) |
| 11 | PP-OCRv5 baseline + RARE 4단계 동시 요구 | 두 세대에 나뉘어 **양립 불가** | **2.10.0 채택**, v5/v6은 참조 수치로만 (Phase 0) |
| 12 | 기본 인식 입력 `[3,32,100]` | CTC 시퀀스 25 < **최대 라벨 32자** | `[3,64,256]` + `out_char_num=64` (Phase 0 §5) |
| 13 | "SVTR은 rectification 미내장" | `rec_svtrnet.yml`에 **STN_ON 내장**(20 제어점) | 1단계부터 정류 확보. 4단계는 "CTC vs attention" 비교로 재정의 |

---

## 7. Phase 0 파생 결정 (상세: [`framework_decision.md`](framework_decision.md))

**① 인식기 입력 규격 — `[3, 64, 256]`, 시퀀스 64, `max_text_length=40`**

CTC는 출력 타임스텝 ≥ 라벨 길이여야 한다. 기본 폭 100은 타임스텝 25뿐이라 **24자 이상 29개 박스(최장 32자)의 정답이 구조적으로 도달 불가능**하다 — 그런데도 손실은 계산되어 학습은 "성공"처럼 보인다. 실측 매핑: 폭 100→25, 160→40, 256→**64**, 320→80. 반복 문자용 blank 여유를 위해 최대 길이의 2배인 64를 채택.

**② SVTR의 STN_ON 내장 — 4단계의 의미 재정의**

`rec_svtrnet.yml`은 `Transform: STN_ON`(TPS 기반, 제어점 20개)을 이미 포함한다. 지시서의 "SVTR은 rectification 미내장"은 **PP-OCRv5 rec(SVTR_LCNet)** 기준이며, 2.x의 원조 SVTR에는 해당하지 않는다.
→ 1단계부터 곡면 정류가 들어간다. 4단계 RARE 비교는 "정류 유무"가 아니라 **"STN+CTC vs TPS+attention 디코더"** 비교가 된다. 긴 코드에 attention이 유리하다는 기획서 5장 논거는 유효하므로 4단계는 유지.
