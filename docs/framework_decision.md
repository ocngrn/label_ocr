# Phase 0 — 프레임워크 세대 확정 결정서

> 지시서 `claude_code_prompt_paddleocr_colab.md` TASK 6이 전제한 인터페이스 세대와, 실제 설치·가용 환경 사이의 불일치를 해소하기 위한 결정.
> 실측 일자: 2026-07-31 / 환경: `o:\Project\label_ocr`, Windows 11, Python 3.13.9, paddlepaddle 3.3.1 (CPU)

---

## 1. 결론

### **PaddleOCR 2.10.0 (2.x dygraph) 채택**

보조로 설치본 `paddleocr` 3.7.0을 **참조 baseline 측정 전용**(추론만)으로 병행 사용한다.

---

## 2. 결정 근거 — 지시서의 계획은 단일 세대로 실현 불가능

지시서는 두 가지를 동시에 요구하는데, **서로 다른 세대에만 존재한다**:

- TASK 4 / 6-A·6-B: **PP-OCRv5** server det + rec 를 baseline·사전학습으로 사용 → **3.x 전용**
- TASK 6 4단계: **RARE(TPS 내장)** 로 교체 실험 → **2.x 전용**

실측 비교:

| 항목 | PaddleOCR **2.10.0** | PaddleOCR 3.7 / PaddleX 3.7.2 |
|---|---|---|
| 학습 진입점 `tools/train.py` | ✅ 존재 (`tools/train.py`, `eval.py`, `export_model.py`) | ❌ 없음 (패키지가 추론 전용, 학습은 PaddleX CLI) |
| RARE / StarNet (TPS 내장) | ✅ `configs/rec/rec_r34_vd_tps_bilstm_{att,ctc}.yml`, `rec_mv3_tps_bilstm_att.yml` | ❌ **전무** |
| TPS 구현체 | ✅ `ppocr/modeling/transforms/{tps.py, tps_spatial_transformer.py, stn.py}` | ❌ **전무** (word-boundary 검색 0건) |
| PP-OCRv5 | ❌ 없음 (최대 **PP-OCRv4**) | ✅ |
| PP-OCRv6 | ❌ | ✅ (지시서가 모르는 최신 세대) |
| SVTR | ✅ `rec_svtrnet.yml`, `SVTRv2` | ✅ `ch_SVTRv2_rec`, `ch_RepSVTR_rec` |
| paddlepaddle 3.3.1 호환 | ✅ **실측 검증 완료**(아래 4장) | ✅ (네이티브) |

→ **4단계(TPS)를 계획에 유지하려면 2.x 외에 선택지가 없다.** 3.x를 택하면 지시서 TASK 6의 5단계 중 4단계가 통째로 삭제되고, F1 폴백이 자동 발동한 것과 같아진다.

### PP-OCRv5 → v4 다운그레이드가 수용 가능한 이유
TASK 4 baseline은 **목표선을 정하기 위한 출발점**일 뿐이고, 지시서 원칙상 학습 전후 비교는 **동일 모델 계열 내에서만** 유효하다. fine-tuning 대상이 2.x 계열이면 baseline도 같은 계열이어야 비교가 정직하다.
다만 "범용 최신 OCR이 이 데이터에서 어느 정도인가"라는 참고 수치는 유용하므로, 설치된 `paddleocr` 3.7.0으로 **PP-OCRv5/v6 참조 baseline을 별도 측정**해 리포트에 병기한다(학습에는 사용하지 않음).

---

## 3. 사전학습 가중치 생존 확인 — 지시서 TASK 6-B 게이트 **통과**

지시서가 "링크 생존 확인만 남았다"고 한 항목:

| 가중치 | HTTP | 판정 |
|---|---|---|
| `rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar` (StarNet) | **200** | ✅ |
| `rec_r34_vd_tps_bilstm_att_v2.0_train.tar` (RARE) | **200** | ✅ |
| `rec_mv3_tps_bilstm_att_v2.0_train.tar` (RARE MobileNetV3) | **200** | ✅ |
| `en_PP-OCRv4_rec_train.tar` | **200** | ✅ |
| `ch_PP-OCRv4_rec_server_train.tar` | **200** | ✅ |
| `ch_PP-OCRv4_det_server_train.tar` | **200** | ✅ |

→ **F1 폴백(RARE 가용 불가) 사전 발동 불필요.** 4단계 TPS 실험을 계획에 유지한다.

---

## 4. 호환성 실측 — PaddleOCR 2.10.0 × paddlepaddle 3.3.1

가장 큰 위험(구세대 코드가 최신 paddle에서 안 도는 경우)을 스모크 테스트로 확인했다.

### 4-1. 모델 빌드 + forward
```
rec_r34_vd_tps_bilstm_att.yml (RARE)     OK -> [1, 25, 40]
rec_r34_vd_tps_bilstm_ctc.yml (StarNet)  OK -> [1, 25, 40]
rec_svtrnet.yml (SVTR)                   OK -> [1, 25, 40]
```
> 주의: 한 프로세스에서 TPS 모델을 두 번 빌드하면 `parameter name [loc_conv0_weights] have be been used` 오류가 난다. **각 모델은 별도 프로세스에서 빌드**할 것(학습 스크립트는 1프로세스 1모델이므로 실사용에는 무영향).

### 4-2. 사전학습 가중치 실제 로드 (Phase 0 종료 조건)
RARE 가중치(180MB) 다운로드 → 로드:
```
model params : 263
  transferred: 260
  shape-mismatch: 3  -> head.attention_cell.rnn.weight_ih, head.generator.weight, head.generator.bias
  missing     : 0
forward after load OK -> [1, 25, 40]
```
- **missing 0** → config와 체크포인트 아키텍처가 정확히 일치.
- shape-mismatch 3건은 **전부 Head** → 지시서 6-B가 예고한 "문자 클래스 수 불일치로 최종 헤드 자동 재초기화, 정상 동작"과 일치. 백본·TPS·인코더는 260개 파라미터 전량 전이됨.

### 4-3. 로컬 환경 주의사항 (Colab 무관)
Anaconda에서 `OMP: Error #15: libiomp5md.dll already initialized` 발생 → **`KMP_DUPLICATE_LIB_OK=TRUE`** 환경변수 필요. Colab(Linux)에서는 발생하지 않음.

---

## 5. 🔴 파생 발견 — 인식기 입력 규격을 지금 확정해야 함

Phase 0 검증 중, **우리 데이터의 최대 라벨 길이 32자를 기본 config가 표현할 수 없음**을 발견했다.

### 문제
CTC는 출력 타임스텝 수 ≥ 라벨 길이여야 한다. 기본 인식 입력은 `[3, 32, 100]`인데:

| 모델 | 입력 폭 → 시퀀스 길이 | 32자 표현 |
|---|---|---|
| StarNet/R34 (CTC) | 100→**25**, 160→40, 256→**64**, 320→80 | 폭 100에서 **불가** |
| SVTR | 폭 무관, `Backbone.out_char_num`으로 결정 (기본 **25**) | 기본값 **불가** |

→ 기본값 그대로 학습하면 **24자 이상 29개 박스(최장 32자 `8173LCB - 8 - G23S CT410 -'10EA'`)의 정답이 구조적으로 도달 불가능**하다. 손실도 조용히 계산되어 학습은 "성공"한 것처럼 보인다.

### 결정
```
rec_image_shape       : [3, 64, 256]
SVTR out_char_num     : 64
Global.max_text_length: 40
```
- 폭 256 → CTC 타임스텝 64 = 최대 라벨 길이 32의 **2배**. CTC는 반복 문자 사이에 blank가 필요하므로(`'10EA'`, `CT410` 등 반복 존재) 2배 여유가 안전 기준이다. 폭 160(40 타임스텝)은 여유가 8뿐이라 채택하지 않는다.
- 실측 검증: `out_char_num=64`, 입력 `64×256` → 시퀀스 **64** 확인.

### 이 결정이 Phase 0에 속해야 하는 이유
Phase 2가 인식용 크롭 **8,103개**를 생성하고 Phase 5(TASK 5)가 합성 크롭을 **동일 규격으로** 만들어야 한다. 뒤에서 바꾸면 양쪽을 전부 재생성해야 한다.

---

## 6. 🟡 파생 발견 — 지시서의 "SVTR은 rectification 미내장" 전제 정정

지시서 6-B는 B안(SVTR)을 "최신·고정확 backbone이나 **rectification 미내장**"으로 규정하고, 그래서 4단계에 RARE(TPS)가 필요하다고 논증한다.

그러나 2.x의 `configs/rec/rec_svtrnet.yml`은 **`Transform: name: STN_ON`을 이미 포함**한다:
```yaml
Transform:
  name: STN_ON
  tps_inputsize: [32, 64]
  tps_outputsize: [32, 100]
  num_control_points: 20
```
즉 원조 SVTR config는 **TPS 기반 공간 변환 정류(20 제어점)를 내장**하고 있다.

- 지시서의 서술이 틀린 것은 아니다 — 그 문장은 **PP-OCRv5 rec(SVTR_LCNet)** 기준이고, 그쪽은 STN이 없다. 우리가 2.x를 택하면서 **원조 SVTR을 쓸 수 있게 되어 전제가 바뀐 것**이다.
- **함의**: 1단계부터 곡면 정류가 들어간다. 4단계 RARE 비교의 의미는 "정류 유무"가 아니라 **"STN+CTC vs TPS+attention 디코더"** 비교로 바뀐다. 긴 코드(20자+)에 attention이 CTC보다 유리하다는 기획서 5장 논거는 그대로 유효하므로 **4단계는 계속 수행할 가치가 있다**.

---

## 7. 후속 계획 반영 사항

| 대상 | 조정 |
|---|---|
| Phase 2 (TASK 1) 크롭 | 크롭 저장 규격을 **64×256 정렬**로 생성 (§5) |
| Phase 5 (TASK 5) 합성 | 동일 규격 강제. LMDB 패킹 시 동일 |
| TASK 4 baseline | **PP-OCRv4** server det+rec (2.x)로 측정. **+ PP-OCRv5/v6 참조 수치**를 3.7로 병기 |
| TASK 6 1단계 | `rec_svtrnet.yml` (STN_ON 포함) 기반, `out_char_num=64` |
| TASK 6 4단계 | `rec_r34_vd_tps_bilstm_att.yml` (RARE) — 가중치 확보 완료 |
| TASK 6 config 공통 | `max_text_length=40`, `character_dict_path=configs/dict.txt`, `use_space_char=True` |
| 로컬 실행 | `KMP_DUPLICATE_LIB_OK=TRUE` 필요 |
| Colab | paddlepaddle-gpu는 **3.3.x 계열로 핀 고정** (2.x 코드와 호환 검증 완료된 버전) |

---

## 8. 미해결 / 후속 확인 대상

1. **Colab GPU에서의 재검증**: 본 검증은 전부 CPU(paddlepaddle 3.3.1 CPU 빌드)에서 수행했다. `paddlepaddle-gpu==3.3.x` + Colab CUDA 조합은 TASK 6 착수 시 동일 스모크 테스트로 재확인할 것.
2. **PP-OCRv4 det의 폴리곤 출력 확인**: 지시서는 DBNet 폴리곤 검출을 요구한다. v4 det config에서 `det_box_type: poly` 동작을 TASK 4에서 확인.
3. **`rec_svtrnet.yml`의 사전학습 가중치**: RARE는 확보했으나 SVTR(원조, STN_ON 포함) 가중치 URL은 미확인. 1단계 착수 전 확인 필요 — 없으면 SVTRv2 또는 PP-OCRv4 rec로 1단계 백본 대체 판단.
