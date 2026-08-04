# M3 — 검출 fine-tuning [목표 미달, 유의미한 개선]

> 실측 일자: 2026-08-04 / Colab A100-80GB / PP-OCRv4_det_server (PaddleOCR 2.10.0)
> 평가: test 234장 + 곡면 93장 = **318장, GT 1022박스**, ICDAR IoU>=0.5 그리디 1:1

---

## ⚠️ 정정 (2026-08-05) — 학습후 수치는 학습 데이터에 오염돼 있다

`detect_baseline` 은 평가 대상에 **곡면 이미지 93장을 전량** 끌어온다. 곡면 박스가 test 에
9개뿐이라 곡면 판단이 불가능해서 넣은 장치이고, 당시 근거는 "사전학습 모델이라 학습이
일어나지 않았으므로 누수가 없다"였다. **fine-tuning 이후 그 전제가 깨졌다.**

```
곡면 이미지 93장 = train 75 / val 9 / test 9
평가 대상 318장의 GT 1022박스 = test 762 / train 236 / val 24
                                          ↑ 23.1% 가 학습 데이터
```

| 서브셋 | 전체 | 그중 train 출처 | 오염도 |
|---|---|---|---|
| plane | 923 | 155 | 16.8% |
| curved | 99 | 81 | **81.8%** |

- **사전학습 행(67.0% / 74.2%)은 유효하다** — 학습이 없었으므로 누수 개념이 성립하지 않는다.
- **학습후 행(73.5% / 77.1% / 78.5%)은 낙관 편향** 이다. 방향(개선했다)은 유지되지만
  크기는 과대평가다. plane 은 16.8%만 오염돼 편향이 작고, **곡면 24.2% 는 사실상
  학습 데이터 위의 수치이므로 폐기해야 한다.**
- `evaluate(..., extra_curved=False)` 가 기본값이 되도록 고쳤다(2026-08-05).
  **재측정은 GPU 재확보 시 후처리 스윕과 함께 1회로 처리한다.**
- 곡면 성능은 곡면 5-Fold(`split_stats.md` §4)로만 정직하게 잴 수 있다.

> 이 결함은 지표 코드가 **모델의 학습 여부를 모른 채** 평가 대상을 정한 데서 왔다.
> 대상 선택이 모델 상태에 의존하면, 그 의존성을 인자로 드러내야 한다.

---

## 결론

| | 시스템 Top-5 (추정) |
|---|---|
| 이전 최선 (M2-B, PP-OCRv5) | 62.9% |
| **M3 도달** | **67.2%** |
| M3 목표 | ≥81% |

**목표 미달.** 다만 개선은 실재하고, 두 개의 **독립적인** 원인으로 분해된다.

> `시스템 Top-5 추정 = plane recall x 85.6%(조건부 인식)`. 정식 L3 는 전체 파이프라인을
> 다시 돌려야 나온다. 여기서는 검출만 바뀌었으므로 조건부 인식률을 고정으로 둔 추정치다.

---

## 1. 전체 측정 — 모두 같은 경로·같은 318장·같은 매칭

| 가중치 | 입력 해상도 | plane recall | precision | Hmean | 곡면 | 예측수 |
|---|---|---|---|---|---|---|
| 사전학습 | 960/max | 67.0% | 52.7% | 57.1% | 19.2% | 1208 |
| 사전학습 | 원본 2560x1920 | 74.2% | 52.8% | 59.9% | 21.2% | 1336 |
| **학습후** | 960/max | 73.5% | **76.2%** | **72.4%** | 26.3% | 924 |
| **학습후** | 1920/max | 77.1% | 68.9% | 70.6% | 27.3% | 1072 |
| **학습후** | **원본 2560x1920** | **78.5%** | 56.5% | 63.8% | 24.2% | 1326 |

### 두 효과는 서로 독립이다

- **해상도** (960/max → 원본): 사전학습에서 **+7.2%p**, 학습후에서 **+5.0%p**
- **fine-tuning** (해상도 고정): 960/max 에서 **+6.5%p**, 원본에서 **+4.3%p**
- 합계: 67.0% → 78.5% = **+11.5%p**

기존 baseline(66.8%)이 낮았던 이유의 절반은 학습 부족이 아니라 **입력 해상도**였다.
`limit_side_len` 상향은 `metrics_policy` 5-A 의 2번 항목으로 이연돼 있었는데, 로컬 CPU 로는
잴 수 없어 미검증으로 남아 있었다. GPU 를 얻자마자 확인됐다.

### precision 이 규약 학습의 증거다

960/max 에서 precision 이 **52.7% → 76.2%** 로 뛰었다(+23.5%p). 예측 수도 1208 → 924 로
줄었다. GT 는 그대로인데 예측이 줄고 정확해진 것은, 모델이 `8196 P32` 를 두 박스로 쪼개던
것을 한 박스로 내보내기 시작했다는 뜻이다 — **의미론적 박스 규약을 실제로 배웠다.**
`reports/m2b_spatial_group.md` 1장에서 "기하 규칙으로는 원리적으로 복원 불가"라고 적었던
바로 그 구분이다.

### 해상도–recall–precision 트레이드오프

해상도를 올리면 recall 이 오르고 precision 이 떨어진다(76.2% → 56.5%).
우리 시스템 KPI 는 **GT 박스마다 Top-5 를 맞추는가**이므로 recall 이 지배한다.
따라서 운영 설정은 **원본 해상도**다. 다만 예측 1326개 중 577개가 GT 와 매칭되지 않으므로,
사용자에게 보여줄 후보를 줄이려면 별도의 필터가 필요하다 — 이는 UI 층 과제다.

---

## 2. 학습 과정에서 잡은 버그 2건

두 버그 모두 **학습이 도는 것처럼 보이면서** 결과를 무의미하게 만든다.

### (1) AMP 가 DBLoss 의 Dice 항을 죽인다

300스텝 내내 손실이 소수점 6자리까지 고정됐다:

```
loss_shrink_maps: 5.000000   loss_binary_maps: 1.000000   loss_cbn: 1.000000
```

`union = sum(pred * mask)` 는 배치 4 x 640 x 640 = 164만 픽셀의 합이라 fp16 상한 65504 를
넘어 `inf` 가 된다. `intersection` 은 텍스트 픽셀(약 2%)만 더하므로 정상 범위다. 따라서
`1 - 2 * intersection / inf` 가 **정확히 1.0** 이 되고 기울기가 0 이 된다.
`loss_threshold_maps` 만 정상으로 보였던 것은 그것이 마스크 합으로 나누는 L1 이라
오버플로가 없기 때문이다. PaddleOCR 의 `amp_level` 기본값은 O2(순수 fp16)다.

**순전파 출력은 fp32/O1/O2 모두 정상이다**(`maps` mean 0.15). 순전파만 확인해서는
발견되지 않는다. 판정은 **손실이 실제로 내려가는가**로 해야 한다.

### (2) 학습률 1e-3 은 사전학습 특징을 파괴한다

| | precision | recall | hmean (val 234장) |
|---|---|---|---|
| 사전학습 | 0.5835 | 0.7013 | 0.6370 |
| lr 1e-3, 6 epoch | 0.5451 | 0.6732 | **0.6024** ↓ |
| lr 1e-4, best(24 epoch) | 0.6398 | 0.7564 | **0.6932** ↑ |

템플릿 기본 1e-3 은 PaddleOCR 이 10만 장급 코퍼스용으로 잡은 값이다. 우리 학습셋은
1,872장이고, 가르쳐야 할 것은 글자 검출 자체가 아니라 **박스 규약 하나**다.

> 이 발견은 **셀 6 게이트가 없었으면 불가능했다.** 학습 전 값을 같은 지표로 재 두지
> 않았다면 `hmean 0.6024` 를 성과로 오독했을 것이다.

### 부수 발견: 50 epoch 은 과했다

best 는 **epoch 24** 에서 나왔고 이후 26 epoch 동안 갱신이 없었다(0.645~0.668 배회).
"더 오래 돌리기"는 남은 지렛대가 아니다.

---

## 3. 재현 방법

```bash
# 학습
python -m src.configs.build_det_config --template <PaddleOCR>/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml \
  --out configs/det_ppocrv4_server.yml --root /content/label_ocr \
  --weights /content/weights/ch_PP-OCRv4_det_server_train/best_accuracy \
  --save-dir <drive>/experiments/det_v4_server --epochs 50 --batch-size 8 --lr 0.0001
cd <PaddleOCR> && python tools/train.py -c /content/label_ocr/configs/det_ppocrv4_server.yml

# 측정 (학습 전/후를 같은 경로로)
python -m src.eval.detect_trained --weights <exp>/best_accuracy --tag det_v4_finetuned
python -m src.eval.detect_trained --weights <weights>/best_accuracy --tag det_v4_pretrained
```

### export 는 쓸 수 없다

`tools/export_model.py` 가 paddle 3.3 에서 깨진다:
`TypeError: sigmoid(): argument (position 1) must be Value, but got Variable`.
2.10.0 의 export 는 구 정적그래프(Variable)를 쓰는데 paddle 3.x 는 PIR(Value)로 넘어갔고,
`FLAGS_enable_pir_api=0` 으로도 되돌아가지 않는다. `src/eval/detect_trained.py` 는 dygraph 로
직접 추론해 이를 우회한다 — 학습·평가와 같은 코드 경로라 교란도 적다.

### `limit_type='min'` 은 우리 데이터에서 무효다

`DetResizeForTest` 의 `limit_type='min'` 은 짧은 변이 기준값보다 **작을 때만 확대**한다.
우리 이미지는 2560x1920 이라 736/1120/1600 어느 값이든 `ratio=1.0` — 셋 다 원본 그대로
돌아 결과가 완전히 동일했다(TP=725). 해상도를 낮추려면 `limit_type='max'` 를 써야 한다.

---

## 4. 남은 격차와 다음 수단

목표 81% 는 plane recall **약 95%** 를 요구한다. 현재 78.5%, 격차 16.5%p.

| 후보 | 근거 | 비용 |
|---|---|---|
| **학습/추론 스케일 정합** | 640 크롭으로 학습하고 2560 원본으로 추론한다. `EastRandomCropData` 크기를 올리거나 추론 해상도를 학습 분포에 맞추면 이득 가능 | 학습 1회 (~2시간) |
| **후처리 재스윕** | M1 스윕은 사전학습 PP-OCRv5 분포에서 했다. 학습으로 출력 분포가 바뀌었으므로 `box_thresh`/`unclip_ratio` 재탐색 | GPU 30분, 학습 불필요 |
| **곡면 전용 대응** | 곡면 recall 이 24.2% 로 여전히 최악. 다만 GT 99박스(9.7%)라 전체 기여는 제한적 | 별도 판단 |
| Phase 0 재검토 (2.x → 3.x) | v4 fine-tuned(78.5%)가 v5 untuned 를 넘었으므로 **현재는 불필요** | — |

**우선순위: 후처리 재스윕 → 스케일 정합.** 전자는 학습 없이 30분이고, M1 때와 달리 이번엔
분포가 실제로 바뀌었으므로 개선 여지가 있다.

---

## 5. 마일스톤 갱신

| 마일스톤 | 목표 | 결과 |
|---|---|---|
| ~~M1 검출 후처리~~ | ≥72% | 61.9% (개선 0) |
| ~~M2-B 토큰 단위 전환~~ | ≥81% | 62.9% (+1.0%p) |
| **M3 검출 fine-tuning** | ≥81% | **67.2%** (+4.3%p) — 미달 |
| M4 인식 fine-tuning | ≥87% | |

---

## 6. 재현성 기록

| 항목 | 값 |
|---|---|
| 산출물 | `reports/det_v4_{pretrained,finetuned,pre_max960,ft_max960,ft_max1920}.json` |
| 체크포인트 | Drive `experiments/det_v4_server/best_accuracy` (epoch 24) — **유일본** |
| 체크포인트 md5 | `172069b843611f08c590d15a8ef92046` (113,981,655B, 파라미터 367, NaN 없음) |
| 폐기된 실험 | Drive `experiments/det_v4_lr1e-3_aborted` (lr 1e-3, hmean 하락) |
| 학습 설정 | Adam + Cosine(1e-4, warmup 2), DBLoss, batch 8, 50 epoch, **AMP off** |
| 데이터 | train 1872 / val 234 / `label_snapshot_v2` / 분할 시드 42 |
| 소요 | 학습 약 2시간 (A100, 2.4분/epoch) |
