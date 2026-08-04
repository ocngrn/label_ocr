# Colab 실행 준비

> 노트북: [`notebooks/train_det.ipynb`](../notebooks/train_det.ipynb)
> 대상: TASK 6-A / M3 검출 fine-tuning (시스템 Top-5 62.9% → ≥81%)

## 사전 준비 (로컬 PC, 1회)

```powershell
cd O:\Project\label_ocr
tar -czf label_ocr_images.tar.gz image_set      # 약 1.34GB, 수 분 소요
```

> **후행 슬래시를 붙이지 말 것.** Windows 의 bsdtar 에서 `image_set/` 처럼 `/` 를 붙이면
> 인자 파싱이 깨져 엉뚱한 경로를 입력으로 잡는다:
> `tar.exe: ...ta\Roaming\npm: Couldn't visit directory: No such file or directory`

생성된 `label_ocr_images.tar.gz` 를 Google Drive 의 `MyDrive/label_ocr/` 에 업로드한다.

**크롭(`crops/`, 233MB)은 올리지 않는다** — 노트북 셀 5가 `build_labels` 로 재생성한다.
코드는 GitHub(`ocngrn/label_ocr`)에서 `git clone` 하므로 전송할 필요가 없다.

### GitHub 인증 (private 저장소)

`ocngrn/label_ocr` 는 private 이라 Colab 런타임에서 익명 clone 이 실패한다:

```
fatal: could not read Username for 'https://github.com': No such device or address
```

셀 3이 이를 감지하면 **PAT 입력창**(`getpass`)을 띄운다. 토큰은 노트북 파일에도 셀 출력에도
남지 않지만, 세션이 초기화될 때마다 다시 입력해야 한다.
매번 입력하기 싫으면 저장소를 public 으로 전환한다 — 이미지·라벨·크롭은 저장소에 없다.

> Colab 의 "GitHub 에서 노트북 열기"는 Colab 자체 OAuth 라 private 도 열리지만,
> 그 자격증명은 **런타임의 `git` 에 상속되지 않는다.** 노트북이 열렸다고 clone 이 되는 게 아니다.

## 실행 순서

Colab 에서 런타임 유형을 **GPU** 로 바꾼 뒤 노트북을 위에서부터 실행한다.

| 셀 | 내용 | 비고 |
|---|---|---|
| 1 | GPU 확인 + `paddlepaddle-gpu` 설치 | CUDA 버전에 맞춰 인덱스 URL 수정 |
| 2 | Drive 마운트 + 이미지를 `/content` 로 해제 | Drive 직접 읽기는 DataLoader 가 느려진다 |
| 3 | 저장소 clone + PaddleOCR 2.10.0 + 가중치 | |
| **4** | **[게이트] Phase 0 스모크 테스트 GPU 재실행** | 실패 시 중단 |
| 5 | 크롭 재생성 + 테스트 141건 | |
| **6** | **[게이트] PP-OCRv4 det baseline 재측정** | 건너뛰면 학습 효과 측정 불가 |
| 7 | config 생성 + 학습 | 세션 끊기면 `RESUME` 켜고 재실행 |
| 8 | 평가 (L2 Hmean) + export | |

## 왜 게이트가 두 개인가

**셀 4** — `docs/framework_decision.md` 의 호환성 검증은 **CPU 에서만** 했다.
PaddleOCR 2.10.0(2021년 세대 코드)이 paddlepaddle 3.3.x **GPU 빌드**에서 도는지는 미검증이다.
여기서 깨지면 그 뒤 전부 무의미하므로 GPU 시간을 쓰기 전에 확인한다.

**셀 6** — 지금까지 모든 측정은 `PP-OCRv5_server_det` 로 했는데, 학습은 Phase 0 결정에 따라
2.x(최대 PP-OCRv4)에서 이뤄진다. 같은 계열로 기준선을 다시 잡지 않으면
**"학습 덕분에 오른 것"과 "모델 세대가 달라서 내린 것"이 섞여** 판정이 불가능해진다.

## 세션이 끊겼을 때

Colab 은 런타임이 수시로 초기화된다. 체크포인트는 Drive(`experiments/det_v4_server/`)에
`save_epoch_step=5` 로 저장되므로:

1. 셀 1~3 재실행 (환경·데이터 복구)
2. 셀 7 의 `RESUME` 을 `f"{DRIVE}/experiments/det_v4_server/latest"` 로 설정
3. 셀 7 재실행

## 알려진 제약

- **L3(end-to-end) 재측정은 노트북 범위 밖이다.** `src/eval/end_to_end.py` 는 `paddleocr` 3.7 의
  `TextDetection` 을 쓰는데, 학습 결과는 2.x 로 export 한 추론 모델이라 어댑터가 필요하다.
  노트북에서는 L2(검출 recall/Hmean)로 방향을 확인하고,
  `시스템 Top-5 ≈ 검출 recall × 85.6%` 로 추정한다. 정식 L3 는 어댑터를 붙인 뒤 측정한다.
- PaddleOCR 2.10.0 의 `requirements.txt` 가 Colab 의 numpy 2.x 와 충돌할 수 있다.
  셀 1에서 설치 로그를 확인하고, 충돌 시 필요한 패키지만 개별 설치한다.
- `paddlepaddle-gpu` 는 PyPI 에 없다(확인함). Baidu 공식 인덱스 URL 이 필요하다.
