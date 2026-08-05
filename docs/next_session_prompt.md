# 새 세션 시작 프롬프트

아래 블록을 새 Claude Code 세션에 그대로 붙여넣는다.
(이 파일 자체를 갱신해 두면 다음 인계에도 재사용된다.)

---

```
label_ocr 프로젝트를 이어서 진행한다. 먼저 docs/work_plan.md 의
"🔴 현재 위치 및 다음 행동" 절을 읽고 시작해.

## 지금 상태 (한 줄씩)

- M3 검출 fine-tuning 완료. plane recall 78.9% (test 234장, 누수 없음), 목표 81% 미달
- 검출 쪽 저비용 수단은 소진됨 — 후처리 스윕 ❌, 해상도 하향 ❌, 더 오래 학습 ❌
- DB 유도 조각 병합이 작동함 (기하 62.9% → DB 64.2%). 다만 구 검출기 캐시 기준

## 이번 세션에 할 일 — Colab 1회 세션 (약 40분)

내가 노트북 notebooks/train_det.ipynb 의 셀 1 → 2 → 2-T 를 실행하고
터널 주소를 줄게. 그 뒤는 SSH 로 네가 직접 진행해.

A. 추론 해상도 '확대' 측정 (2880, 3840 / limit_type=min)
   - 원본보다 큰 해상도는 한 번도 시험한 적이 없다
B. fine-tuned 검출기로 val + test 캐시 재생성
C. 그 캐시로 조건부 인식률 재측정 + db_merge 의 margin 을 val 에서 확정

B 의 캐시 하나가 열린 스레드 13·14·15 를 동시에 푼다.

## 반드시 지킬 것

- 67.5% 와 64.2% 를 곱하거나 비교하지 마라. 축이 다르다
  (전자는 검출만 재고 낡은 85.6%를 곱한 추정, 후자는 구 검출기의 전 구간 실측)
- 하이퍼파라미터는 val 에서 고르고 test 로는 확인만 한다.
  이 프로젝트에서 이미 두 번 어겨서 수치를 폐기했다
- 평가 대상을 넓힐 때 학습 데이터가 섞이는지 먼저 확인해라
  (detect_baseline 의 extra_curved 가 그 사례)
- 8↔B, 5↔S, 0↔O, 1↔I, 2↔Z 는 절대 병합하지 않는다. 둘 다 실재하는 부품코드다

## 알려진 함정 (다시 밟으면 시간 낭비)

- pkill -f tools/train.py 는 SSH 래퍼의 명령줄에도 매칭돼 자기 세션을 죽인다.
  pkill -f "^python3\? tools/train" 처럼 좁혀 써라
- 원격에서 test_every_crop_file_exists 1건 실패는 정상이다
  (검출 작업에 크롭이 불필요해 재생성을 건너뛴다)
- tools/export_model.py 는 paddle 3.3 에서 깨진다. dygraph 로 우회 중이다
- Colab 런타임·터널 모두 수시로 죽는다. 산출물은 Drive 경로로 저장해라

## 접속 절차

curl -fsSL -o scripts/cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
sh scripts/colab_push_code.sh <터널주소>
sh scripts/colab_ssh.sh <터널주소> 'nvidia-smi'

키는 ~/.ssh/colab_label_ocr 에 있고 공개키는 셀 2-T 에 박혀 있다.

먼저 계획을 말해주고, 내가 터널 주소를 주면 시작해.
```

---

## 이 프롬프트가 담고 있는 것

새 세션은 대화 맥락이 없으므로, **다시 유도하면 비싼 것**만 골라 넣었다.

| 항목 | 없으면 생기는 일 |
|---|---|
| 두 수치를 곱하지 말라 | 67.5% × 64.2% 같은 무의미한 계산, 또는 개선 착시 |
| val 에서 고르라 | 이미 두 번 발생한 누수를 세 번째 반복 |
| `pkill` 함정 | SSH 세션이 죽고 원인 파악에 왕복 소모 |
| 크롭 테스트 실패는 정상 | 정상 상태를 버그로 오인해 불필요한 수정 |
| 혼동쌍 미병합 | **불변 규칙 위반** — 프로젝트 전체가 무효가 된다 |

나머지는 `docs/work_plan.md` 와 `reports/*.md` 가 담고 있으므로 프롬프트에 넣지 않는다.
프롬프트가 길어지면 정작 중요한 제약이 묻힌다.
