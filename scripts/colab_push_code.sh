#!/bin/sh
# 로컬 작업본을 Colab 으로 밀어넣는다.
#
#   sh scripts/colab_push_code.sh <터널주소>
#
# 저장소가 private 이라 Colab 에서 clone 하려면 PAT 가 필요하다. 그런데 지금 이 PC 의
# 작업본이 곧 진실이므로, PAT 없이 tar 로 직접 보내는 편이 빠르고 확실하다.
# 이미지(image_set)와 크롭은 보내지 않는다 — 이미지는 Drive 아카이브에서 풀고,
# 크롭은 검출 작업에 필요 없다(인식 학습 때만 `build_labels` 로 재생성).

set -e
HOST="$1"
[ -n "$HOST" ] || { echo "사용법: sh scripts/colab_push_code.sh <터널주소>" >&2; exit 2; }

DIR=$(dirname "$0")
CF="$DIR/cloudflared.exe"
[ -f "$CF" ] || CF="cloudflared"

cd "$DIR/.."
tar -czf - --exclude='__pycache__' \
    src tests labels splits snapshots configs conftest.py \
  | ssh -i "$HOME/.ssh/colab_label_ocr" \
        -o ProxyCommand="$CF access ssh --hostname %h" \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o BatchMode=yes -o ConnectTimeout=40 -o LogLevel=ERROR \
        root@"$HOST" 'mkdir -p /content/label_ocr && tar -xzf - -C /content/label_ocr && echo "코드 전송 완료"'
