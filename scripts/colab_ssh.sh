#!/bin/sh
# Colab 원격 셸 래퍼.
#
#   sh scripts/colab_ssh.sh <터널주소> '<원격에서 실행할 명령>'
#   sh scripts/colab_ssh.sh <터널주소> 'nvidia-smi'
#
# 사전 준비 (1회씩):
#   1) 키 — 이미 있으면 건너뛴다. 공개키는 노트북 셀 2-T 에 박혀 있다.
#        ssh-keygen -t ed25519 -N '' -C colab@label_ocr -f ~/.ssh/colab_label_ocr
#   2) cloudflared 클라이언트 (Windows):
#        curl -fsSL -o scripts/cloudflared.exe \
#          https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
#      (.gitignore 로 제외돼 있다. 세션마다 다시 받아도 몇 초면 된다.)
#
# 터널 주소는 노트북 셀 2-T 를 실행하면 출력된다. 매번 바뀐다.
#
# ## 왜 환경변수를 명령 앞에 붙이는가
#
# sshd 세션은 Colab 커널의 환경을 상속하지 않는다. LD_LIBRARY_PATH 가 비어 있어
# libnvidia-ml.so 를 못 찾고 GPU 가 아예 안 보인다. Ubuntu 의 ~/.bashrc 는
# 비대화형 셸에서 즉시 return 하므로 거기 넣어도 먹지 않는다. 그래서 직접 붙인다.

set -e
HOST="$1"
CMD="$2"
[ -n "$HOST" ] || { echo "사용법: sh scripts/colab_ssh.sh <터널주소> '<명령>'" >&2; exit 2; }

DIR=$(dirname "$0")
CF="$DIR/cloudflared.exe"
[ -f "$CF" ] || CF="cloudflared"        # PATH 에 있으면 그것을 쓴다

ENV='export LD_LIBRARY_PATH=/usr/lib64-nvidia
export PATH=/opt/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/tools/node/bin:/tools/google-cloud-sdk/bin
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
'

exec ssh -i "$HOME/.ssh/colab_label_ocr" \
    -o ProxyCommand="$CF access ssh --hostname %h" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o BatchMode=yes -o ConnectTimeout=40 -o LogLevel=ERROR \
    -o ServerAliveInterval=30 \
    root@"$HOST" "$ENV$CMD"
