#!/bin/bash
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
if ! .venv/bin/python demo/server.py --open "$@"; then
    echo '启动失败。请检查上方提示，按回车键关闭。'
    read -r
    exit 1
fi
