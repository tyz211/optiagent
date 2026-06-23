#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
REQUESTED_PORT="$PORT"

cd "$PROJECT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "未找到 python3，请先安装 Python 3。"
  exit 1
fi

PYTHON_BIN="python3"

if [[ -x ".venv/bin/python" ]]; then
  if .venv/bin/python - <<'PY' >/dev/null 2>&1
import fastapi
import uvicorn
PY
  then
    PYTHON_BIN=".venv/bin/python"
  else
    echo "检测到 .venv，但其中缺少依赖，将尝试使用系统 Python。"
  fi
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import fastapi
import uvicorn
PY
then
  echo "当前 Python 环境缺少依赖。请先运行以下命令安装："
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi

PORT="$("$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
start_port = int(sys.argv[2])

def is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True

for port in range(start_port, start_port + 50):
    if is_available(port):
        print(port)
        break
else:
    raise SystemExit(f"未找到可用端口：{start_port}-{start_port + 49}")
PY
)"

echo "OptiAgent 启动中..."
echo "项目目录：$PROJECT_DIR"
echo "Python：$PYTHON_BIN"
if [[ "$PORT" != "$REQUESTED_PORT" ]]; then
  echo "端口 ${REQUESTED_PORT} 已被占用，已自动切换到 ${PORT}。"
fi
echo "访问地址：http://${HOST}:${PORT}"
echo "按 Ctrl+C 停止服务。"

exec "$PYTHON_BIN" -m uvicorn api.main:app --host "$HOST" --port "$PORT" --reload
