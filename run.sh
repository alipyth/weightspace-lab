#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required (3.10–3.12)." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip

if ! python -c 'import torch' >/dev/null 2>&1; then
  if [ "$(uname -s)" = "Darwin" ]; then
    python -m pip install torch
  else
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
fi

python -m pip install --prefer-binary -r requirements.txt
exec python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8765
