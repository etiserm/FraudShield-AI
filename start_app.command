#!/bin/zsh
set -e

cd "$(dirname "$0")"

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "Python 3.13 is required. Install Python 3.13, then open this file again."
  read -k 1 "?Press any key to close this window..."
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  python3.13 -m venv .venv
fi

if ! .venv/bin/python -c "import fastapi, joblib, multipart, numpy, pandas, sklearn, streamlit, uvicorn" >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/uvicorn api:app --host 127.0.0.1 --port 8080
