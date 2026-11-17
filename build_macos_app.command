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

if ! .venv/bin/python -c "import fastapi, joblib, multipart, numpy, pandas, sklearn, uvicorn, webview, PyInstaller" >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements-macos.txt
fi

.venv/bin/python -m PyInstaller --noconfirm --clean "FraudShield AI.spec"

ditto -c -k --sequesterRsrc --keepParent \
  "dist/FraudShield AI.app" \
  "dist/FraudShield AI-macOS-arm64.zip"

echo "Built dist/FraudShield AI.app"
echo "Built dist/FraudShield AI-macOS-arm64.zip"
echo "Unzip it, then open FraudShield AI.app."
