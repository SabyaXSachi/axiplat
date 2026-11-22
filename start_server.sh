#!/usr/bin/env bash
# AXIPLAT Server Launcher (macOS/Linux)

cd "$(dirname "$0")" || exit 1

echo "========================================"
echo "AXIPLAT Server Launcher"
echo "========================================"
echo

if [ -x "venv/bin/python" ]; then
    echo "[OK] Using project virtual environment"
    ./venv/bin/python app.py
else
    echo "[INFO] No venv folder found - using system python3."
    echo "       Create one with: python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    echo
    python3 app.py
fi
