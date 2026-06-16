#!/usr/bin/env bash
# One-shot launcher: creates the venv on first run, installs deps, starts server.
set -euo pipefail
cd "$(dirname "$0")"

PY=python3.12
VENV=.venv

if [ ! -d "$VENV" ]; then
  echo "→ Creating virtual environment ($PY)…"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip >/dev/null
  echo "→ Installing dependencies (first run pulls MLX + Kokoro tooling)…"
  "$VENV/bin/pip" install -r requirements.txt
fi

echo "→ Starting Narra on http://127.0.0.1:8000"
# Open the browser once the server is up (skip with: NO_OPEN=1 narra)
if [ -z "${NO_OPEN:-}" ]; then
  ( for _ in $(seq 1 40); do
      curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1 && { open http://127.0.0.1:8000; break; }
      sleep 0.5
    done ) &
fi
exec "$VENV/bin/uvicorn" backend.server:app --host 127.0.0.1 --port 8000
