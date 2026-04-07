#!/usr/bin/env bash
set -euo pipefail

cd /root/wdkns-skills-web

export APP_SECRET_KEY="${APP_SECRET_KEY:-replace-this-in-production}"
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-8765}"
export APP_DEBUG="${APP_DEBUG:-false}"
export APP_MAX_UPLOAD_BYTES="${APP_MAX_UPLOAD_BYTES:-1073741824}"
export GROQ_API_KEY="${GROQ_API_KEY:-}"
export GROQ_TRANSCRIBE_MODEL="${GROQ_TRANSCRIBE_MODEL:-whisper-large-v3-turbo}"

exec /usr/bin/env gunicorn \
  --workers 1 \
  --threads 8 \
  --bind "${APP_HOST}:${APP_PORT}" \
  --timeout 3600 \
  --graceful-timeout 60 \
  --worker-tmp-dir /dev/shm \
  run:app
