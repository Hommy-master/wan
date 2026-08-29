#!/bin/bash
set -euo pipefail

APP_ROOT="${APP_ROOT:-/app}"
COMFYUI_DIR="${COMFYUI_DIR:-${APP_ROOT}/ComfyUI}"
OUTPUT_DIR="${OUTPUT_DIR:-${APP_ROOT}/output}"
COMFY_PORT="${COMFY_PORT:-8188}"
API_PORT="${PORT:-8000}"

mkdir -p "${OUTPUT_DIR}" \
  "${COMFYUI_DIR}/models/diffusion_models" \
  "${COMFYUI_DIR}/models/text_encoders" \
  "${COMFYUI_DIR}/models/vae" \
  "${COMFYUI_DIR}/input" \
  "${COMFYUI_DIR}/output" \
  "${COMFYUI_DIR}/custom_nodes"

echo "[entrypoint] ensure models..."
python -u "${APP_ROOT}/docker/download_models.py"

echo "[entrypoint] start ComfyUI on :${COMFY_PORT}"
cd "${COMFYUI_DIR}"
python -u main.py \
  --listen 127.0.0.1 \
  --port "${COMFY_PORT}" \
  --disable-auto-launch \
  --force-fp16 \
  >/tmp/comfyui.log 2>&1 &
COMFY_PID=$!

cleanup() {
  echo "[entrypoint] shutting down..."
  kill "${COMFY_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Wait until ComfyUI answers
for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
    echo "[entrypoint] ComfyUI ready"
    break
  fi
  if ! kill -0 "${COMFY_PID}" 2>/dev/null; then
    echo "[entrypoint] ComfyUI exited early; log:"
    tail -n 200 /tmp/comfyui.log || true
    exit 1
  fi
  sleep 2
done

if ! curl -fsS "http://127.0.0.1:${COMFY_PORT}/system_stats" >/dev/null 2>&1; then
  echo "[entrypoint] ComfyUI failed to become ready; log:"
  tail -n 200 /tmp/comfyui.log || true
  exit 1
fi

echo "[entrypoint] start REST API on :${API_PORT}"
cd "${APP_ROOT}"
exec python -u "${APP_ROOT}/server.py"
