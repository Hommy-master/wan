FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_ROOT=/app \
    COMFYUI_DIR=/app/ComfyUI \
    WORKFLOW_DIR=/app/docker/workflows \
    OUTPUT_DIR=/app/output \
    COMFY_OUTPUT_DIR=/app/ComfyUI/output \
    MODELS_DIR=/app/ComfyUI/models \
    DOWNLOAD_URL=http://127.0.0.1/ \
    COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    PORT=8000 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/app/ComfyUI/models/.hf_home \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI \
    && pip install --no-cache-dir -r /app/ComfyUI/requirements.txt

# First/Last frame node for Wan2.2 5B VAE
RUN git clone --depth 1 -b master \
        https://github.com/stduhpf/ComfyUI--Wan22FirstLastFrameToVideoLatent.git \
        /app/ComfyUI/custom_nodes/ComfyUI--Wan22FirstLastFrameToVideoLatent

COPY docker/api-requirements.txt /tmp/api-requirements.txt
RUN sed -i 's/\r$//' /tmp/api-requirements.txt \
    && pip install --no-cache-dir -r /tmp/api-requirements.txt \
    && rm -rf /tmp/api-requirements.txt /root/.cache/pip

COPY docker/workflows /app/docker/workflows
COPY docker/download_models.py /app/docker/download_models.py
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
COPY server.py /app/server.py

RUN sed -i 's/\r$//' /app/docker/entrypoint.sh /app/docker/download_models.py \
    && chmod +x /app/docker/entrypoint.sh \
    && mkdir -p \
        /app/output \
        /app/ComfyUI/models/diffusion_models \
        /app/ComfyUI/models/text_encoders \
        /app/ComfyUI/models/vae \
        /app/ComfyUI/input \
        /app/ComfyUI/output

EXPOSE 8000

CMD ["/app/docker/entrypoint.sh"]
