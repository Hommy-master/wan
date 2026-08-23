FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    APP_ROOT=/app \
    DOWNLOAD_URL=http://127.0.0.1/ \
    CKPT_DIR=/app/Wan2.2-TI2V-5B \
    OUTPUT_DIR=/app/output \
    MODEL_REPO=Wan-AI/Wan2.2-TI2V-5B \
    MODEL_SOURCE=huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/app/Wan2.2-TI2V-5B/.hf_home \
    MODELSCOPE_CACHE=/app/Wan2.2-TI2V-5B/.modelscope \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
COPY docker/requirements.txt /tmp/docker-requirements.txt

# flash_attn needs nvcc (devel image). Strip CRLF from Windows checkouts, then skip it;
# Wan attention already falls back to SDPA.
RUN sed -i 's/\r$//' /tmp/requirements.txt /tmp/docker-requirements.txt \
    && grep -vE '^flash_attn' /tmp/requirements.txt > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/docker-requirements.txt \
    && rm -rf /tmp/requirements.txt /tmp/requirements.docker.txt /tmp/docker-requirements.txt /root/.cache/pip

COPY . /app

RUN mkdir -p /app/output /app/Wan2.2-TI2V-5B

EXPOSE 8000

# docker run --gpus all -p 8000:8000 \
#   -e DOWNLOAD_URL=http://127.0.0.1/ \
#   -v $PWD/Wan2.2-TI2V-5B:/app/Wan2.2-TI2V-5B \
#   -v $PWD/output:/app/output \
#   IMAGE
CMD ["python", "-u", "/app/server.py"]
