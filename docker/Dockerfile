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

# Avoid flaky apt mirrors: install system tools via conda (bundled in base image)
RUN conda install -y -c conda-forge git ffmpeg curl \
    && conda clean -afy

# ComfyUI (+ China-friendly pip index as fallback)
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI \
    && (pip install --no-cache-dir -r /app/ComfyUI/requirements.txt \
        || pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/ComfyUI/requirements.txt) \
    && python -c "import pathlib,site; print('comfy_kitchen check'); import importlib.util; print(importlib.util.find_spec('comfy_kitchen'))"

# First/Last frame node for Wan2.2 5B VAE
RUN git clone --depth 1 -b master \
        https://github.com/stduhpf/ComfyUI--Wan22FirstLastFrameToVideoLatent.git \
        /app/ComfyUI/custom_nodes/ComfyUI--Wan22FirstLastFrameToVideoLatent

COPY docker/api-requirements.txt /tmp/api-requirements.txt
RUN sed -i 's/\r$//' /tmp/api-requirements.txt \
    && (pip install --no-cache-dir -r /tmp/api-requirements.txt \
        || pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/api-requirements.txt) \
    && rm -rf /tmp/api-requirements.txt /root/.cache/pip

COPY docker/workflows /app/docker/workflows
COPY docker/download_models.py /app/docker/download_models.py
COPY docker/patch_comfy_kitchen.py /app/docker/patch_comfy_kitchen.py
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
COPY server.py /app/server.py

RUN sed -i 's/\r$//' /app/docker/entrypoint.sh /app/docker/download_models.py /app/docker/patch_comfy_kitchen.py \
    && chmod +x /app/docker/entrypoint.sh \
    && python /app/docker/patch_comfy_kitchen.py \
    && mkdir -p \
        /app/output \
        /app/ComfyUI/models/diffusion_models \
        /app/ComfyUI/models/text_encoders \
        /app/ComfyUI/models/vae \
        /app/ComfyUI/input \
        /app/ComfyUI/output

EXPOSE 8000

CMD ["/app/docker/entrypoint.sh"]
