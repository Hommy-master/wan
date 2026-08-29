#!/usr/bin/env python3
"""Download Wan2.2 ComfyUI-repackaged weights with resume support."""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger("wan.download")

COMFY_ROOT = Path(os.environ.get("COMFYUI_DIR", "/app/ComfyUI"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", str(COMFY_ROOT / "models")))

FILES = [
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/vae/wan2.2_vae.safetensors",
        "vae/wan2.2_vae.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
]

MIN_BYTES = {
    "diffusion_models/wan2.2_ti2v_5B_fp16.safetensors": 5 * 1024**3,
    "vae/wan2.2_vae.safetensors": 100 * 1024**2,
    "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors": 1 * 1024**3,
}


def _sanitize_env() -> None:
    for key in ("HF_ENDPOINT", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value is not None and not value.strip():
            os.environ.pop(key, None)


def is_complete(path: Path, min_size: int) -> bool:
    return path.is_file() and path.stat().st_size >= min_size


def download_file(repo_id: str, remote_path: str, local_path: Path) -> None:
    from huggingface_hub import hf_hub_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("download %s:%s -> %s", repo_id, remote_path, local_path)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        token=token,
    )
    tmp = local_path.with_suffix(local_path.suffix + ".partial")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(cached, tmp)
    os.replace(tmp, local_path)
    logger.info("saved %s (%s bytes)", local_path, local_path.stat().st_size)


def main() -> None:
    _sanitize_env()
    for repo_id, remote_path, rel in FILES:
        dest = MODELS_DIR / rel
        min_size = MIN_BYTES.get(rel, 1)
        if is_complete(dest, min_size):
            logger.info("already present: %s (%s bytes)", dest, dest.stat().st_size)
            continue
        download_file(repo_id, remote_path, dest)
        if not is_complete(dest, min_size):
            raise SystemExit(f"downloaded file still incomplete: {dest}")

    logger.info("all ComfyUI Wan2.2 models ready under %s", MODELS_DIR)


if __name__ == "__main__":
    main()
