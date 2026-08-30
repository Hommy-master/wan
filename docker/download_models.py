#!/usr/bin/env python3
"""Download Wan2.2 ComfyUI weights with HTTP resume (mirror-friendly)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from urllib.parse import urljoin

import httpx

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


def endpoints() -> list[str]:
    primary = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    endpoints_list = [primary]
    for candidate in ("https://hf-mirror.com", "https://huggingface.co"):
        if candidate not in endpoints_list:
            endpoints_list.append(candidate)
    return endpoints_list


def is_complete(path: Path, min_size: int) -> bool:
    return path.is_file() and path.stat().st_size >= min_size


def build_url(base: str, repo_id: str, remote_path: str) -> str:
    return f"{base}/{repo_id}/resolve/main/{remote_path}"


def download_file(repo_id: str, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    partial = local_path.with_suffix(local_path.suffix + ".partial")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    errors: list[str] = []
    for base in endpoints():
        url = build_url(base, repo_id, remote_path)
        try:
            logger.info("download %s", url)
            existing = partial.stat().st_size if partial.is_file() else 0
            req_headers = dict(headers)
            if existing > 0:
                req_headers["Range"] = f"bytes={existing}-"
                logger.info("resume from byte %s", existing)

            with httpx.Client(timeout=httpx.Timeout(60.0, read=600.0), follow_redirects=True) as client:
                with client.stream("GET", url, headers=req_headers) as response:
                    if response.status_code == 416:
                        # already complete according to server
                        if partial.is_file():
                            partial.replace(local_path)
                        return
                    if response.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {response.status_code}")
                    mode = "ab" if response.status_code == 206 and existing > 0 else "wb"
                    if mode == "wb" and partial.exists():
                        partial.unlink()
                    with partial.open(mode) as f:
                        for chunk in response.iter_bytes(1024 * 1024):
                            f.write(chunk)
            partial.replace(local_path)
            logger.info("saved %s (%s bytes)", local_path, local_path.stat().st_size)
            return
        except Exception as exc:
            logger.warning("download via %s failed: %s", base, exc)
            errors.append(f"{base}: {exc}")
            continue
    raise RuntimeError(f"all mirrors failed for {remote_path}: {'; '.join(errors)}")


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
