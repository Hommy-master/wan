# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Wan2.2-TI2V-5B REST API. Downloads the checkpoint on startup (resumable), then serves t2v / i2v."""
import logging
import os
import sys
import threading
import uuid
import warnings
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

warnings.filterwarnings("ignore")

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video

APP_ROOT = os.environ.get("APP_ROOT", "/app")
CKPT_DIR = os.environ.get("CKPT_DIR", os.path.join(APP_ROOT, "Wan2.2-TI2V-5B"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(APP_ROOT, "output"))
DOWNLOAD_URL = os.environ.get("DOWNLOAD_URL", "http://127.0.0.1/")
MODEL_REPO = os.environ.get("MODEL_REPO", "Wan-AI/Wan2.2-TI2V-5B")
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "huggingface").strip().lower()
TASK = "ti2v-5B"
DEFAULT_SIZE = "1280*704"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

REQUIRED_FILES = (
    "config.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "Wan2.2_VAE.pth",
    "google/umt5-xxl/tokenizer_config.json",
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger("wan.api")

_pipeline = None
_gen_lock = threading.Lock()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def path_to_url(abs_path: str) -> str:
    """Replace the container prefix /app/ with DOWNLOAD_URL."""
    download_url = DOWNLOAD_URL if DOWNLOAD_URL.endswith("/") else DOWNLOAD_URL + "/"
    normalized = abs_path.replace("\\", "/")
    if normalized.startswith("/app/"):
        return download_url + normalized[len("/app/"):]
    app_root = os.path.abspath(APP_ROOT).replace("\\", "/")
    if not app_root.endswith("/"):
        app_root += "/"
    if normalized.startswith(app_root):
        return download_url + normalized[len(app_root):]
    return download_url + os.path.basename(normalized)


def _has_incomplete(root: str) -> bool:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".incomplete") or name.endswith(".lock"):
                return True
    return False


def _has_required_files(root: str) -> bool:
    for rel in REQUIRED_FILES:
        path = os.path.join(root, rel)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            return False
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".safetensors") and os.path.getsize(os.path.join(dirpath, name)) > 0:
                return True
    return False


def _hf_sizes_match(root: str, repo_id: str) -> bool:
    from huggingface_hub import HfApi

    endpoint = os.environ.get("HF_ENDPOINT") or None
    api = HfApi(endpoint=endpoint, token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    for item in api.list_repo_tree(repo_id, recursive=True):
        size = getattr(item, "size", None)
        path = getattr(item, "path", None)
        if not path or size is None:
            continue
        local = os.path.join(root, path)
        if not os.path.isfile(local) or os.path.getsize(local) != size:
            logger.info("checkpoint incomplete: %s", path)
            return False
    return True


def is_model_complete(root: str) -> bool:
    if not os.path.isdir(root):
        return False
    if _has_incomplete(root):
        return False
    if not _has_required_files(root):
        return False
    if MODEL_SOURCE == "huggingface":
        try:
            return _hf_sizes_match(root, MODEL_REPO)
        except Exception as exc:
            logger.warning("skip remote size check (%s), use local files", exc)
            return True
    return True


def download_model(root: str) -> None:
    os.makedirs(root, exist_ok=True)
    if MODEL_SOURCE == "modelscope":
        logger.info("resuming ModelScope download: %s -> %s", MODEL_REPO, root)
        from modelscope.hub.snapshot_download import snapshot_download as ms_download

        ms_download(MODEL_REPO, local_dir=root)
        return

    logger.info("resuming Hugging Face download: %s -> %s", MODEL_REPO, root)
    from huggingface_hub import snapshot_download

    kwargs = dict(
        repo_id=MODEL_REPO,
        local_dir=root,
        token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        max_workers=int(os.environ.get("HF_DOWNLOAD_WORKERS", "8")),
    )
    try:
        snapshot_download(resume_download=True, **kwargs)
    except TypeError:
        snapshot_download(**kwargs)


def ensure_model() -> None:
    os.makedirs(root := CKPT_DIR, exist_ok=True)
    if is_model_complete(root):
        logger.info("Wan2.2-TI2V-5B already complete at %s", root)
        return
    logger.info("Wan2.2-TI2V-5B missing or incomplete, start / resume download")
    download_model(root)
    if not _has_required_files(root) or _has_incomplete(root):
        raise RuntimeError(f"model download finished but checkpoint is still incomplete: {root}")
    logger.info("Wan2.2-TI2V-5B download finished")


def load_pipeline():
    global _pipeline
    import wan

    cfg = WAN_CONFIGS[TASK]
    logger.info("loading WanTI2V from %s", CKPT_DIR)
    _pipeline = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=CKPT_DIR,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=_bool_env("T5_CPU", True),
        convert_model_dtype=_bool_env("CONVERT_MODEL_DTYPE", True),
    )
    logger.info("WanTI2V ready")


def _validate_size(size: str) -> str:
    supported = SUPPORTED_SIZES[TASK]
    if size not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported size {size}, supported: {', '.join(supported)}",
        )
    return size


def _validate_frame_num(frame_num: int) -> int:
    if frame_num < 1 or (frame_num - 1) % 4 != 0:
        raise HTTPException(status_code=400, detail="frame_num must be 4n+1")
    return frame_num


def download_image(image_url: str) -> Image.Image:
    parsed = urlparse(image_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail="image_url must be an http or https URL")

    import httpx

    max_bytes = int(os.environ.get("MAX_IMAGE_BYTES", str(30 * 1024 * 1024)))
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            with client.stream("GET", image_url) as response:
                response.raise_for_status()
                chunks = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(status_code=400, detail="image is larger than the allowed size")
                    chunks.append(chunk)
        from io import BytesIO

        image = Image.open(BytesIO(b"".join(chunks))).convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to download image: {exc}") from exc
    return image


def generate_video(prompt: str, image: Optional[Image.Image], req) -> dict:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model is not ready")

    size = _validate_size(req.size)
    frame_num = _validate_frame_num(req.frame_num)
    cfg = WAN_CONFIGS[TASK]
    seed = req.seed if req.seed is not None and req.seed >= 0 else int.from_bytes(os.urandom(8), "little") % (2**31)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    task_name = "i2v" if image is not None else "t2v"
    filename = f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp4"
    save_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))

    with _gen_lock:
        logger.info("generate %s prompt=%s size=%s frames=%s seed=%s", task_name, prompt[:80], size, frame_num, seed)
        video = _pipeline.generate(
            prompt,
            img=image,
            size=SIZE_CONFIGS[size],
            max_area=MAX_AREA_CONFIGS[size],
            frame_num=frame_num,
            shift=req.sample_shift if req.sample_shift is not None else cfg.sample_shift,
            sample_solver=req.sample_solver,
            sampling_steps=req.sample_steps if req.sample_steps is not None else cfg.sample_steps,
            guide_scale=req.guide_scale if req.guide_scale is not None else cfg.sample_guide_scale,
            n_prompt=req.negative_prompt or "",
            seed=seed,
            offload_model=_bool_env("OFFLOAD_MODEL", True),
        )
        save_video(
            tensor=video[None],
            save_file=save_path,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        del video
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not os.path.isfile(save_path):
        raise HTTPException(status_code=500, detail="video file was not written")

    container_path = "/app/output/" + filename
    return {
        "task": task_name,
        "video_path": container_path,
        "video_url": path_to_url(container_path),
        "seed": seed,
    }


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    size: str = Field(default=DEFAULT_SIZE, description="Video size, e.g. 1280*704 or 704*1280")
    frame_num: int = Field(default=121, description="Number of frames, must be 4n+1")
    seed: int = Field(default=-1, description="Random seed, -1 for random")
    sample_steps: Optional[int] = Field(default=None, description="Diffusion sampling steps")
    sample_shift: Optional[float] = Field(default=None, description="Flow-matching shift")
    guide_scale: Optional[float] = Field(default=None, description="CFG scale")
    sample_solver: str = Field(default="unipc", description="unipc or dpm++")

    @field_validator("sample_solver")
    @classmethod
    def _solver(cls, value: str) -> str:
        if value not in ("unipc", "dpm++"):
            raise ValueError("sample_solver must be unipc or dpm++")
        return value


class T2VRequest(GenerateRequest):
    pass


class I2VRequest(GenerateRequest):
    image_url: str = Field(..., description="Source image URL, http or https only")

    @field_validator("image_url")
    @classmethod
    def _image_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("image_url must be an http or https URL")
        return value


app = FastAPI(
    title="Wan2.2 TI2V-5B API",
    description="Text-to-video and image-to-video REST API powered by Wan2.2-TI2V-5B",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "model": "Wan2.2-TI2V-5B",
        "ready": _pipeline is not None,
        "endpoints": {
            "GET /health": "service health",
            "POST /t2v": "text to video",
            "POST /i2v": "image URL to video",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "Wan2.2-TI2V-5B", "ready": _pipeline is not None}


@app.post("/t2v")
def t2v(req: T2VRequest):
    try:
        return generate_video(req.prompt, None, req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("t2v failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/i2v")
def i2v(req: I2VRequest):
    try:
        image = download_image(req.image_url)
        return generate_video(req.prompt, image, req)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("i2v failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available; generation will fail on CPU")
    ensure_model()
    load_pipeline()
    import uvicorn

    logger.info("API listening on %s:%s  DOWNLOAD_URL=%s", HOST, PORT, DOWNLOAD_URL)
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=3600, log_level="info")


if __name__ == "__main__":
    main()
