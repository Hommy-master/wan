# Copyright 2024-2025. Wan2.2 ComfyUI REST API gateway.
"""FastAPI gateway over ComfyUI for Wan2.2-TI2V-5B: t2v / i2v / flf2v."""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

APP_ROOT = os.environ.get("APP_ROOT", "/app")
DOWNLOAD_URL = os.environ.get("DOWNLOAD_URL", "http://127.0.0.1/")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(APP_ROOT, "output"))
WORKFLOW_DIR = os.environ.get(
    "WORKFLOW_DIR", os.path.join(APP_ROOT, "docker", "workflows")
)
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_BASE = os.environ.get("COMFY_BASE", f"http://{COMFY_HOST}:{COMFY_PORT}")
COMFY_OUTPUT_DIR = os.environ.get(
    "COMFY_OUTPUT_DIR", os.path.join(APP_ROOT, "ComfyUI", "output")
)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_SIZE = "1280*704"
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
SUPPORTED_SIZES = ("1280*704", "704*1280")
POLL_INTERVAL = float(os.environ.get("COMFY_POLL_INTERVAL", "1.5"))
POLL_TIMEOUT = float(os.environ.get("COMFY_POLL_TIMEOUT", "7200"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger("wan.comfy.api")
_gen_lock = threading.Lock()
_client_id = str(uuid.uuid4())


def path_to_url(abs_path: str) -> str:
    download_url = DOWNLOAD_URL if DOWNLOAD_URL.endswith("/") else DOWNLOAD_URL + "/"
    normalized = abs_path.replace("\\", "/")
    if normalized.startswith("/app/"):
        return download_url + normalized[len("/app/") :]
    app_root = os.path.abspath(APP_ROOT).replace("\\", "/")
    if not app_root.endswith("/"):
        app_root += "/"
    if normalized.startswith(app_root):
        return download_url + normalized[len(app_root) :]
    return download_url + os.path.basename(normalized)


def parse_size(size: str) -> tuple[int, int]:
    if size not in SUPPORTED_SIZES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported size {size}, supported: {', '.join(SUPPORTED_SIZES)}",
        )
    width_s, height_s = size.split("*")
    return int(width_s), int(height_s)


def validate_frame_num(frame_num: int) -> int:
    if frame_num < 1 or (frame_num - 1) % 4 != 0:
        raise HTTPException(status_code=400, detail="frame_num must be 4n+1")
    return frame_num


def validate_http_url(value: str, field: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"{field} must be an http or https URL")
    return value


def load_workflow(name: str) -> dict:
    path = Path(WORKFLOW_DIR) / name
    if not path.is_file():
        raise HTTPException(status_code=500, detail=f"workflow not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def wait_comfy_ready(timeout: float = 600.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{COMFY_BASE}/system_stats")
                if r.status_code == 200:
                    logger.info("ComfyUI is ready at %s", COMFY_BASE)
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"ComfyUI not ready within {timeout}s: {COMFY_BASE}")


def download_bytes(url: str) -> bytes:
    validate_http_url(url, "image_url")
    max_bytes = int(os.environ.get("MAX_IMAGE_BYTES", str(30 * 1024 * 1024)))
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(status_code=400, detail="image is larger than allowed")
                    chunks.append(chunk)
        return b"".join(chunks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"failed to download image: {exc}") from exc


def upload_image(data: bytes, filename: str) -> str:
    files = {"image": (filename, data, "application/octet-stream")}
    form = {"overwrite": "true"}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{COMFY_BASE}/upload/image", files=files, data=form)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"ComfyUI upload failed: {r.text}")
        payload = r.json()
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=502, detail=f"unexpected upload response: {payload}")
    subfolder = payload.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name


def queue_prompt(workflow: dict) -> str:
    body = {"prompt": workflow, "client_id": _client_id}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{COMFY_BASE}/prompt", json=body)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"ComfyUI prompt failed: {r.text}")
        payload = r.json()
    if "error" in payload:
        raise HTTPException(status_code=502, detail=str(payload["error"]))
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise HTTPException(status_code=502, detail=f"no prompt_id in response: {payload}")
    return prompt_id


def _collect_videos(outputs: dict[str, Any]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for node_out in outputs.values():
        if not isinstance(node_out, dict):
            continue
        for key in ("videos", "gifs", "images"):
            for item in node_out.get(key) or []:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename")
                if not filename:
                    continue
                ext = Path(filename).suffix.lower()
                if key != "images" or ext in {".mp4", ".webm", ".mov", ".mkv", ".gif"}:
                    found.append((filename, item.get("subfolder") or ""))
    return found


def wait_prompt(prompt_id: str) -> list[tuple[str, str]]:
    deadline = time.time() + POLL_TIMEOUT
    with httpx.Client(timeout=30.0) as client:
        while time.time() < deadline:
            r = client.get(f"{COMFY_BASE}/history/{prompt_id}")
            r.raise_for_status()
            history = r.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status") or {}
                if status.get("status_str") == "error" or status.get("completed") is False:
                    messages = status.get("messages") or entry.get("messages") or []
                    raise HTTPException(
                        status_code=500,
                        detail=f"ComfyUI job failed: {messages or status}",
                    )
                outputs = entry.get("outputs") or {}
                videos = _collect_videos(outputs)
                if videos:
                    return videos
                # completed but no video yet — still treat as failure after status complete
                if status.get("completed"):
                    raise HTTPException(
                        status_code=500,
                        detail=f"ComfyUI finished without video output: {outputs}",
                    )
            time.sleep(POLL_INTERVAL)
    raise HTTPException(status_code=504, detail=f"ComfyUI job timed out: {prompt_id}")


def publish_video(filename: str, subfolder: str, task_name: str) -> dict:
    src = Path(COMFY_OUTPUT_DIR) / subfolder / filename if subfolder else Path(COMFY_OUTPUT_DIR) / filename
    if not src.is_file():
        # SaveVideo may nest under filename_prefix folders
        matches = list(Path(COMFY_OUTPUT_DIR).rglob(filename))
        if not matches:
            raise HTTPException(status_code=500, detail=f"output video not found: {src}")
        src = matches[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{task_name}_{stamp}_{uuid.uuid4().hex[:8]}{src.suffix or '.mp4'}"
    dest = Path(OUTPUT_DIR) / out_name
    shutil.copy2(src, dest)
    container_path = f"/app/output/{out_name}"
    return {
        "task": task_name,
        "video_path": container_path,
        "video_url": path_to_url(container_path),
        "seed": None,
    }


def apply_common(
    workflow: dict,
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frame_num: int,
    seed: int,
    sample_steps: int,
    guide_scale: float,
    sample_shift: float,
    filename_prefix: str,
) -> dict:
    wf = copy.deepcopy(workflow)
    wf["6"]["inputs"]["text"] = prompt
    wf["7"]["inputs"]["text"] = negative_prompt or DEFAULT_NEGATIVE
    wf["3"]["inputs"]["seed"] = seed
    wf["3"]["inputs"]["steps"] = sample_steps
    wf["3"]["inputs"]["cfg"] = guide_scale
    wf["48"]["inputs"]["shift"] = sample_shift
    wf["55"]["inputs"]["width"] = width
    wf["55"]["inputs"]["height"] = height
    wf["55"]["inputs"]["length"] = frame_num
    wf["58"]["inputs"]["filename_prefix"] = filename_prefix
    return wf


def run_job(task_name: str, workflow: dict, seed: int) -> dict:
    with _gen_lock:
        logger.info("queue %s seed=%s", task_name, seed)
        prompt_id = queue_prompt(workflow)
        logger.info("%s prompt_id=%s", task_name, prompt_id)
        videos = wait_prompt(prompt_id)
        filename, subfolder = videos[0]
        result = publish_video(filename, subfolder, task_name)
        result["seed"] = seed
        logger.info("%s done -> %s", task_name, result["video_path"])
        return result


def resolve_seed(seed: int) -> int:
    if seed is not None and seed >= 0:
        return seed
    return int.from_bytes(os.urandom(8), "little") % (2**31)


class GenerateBase(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: Optional[str] = Field(default="", description="Negative prompt")
    size: str = Field(default=DEFAULT_SIZE, description="1280*704 or 704*1280")
    frame_num: int = Field(default=121, description="Frame count, must be 4n+1")
    seed: int = Field(default=-1, description="Random seed, -1 for random")
    sample_steps: int = Field(default=20, description="Sampling steps")
    sample_shift: float = Field(default=8.0, description="ModelSamplingSD3 shift")
    guide_scale: float = Field(default=5.0, description="CFG scale")


class T2VRequest(GenerateBase):
    pass


class I2VRequest(GenerateBase):
    image_url: str = Field(..., description="Source image URL (http/https)")

    @field_validator("image_url")
    @classmethod
    def _image_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("image_url must be an http or https URL")
        return value


class FLF2VRequest(GenerateBase):
    start_image_url: str = Field(..., description="First frame image URL (http/https)")
    end_image_url: str = Field(..., description="Last frame image URL (http/https)")

    @field_validator("start_image_url", "end_image_url")
    @classmethod
    def _urls(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("image URL must be http or https")
        return value


app = FastAPI(
    title="Wan2.2 TI2V-5B ComfyUI API",
    description="REST API for text-to-video, image-to-video, and first-last-frame-to-video",
    version="2.0.0",
)


@app.get("/")
def root():
    return {
        "model": "Wan2.2-TI2V-5B",
        "engine": "ComfyUI",
        "endpoints": {
            "GET /health": "service health",
            "POST /t2v": "text to video",
            "POST /i2v": "image URL to video",
            "POST /flf2v": "first/last frame URLs to video",
        },
    }


@app.get("/health")
def health():
    ready = False
    try:
        with httpx.Client(timeout=3.0) as client:
            ready = client.get(f"{COMFY_BASE}/system_stats").status_code == 200
    except Exception:
        ready = False
    return {
        "status": "ok" if ready else "starting",
        "model": "Wan2.2-TI2V-5B",
        "engine": "ComfyUI",
        "ready": ready,
    }


@app.post("/t2v")
def t2v(req: T2VRequest):
    try:
        width, height = parse_size(req.size)
        frame_num = validate_frame_num(req.frame_num)
        seed = resolve_seed(req.seed)
        workflow = apply_common(
            load_workflow("t2v_api.json"),
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            width=width,
            height=height,
            frame_num=frame_num,
            seed=seed,
            sample_steps=req.sample_steps,
            guide_scale=req.guide_scale,
            sample_shift=req.sample_shift,
            filename_prefix=f"api/t2v_{uuid.uuid4().hex[:8]}",
        )
        return run_job("t2v", workflow, seed)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("t2v failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/i2v")
def i2v(req: I2VRequest):
    try:
        width, height = parse_size(req.size)
        frame_num = validate_frame_num(req.frame_num)
        seed = resolve_seed(req.seed)
        image_name = upload_image(download_bytes(req.image_url), f"i2v_{uuid.uuid4().hex[:8]}.png")
        workflow = apply_common(
            load_workflow("i2v_api.json"),
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            width=width,
            height=height,
            frame_num=frame_num,
            seed=seed,
            sample_steps=req.sample_steps,
            guide_scale=req.guide_scale,
            sample_shift=req.sample_shift,
            filename_prefix=f"api/i2v_{uuid.uuid4().hex[:8]}",
        )
        workflow["56"]["inputs"]["image"] = image_name
        return run_job("i2v", workflow, seed)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("i2v failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/flf2v")
def flf2v(req: FLF2VRequest):
    try:
        width, height = parse_size(req.size)
        frame_num = validate_frame_num(req.frame_num)
        seed = resolve_seed(req.seed)
        start_name = upload_image(
            download_bytes(req.start_image_url), f"flf_start_{uuid.uuid4().hex[:8]}.png"
        )
        end_name = upload_image(
            download_bytes(req.end_image_url), f"flf_end_{uuid.uuid4().hex[:8]}.png"
        )
        workflow = apply_common(
            load_workflow("flf2v_api.json"),
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or "",
            width=width,
            height=height,
            frame_num=frame_num,
            seed=seed,
            sample_steps=req.sample_steps,
            guide_scale=req.guide_scale,
            sample_shift=req.sample_shift,
            filename_prefix=f"api/flf2v_{uuid.uuid4().hex[:8]}",
        )
        workflow["56"]["inputs"]["image"] = start_name
        workflow["60"]["inputs"]["image"] = end_name
        return run_job("flf2v", workflow, seed)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("flf2v failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wait_comfy_ready()
    import uvicorn

    logger.info("API listening on %s:%s  DOWNLOAD_URL=%s", HOST, PORT, DOWNLOAD_URL)
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=3600, log_level="info")


if __name__ == "__main__":
    main()
