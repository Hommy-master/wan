# Copyright 2024-2025. Wan2.2 ComfyUI REST API gateway.
"""FastAPI gateway over ComfyUI for Wan2.2-TI2V-5B: async t2v / i2v / flf2v."""
from __future__ import annotations

import copy
import json
import logging
import os
import queue
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
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
COMFY_WS = os.environ.get(
    "COMFY_WS", f"ws://{COMFY_HOST}:{COMFY_PORT}/ws?clientId={{client_id}}"
)
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
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "32"))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger("wan.comfy.api")
_client_id = str(uuid.uuid4())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise RuntimeError(f"workflow not found: {path}")
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
                        raise RuntimeError("image is larger than allowed")
                    chunks.append(chunk)
        return b"".join(chunks)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"failed to download image: {exc}") from exc


def upload_image(data: bytes, filename: str) -> str:
    files = {"image": (filename, data, "application/octet-stream")}
    form = {"overwrite": "true"}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(f"{COMFY_BASE}/upload/image", files=files, data=form)
        if r.status_code >= 400:
            raise RuntimeError(f"ComfyUI upload failed: {r.text}")
        payload = r.json()
    name = payload.get("name")
    if not name:
        raise RuntimeError(f"unexpected upload response: {payload}")
    subfolder = payload.get("subfolder") or ""
    return f"{subfolder}/{name}" if subfolder else name


def queue_prompt(workflow: dict) -> str:
    body = {"prompt": workflow, "client_id": _client_id}
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{COMFY_BASE}/prompt", json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"ComfyUI prompt failed: {r.text}")
        payload = r.json()
    if "error" in payload:
        raise RuntimeError(str(payload["error"]))
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"no prompt_id in response: {payload}")
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


def _start_progress_listener(
    prompt_id: str, on_progress: Callable[[Optional[float], str], None]
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def _run() -> None:
        try:
            import websocket  # type: ignore
        except ImportError:
            logger.warning("websocket-client not installed; step progress unavailable")
            return

        url = COMFY_WS.format(client_id=_client_id)

        def on_message(_ws: Any, message: str) -> None:
            if stop.is_set():
                return
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                return
            msg_type = payload.get("type")
            data = payload.get("data") or {}
            if msg_type == "progress":
                value = float(data.get("value") or 0)
                maximum = float(data.get("max") or 0) or 1.0
                # Map sampler progress into 0.15 .. 0.90
                ratio = max(0.0, min(1.0, value / maximum))
                on_progress(0.15 + ratio * 0.75, f"sampling {int(value)}/{int(maximum)}")
            elif msg_type == "executing":
                node = data.get("node")
                if node is None and data.get("prompt_id") in (None, prompt_id):
                    on_progress(0.92, "finalizing")
                elif node is not None:
                    on_progress(None, f"executing node {node}")

        def on_error(_ws: Any, error: Exception) -> None:
            if not stop.is_set():
                logger.debug("comfy ws error: %s", error)

        while not stop.is_set():
            ws = None
            try:
                ws = websocket.WebSocketApp(
                    url,
                    on_message=on_message,
                    on_error=on_error,
                )
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:
                if not stop.is_set():
                    logger.debug("comfy ws reconnect: %s", exc)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
            if not stop.wait(2.0):
                continue
            break

    thread = threading.Thread(target=_run, name=f"comfy-ws-{prompt_id[:8]}", daemon=True)
    thread.start()
    return stop, thread


def wait_prompt(
    prompt_id: str,
    on_progress: Optional[Callable[[Optional[float], str], None]] = None,
) -> list[tuple[str, str]]:
    stop_ws: Optional[threading.Event] = None
    if on_progress is not None:
        stop_ws, _ = _start_progress_listener(prompt_id, on_progress)

    deadline = time.time() + POLL_TIMEOUT
    try:
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
                        raise RuntimeError(f"ComfyUI job failed: {messages or status}")
                    outputs = entry.get("outputs") or {}
                    videos = _collect_videos(outputs)
                    if videos:
                        return videos
                    if status.get("completed"):
                        raise RuntimeError(
                            f"ComfyUI finished without video output: {outputs}"
                        )
                time.sleep(POLL_INTERVAL)
        raise RuntimeError(f"ComfyUI job timed out: {prompt_id}")
    finally:
        if stop_ws is not None:
            stop_ws.set()


def publish_video(filename: str, subfolder: str, task_name: str) -> dict:
    src = Path(COMFY_OUTPUT_DIR) / subfolder / filename if subfolder else Path(COMFY_OUTPUT_DIR) / filename
    if not src.is_file():
        matches = list(Path(COMFY_OUTPUT_DIR).rglob(filename))
        if not matches:
            raise RuntimeError(f"output video not found: {src}")
        src = matches[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{task_name}_{stamp}_{uuid.uuid4().hex[:8]}{src.suffix or '.mp4'}"
    dest = Path(OUTPUT_DIR) / out_name
    shutil.copy2(src, dest)
    container_path = f"/app/output/{out_name}"
    return {
        "video_path": container_path,
        "video_url": path_to_url(container_path),
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


def resolve_seed(seed: int) -> int:
    if seed is not None and seed >= 0:
        return seed
    return int.from_bytes(os.urandom(8), "little") % (2**31)


# ---------------------------------------------------------------------------
# Async task queue (single worker — at most one job executing)
# ---------------------------------------------------------------------------

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCEEDED = "succeeded"
STATUS_FAILED = "failed"


@dataclass
class TaskJob:
    task_id: str
    task: str
    seed: int
    build_workflow: Callable[[], dict]
    status: str = STATUS_QUEUED
    progress: float = 0.0
    message: str = "queued"
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": self.task,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "seed": self.seed,
            "video_path": self.video_path,
            "video_url": self.video_url,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class TaskManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, TaskJob] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._current_id: Optional[str] = None
        self._worker = threading.Thread(target=self._worker_loop, name="wan-worker", daemon=True)
        self._worker.start()

    def submit(self, job: TaskJob) -> TaskJob:
        with self._lock:
            queued = sum(1 for j in self._jobs.values() if j.status == STATUS_QUEUED)
            if queued >= MAX_QUEUE_SIZE:
                raise HTTPException(
                    status_code=429,
                    detail=f"task queue is full (max {MAX_QUEUE_SIZE})",
                )
            self._jobs[job.task_id] = job
        self._queue.put(job.task_id)
        logger.info("enqueued %s task_id=%s seed=%s", job.task, job.task_id, job.seed)
        return job

    def get(self, task_id: str) -> TaskJob:
        with self._lock:
            job = self._jobs.get(task_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"task not found: {task_id}")
            return job

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            queued = sum(1 for j in self._jobs.values() if j.status == STATUS_QUEUED)
            running = self._current_id
        return {
            "queued": queued,
            "running_task_id": running,
            "max_concurrent": 1,
            "max_queue_size": MAX_QUEUE_SIZE,
        }

    def _update(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        video_path: Optional[str] = None,
        video_url: Optional[str] = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        with self._lock:
            job = self._jobs.get(task_id)
            if job is None:
                return
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = max(job.progress, min(1.0, float(progress)))
            if message is not None:
                job.message = message
            if error is not None:
                job.error = error
            if video_path is not None:
                job.video_path = video_path
            if video_url is not None:
                job.video_url = video_url
            now = utc_now()
            job.updated_at = now
            if started and job.started_at is None:
                job.started_at = now
            if finished:
                job.finished_at = now

    def _worker_loop(self) -> None:
        while True:
            task_id = self._queue.get()
            with self._lock:
                job = self._jobs.get(task_id)
                self._current_id = task_id
            if job is None:
                self._queue.task_done()
                with self._lock:
                    self._current_id = None
                continue
            try:
                self._run_job(job)
            except Exception as exc:
                logger.exception("task %s failed", task_id)
                self._update(
                    task_id,
                    status=STATUS_FAILED,
                    progress=1.0,
                    message="failed",
                    error=str(exc),
                    finished=True,
                )
            finally:
                with self._lock:
                    self._current_id = None
                self._queue.task_done()

    def _run_job(self, job: TaskJob) -> None:
        task_id = job.task_id
        self._update(
            task_id,
            status=STATUS_RUNNING,
            progress=0.02,
            message="preparing",
            started=True,
        )

        def on_progress(progress: Optional[float], message: str) -> None:
            kwargs: dict[str, Any] = {"message": message}
            if progress is not None:
                kwargs["progress"] = progress
            self._update(task_id, **kwargs)

        workflow = job.build_workflow()
        self._update(task_id, progress=0.1, message="queued on ComfyUI")
        prompt_id = queue_prompt(workflow)
        logger.info("%s task_id=%s prompt_id=%s", job.task, task_id, prompt_id)
        self._update(task_id, progress=0.15, message="generating")

        videos = wait_prompt(prompt_id, on_progress=on_progress)
        filename, subfolder = videos[0]
        self._update(task_id, progress=0.95, message="publishing")
        published = publish_video(filename, subfolder, job.task)
        self._update(
            task_id,
            status=STATUS_SUCCEEDED,
            progress=1.0,
            message="succeeded",
            video_path=published["video_path"],
            video_url=published["video_url"],
            finished=True,
        )
        logger.info("%s task_id=%s done -> %s", job.task, task_id, published["video_path"])


task_manager = TaskManager()


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


class SubmitResponse(BaseModel):
    task_id: str
    task: str
    status: str
    message: str = "accepted"


app = FastAPI(
    title="Wan2.2 TI2V-5B ComfyUI API",
    description="Async REST API for text-to-video, image-to-video, and first-last-frame-to-video",
    version="3.0.0",
)


def _enqueue(
    task_name: str,
    seed: int,
    build_workflow: Callable[[], dict],
) -> dict:
    job = TaskJob(
        task_id=str(uuid.uuid4()),
        task=task_name,
        seed=seed,
        build_workflow=build_workflow,
    )
    task_manager.submit(job)
    return {
        "task_id": job.task_id,
        "task": job.task,
        "status": job.status,
        "message": "accepted",
    }


@app.get("/")
def root():
    return {
        "model": "Wan2.2-TI2V-5B",
        "engine": "ComfyUI",
        "mode": "async",
        "endpoints": {
            "GET /health": "service health",
            "POST /t2v": "submit text-to-video (returns task_id)",
            "POST /i2v": "submit image-to-video (returns task_id)",
            "POST /flf2v": "submit first/last-frame-to-video (returns task_id)",
            "GET /tasks/{task_id}": "query task progress and result",
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
        "queue": task_manager.snapshot(),
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    return task_manager.get(task_id).to_dict()


@app.post("/t2v", response_model=SubmitResponse)
def t2v(req: T2VRequest):
    width, height = parse_size(req.size)
    frame_num = validate_frame_num(req.frame_num)
    seed = resolve_seed(req.seed)
    prompt = req.prompt
    negative_prompt = req.negative_prompt or ""
    sample_steps = req.sample_steps
    guide_scale = req.guide_scale
    sample_shift = req.sample_shift

    def build() -> dict:
        return apply_common(
            load_workflow("t2v_api.json"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frame_num=frame_num,
            seed=seed,
            sample_steps=sample_steps,
            guide_scale=guide_scale,
            sample_shift=sample_shift,
            filename_prefix=f"api/t2v_{uuid.uuid4().hex[:8]}",
        )

    return _enqueue("t2v", seed, build)


@app.post("/i2v", response_model=SubmitResponse)
def i2v(req: I2VRequest):
    width, height = parse_size(req.size)
    frame_num = validate_frame_num(req.frame_num)
    seed = resolve_seed(req.seed)
    prompt = req.prompt
    negative_prompt = req.negative_prompt or ""
    sample_steps = req.sample_steps
    guide_scale = req.guide_scale
    sample_shift = req.sample_shift
    image_url = req.image_url

    def build() -> dict:
        image_name = upload_image(
            download_bytes(image_url), f"i2v_{uuid.uuid4().hex[:8]}.png"
        )
        workflow = apply_common(
            load_workflow("i2v_api.json"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frame_num=frame_num,
            seed=seed,
            sample_steps=sample_steps,
            guide_scale=guide_scale,
            sample_shift=sample_shift,
            filename_prefix=f"api/i2v_{uuid.uuid4().hex[:8]}",
        )
        workflow["56"]["inputs"]["image"] = image_name
        return workflow

    return _enqueue("i2v", seed, build)


@app.post("/flf2v", response_model=SubmitResponse)
def flf2v(req: FLF2VRequest):
    width, height = parse_size(req.size)
    frame_num = validate_frame_num(req.frame_num)
    seed = resolve_seed(req.seed)
    prompt = req.prompt
    negative_prompt = req.negative_prompt or ""
    sample_steps = req.sample_steps
    guide_scale = req.guide_scale
    sample_shift = req.sample_shift
    start_image_url = req.start_image_url
    end_image_url = req.end_image_url

    def build() -> dict:
        start_name = upload_image(
            download_bytes(start_image_url), f"flf_start_{uuid.uuid4().hex[:8]}.png"
        )
        end_name = upload_image(
            download_bytes(end_image_url), f"flf_end_{uuid.uuid4().hex[:8]}.png"
        )
        workflow = apply_common(
            load_workflow("flf2v_api.json"),
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            frame_num=frame_num,
            seed=seed,
            sample_steps=sample_steps,
            guide_scale=guide_scale,
            sample_shift=sample_shift,
            filename_prefix=f"api/flf2v_{uuid.uuid4().hex[:8]}",
        )
        workflow["56"]["inputs"]["image"] = start_name
        workflow["60"]["inputs"]["image"] = end_name
        return workflow

    return _enqueue("flf2v", seed, build)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    wait_comfy_ready()
    import uvicorn

    logger.info("API listening on %s:%s  DOWNLOAD_URL=%s", HOST, PORT, DOWNLOAD_URL)
    uvicorn.run(app, host=HOST, port=PORT, timeout_keep_alive=120, log_level="info")


if __name__ == "__main__":
    main()
