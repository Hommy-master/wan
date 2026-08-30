# Wan2.2 TI2V-5B（ComfyUI）Docker 部署与 API 说明

基于 **ComfyUI** 运行 [Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2)，对外提供 **异步** RESTful API，支持：

- **文生视频** `POST /t2v` → 返回 `task_id`
- **图生视频** `POST /i2v` → 返回 `task_id`
- **首尾帧生视频** `POST /flf2v` → 返回 `task_id`
- **查询任务** `GET /tasks/{task_id}` → 进度与结果

同一时间 **仅执行 1 个生成任务**；新请求进入队列等待。

容器启动后会自动下载 ComfyUI 格式模型（支持断点续传），拉起 ComfyUI，再启动 FastAPI 网关（默认端口 `8000`）。

## 环境要求

- Docker / Docker Compose
- NVIDIA GPU（建议显存 ≥ 12GB；3060 12GB 可用，建议适当降低 `sample_steps`）
- NVIDIA Container Toolkit（`gpus: all`）
- 系统内存建议 ≥ 32GB

## 快速启动

```bash
cd docker
docker-compose up -d --build
```

查看日志：

```bash
docker logs wan -f
```

健康检查（`ready=true` 表示 ComfyUI 已就绪）：

```bash
curl http://127.0.0.1:8000/health
```

交互式文档：`http://127.0.0.1:8000/docs`

## 目录与挂载

| 宿主机 | 容器内 | 说明 |
| --- | --- | --- |
| `docker/models` | `/app/ComfyUI/models` | 模型与 HF 缓存（断点续传） |
| `docker/output` | `/app/output` | API 输出视频 |

模型文件布局：

```text
models/
├── diffusion_models/wan2.2_ti2v_5B_fp16.safetensors
├── text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
└── vae/wan2.2_vae.safetensors
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DOWNLOAD_URL` | `http://127.0.0.1/` | 将 `/app/` 替换为该前缀，得到下载 URL |
| `HF_ENDPOINT` | `https://huggingface.co` | Hugging Face 地址；国内可用 `https://hf-mirror.com` |
| `HF_TOKEN` | 空 | Hugging Face Token（可选） |
| `COMFY_POLL_TIMEOUT` | `7200` | 单次生成最长等待秒数 |
| `MAX_QUEUE_SIZE` | `32` | 等待队列上限（不含正在执行的任务） |

输出 URL 示例：

- 容器路径：`/app/output/t2v_xxx.mp4`
- `DOWNLOAD_URL=http://127.0.0.1/`
- 返回：`http://127.0.0.1/output/t2v_xxx.mp4`

## RESTful API（异步）

### 调用流程

1. `POST /t2v`（或 `/i2v` / `/flf2v`）提交任务，立即返回 `task_id`
2. 轮询 `GET /tasks/{task_id}` 查看 `status` / `progress`
3. `status=succeeded` 时读取 `video_path` / `video_url`

### 提交响应

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "t2v",
  "status": "queued",
  "message": "accepted"
}
```

### 查询任务 `GET /tasks/{task_id}`

| 字段 | 说明 |
| --- | --- |
| `status` | `queued` / `running` / `succeeded` / `failed` |
| `progress` | `0.0`～`1.0` |
| `message` | 当前阶段说明 |
| `seed` | 实际使用的种子 |
| `video_path` / `video_url` | 成功时才有 |
| `error` | 失败原因 |

成功示例：

```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "task": "t2v",
  "status": "succeeded",
  "progress": 1.0,
  "message": "succeeded",
  "seed": 123456789,
  "video_path": "/app/output/t2v_20260829_210000_ab12cd34.mp4",
  "video_url": "http://127.0.0.1/output/t2v_20260829_210000_ab12cd34.mp4",
  "error": null,
  "created_at": "2026-08-30T02:00:00+00:00",
  "updated_at": "2026-08-30T02:35:00+00:00",
  "started_at": "2026-08-30T02:00:01+00:00",
  "finished_at": "2026-08-30T02:35:00+00:00"
}
```

### 公共参数

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `prompt` | string | 必填 | 提示词 |
| `negative_prompt` | string | 内置中文负向词 | 负向提示词 |
| `size` | string | `1280*704` | 仅支持 `1280*704` / `704*1280` |
| `frame_num` | int | `121` | 帧数，须为 `4n+1`（121≈5 秒@24fps） |
| `seed` | int | `-1` | `-1` 表示随机 |
| `sample_steps` | int | `20` | 采样步数（3060 可试 `12~20`） |
| `sample_shift` | float | `8.0` | ModelSamplingSD3 shift |
| `guide_scale` | float | `5.0` | CFG |

### 1. 文生视频

`POST /t2v`

```bash
curl -X POST http://127.0.0.1:8000/t2v \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "两只拟人化的猫穿着拳击装备，在聚光灯舞台上激烈对打",
    "size": "1280*704",
    "frame_num": 121,
    "sample_steps": 20
  }'
```

### 2. 图生视频

`POST /i2v`  
`image_url` 必须是 `http` / `https`。

```bash
curl -X POST http://127.0.0.1:8000/i2v \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "夏日海滩度假风格，一只戴墨镜的白猫坐在冲浪板上",
    "image_url": "https://example.com/cat.jpg",
    "size": "1280*704",
    "frame_num": 121,
    "sample_steps": 20
  }'
```

### 3. 首尾帧生视频

`POST /flf2v`  
提供起始帧与结束帧 URL，生成中间过渡视频。

```bash
curl -X POST http://127.0.0.1:8000/flf2v \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "镜头从近景平滑拉远，人物转身走向远方",
    "start_image_url": "https://example.com/start.png",
    "end_image_url": "https://example.com/end.png",
    "size": "1280*704",
    "frame_num": 121,
    "sample_steps": 20
  }'
```

### 查询进度与结果

```bash
curl http://127.0.0.1:8000/tasks/<task_id>
```

### 健康检查

`GET /health`

```json
{
  "status": "ok",
  "model": "Wan2.2-TI2V-5B",
  "engine": "ComfyUI",
  "ready": true,
  "queue": {
    "queued": 0,
    "running_task_id": null,
    "max_concurrent": 1,
    "max_queue_size": 32
  }
}
```

## 实现说明

- 推理引擎：ComfyUI（官方 Wan2.2 5B 节点）
- 首尾帧：自定义节点 [Wan22FirstLastFrameToVideoLatent](https://github.com/stduhpf/ComfyUI--Wan22FirstLastFrameToVideoLatent)
- 网关：`server.py` 异步任务队列 + 单 worker；进度可经 ComfyUI WebSocket 更新
- 工作流模板：`docker/workflows/{t2v,i2v,flf2v}_api.json`

## 常用命令

```bash
cd docker
docker-compose up -d --build
docker-compose down
docker logs wan -f
```

国内下载模型慢时：

```bash
HF_ENDPOINT=https://hf-mirror.com docker-compose up -d
```

## 注意事项

1. 首次启动需下载约十余 GB 模型，请保持 `docker/models` 挂载以便续传
2. 生成接口为异步：提交立即返回；请轮询 `GET /tasks/{task_id}`
3. 同一时间只执行 1 个任务，其余排队；队列满返回 `429`
4. 3060 12GB 建议 `sample_steps=12~20`，并保证系统内存充足
5. `ready=false` 时不要提交任务（ComfyUI 尚未就绪）
