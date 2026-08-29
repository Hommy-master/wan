# Wan2.2 TI2V-5B（ComfyUI）Docker 部署与 API 说明

基于 **ComfyUI** 运行 [Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2)，对外提供 RESTful API，支持：

- **文生视频** `POST /t2v`
- **图生视频** `POST /i2v`
- **首尾帧生视频** `POST /flf2v`

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

输出 URL 示例：

- 容器路径：`/app/output/t2v_xxx.mp4`
- `DOWNLOAD_URL=http://127.0.0.1/`
- 返回：`http://127.0.0.1/output/t2v_xxx.mp4`

## RESTful API

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

成功响应示例：

```json
{
  "task": "t2v",
  "video_path": "/app/output/t2v_20260829_210000_ab12cd34.mp4",
  "video_url": "http://127.0.0.1/output/t2v_20260829_210000_ab12cd34.mp4",
  "seed": 123456789
}
```

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

### 健康检查

`GET /health`

```json
{"status":"ok","model":"Wan2.2-TI2V-5B","engine":"ComfyUI","ready":true}
```

## 实现说明

- 推理引擎：ComfyUI（官方 Wan2.2 5B 节点）
- 首尾帧：自定义节点 [Wan22FirstLastFrameToVideoLatent](https://github.com/stduhpf/ComfyUI--Wan22FirstLastFrameToVideoLatent)
- 网关：`server.py` 将 REST 请求转为 ComfyUI `/prompt` 工作流并回写 `/app/output`
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
2. 生成接口为同步调用，耗时较长，请加大客户端超时
3. 3060 12GB 建议 `sample_steps=12~20`，并保证系统内存充足
4. `ready=false` 时不要调用生成接口（ComfyUI 尚未就绪）
