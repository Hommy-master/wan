# Wan2.2 TI2V-5B Docker 部署与 API 说明

本项目基于 [Wan2.2](https://github.com/Wan-Video/Wan2.2) 的 **Wan2.2-TI2V-5B** 模型，提供 Docker 容器化部署，以及文生视频（t2v）、图生视频（i2v）的 RESTful API。

容器启动后会自动下载（支持断点续传）并加载模型，服务监听 `8000` 端口。

## 环境要求

- Docker / Docker Compose
- NVIDIA GPU（建议显存 ≥ 24GB，如 RTX 4090）
- NVIDIA Container Toolkit（`--gpus` 可用）
- 宿主机内存建议 ≥ 64GB（加载 T5 + DiT 时占用较大）

## 快速启动

在 `docker` 目录下执行：

```bash
cd docker
docker-compose up -d --build
```

说明：

- 首次启动会下载 **Wan2.2-TI2V-5B**（约 20GB+），支持断点续传
- 模型目录挂载到宿主机 `./Wan2.2-TI2V-5B`，输出目录挂载到 `./output`
- 启动成功后日志中会出现 `WanTI2V ready` 与 `API listening on 0.0.0.0:8000`

查看日志：

```bash
docker logs wan -f
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

返回示例：

```json
{"status":"ok","model":"Wan2.2-TI2V-5B","ready":true}
```

`ready` 为 `true` 表示模型已加载完成，可以调用生成接口。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DOWNLOAD_URL` | `http://127.0.0.1/` | 将容器内路径 `/app/` 替换为该前缀，得到可下载的 URL |
| `MODEL_SOURCE` | `huggingface` | 模型来源：`huggingface` 或 `modelscope` |
| `HF_ENDPOINT` | `https://huggingface.co` | Hugging Face 镜像地址，国内可用 `https://hf-mirror.com` |
| `HF_TOKEN` | 空 | Hugging Face Token（可选） |
| `MODELSCOPE_TOKEN` | 空 | ModelScope Token（可选） |

示例（国内优先用 ModelScope）：

```bash
MODEL_SOURCE=modelscope docker-compose up -d
```

示例（使用 Hugging Face 镜像站）：

```bash
HF_ENDPOINT=https://hf-mirror.com docker-compose up -d
```

## 目录约定

| 宿主机 | 容器内 | 说明 |
| --- | --- | --- |
| `docker/Wan2.2-TI2V-5B` | `/app/Wan2.2-TI2V-5B` | 模型权重（断点续传缓存也在此） |
| `docker/output` | `/app/output` | 生成的视频文件 |

输出 URL 规则：将容器路径中的 `/app/` 替换为 `DOWNLOAD_URL`。

例如：

- 容器路径：`/app/output/t2v_xxx.mp4`
- `DOWNLOAD_URL=http://127.0.0.1/`
- 返回 URL：`http://127.0.0.1/output/t2v_xxx.mp4`

## RESTful API

默认地址：`http://127.0.0.1:8000`  
交互式文档：`http://127.0.0.1:8000/docs`

### 1. 服务信息

`GET /`

```bash
curl http://127.0.0.1:8000/
```

### 2. 健康检查

`GET /health`

```bash
curl http://127.0.0.1:8000/health
```

### 3. 文生视频（t2v）

`POST /t2v`  
`Content-Type: application/json`

请求参数：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `prompt` | string | 是 | - | 文本提示词 |
| `negative_prompt` | string | 否 | `""` | 负向提示词 |
| `size` | string | 否 | `1280*704` | 分辨率，仅支持 `1280*704` 或 `704*1280` |
| `frame_num` | int | 否 | `121` | 帧数，须为 `4n+1` |
| `seed` | int | 否 | `-1` | 随机种子，`-1` 表示随机 |
| `sample_steps` | int | 否 | 模型默认 | 采样步数 |
| `sample_shift` | float | 否 | 模型默认 | Flow Matching shift |
| `guide_scale` | float | 否 | 模型默认 | CFG 引导系数 |
| `sample_solver` | string | 否 | `unipc` | 采样器：`unipc` 或 `dpm++` |

示例：

```bash
curl -X POST http://127.0.0.1:8000/t2v \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "两只拟人化的猫穿着拳击装备，在聚光灯舞台上激烈对打",
    "size": "1280*704",
    "frame_num": 121,
    "seed": -1
  }'
```

成功响应示例：

```json
{
  "task": "t2v",
  "video_path": "/app/output/t2v_20260829_120000_ab12cd34.mp4",
  "video_url": "http://127.0.0.1/output/t2v_20260829_120000_ab12cd34.mp4",
  "seed": 123456789
}
```

### 4. 图生视频（i2v）

`POST /i2v`  
`Content-Type: application/json`

在 t2v 参数基础上增加：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `image_url` | string | 是 | 输入图片 URL，**必须是 http 或 https** |

说明：

- 服务会从 `image_url` 下载图片后生成视频
- `size` 表示目标像素面积，输出宽高比会尽量跟随输入图片

示例：

```bash
curl -X POST http://127.0.0.1:8000/i2v \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "夏日海滩度假风格，一只戴墨镜的白猫坐在冲浪板上",
    "image_url": "https://example.com/cat.jpg",
    "size": "1280*704",
    "frame_num": 121
  }'
```

成功响应示例：

```json
{
  "task": "i2v",
  "video_path": "/app/output/i2v_20260829_120100_ef56gh78.mp4",
  "video_url": "http://127.0.0.1/output/i2v_20260829_120100_ef56gh78.mp4",
  "seed": 987654321
}
```

## 常用运维命令

```bash
# 启动 / 重建
cd docker
docker-compose up -d --build

# 停止
docker-compose down

# 查看日志
docker logs wan -f

# 查看容器状态
docker ps
```

## 注意事项

1. 模型下载与加载期间，`/health` 中 `ready` 可能为 `false`，此时调用 `/t2v`、`/i2v` 会返回 `503`
2. 生成接口为同步调用，耗时较长，请适当增大客户端超时时间
3. 若容器反复重启且日志停在 `loading WanTI2V`，多为内存不足（OOM），请检查 Docker 内存限制与宿主机可用内存
4. 访问不了 Hugging Face 时，请设置 `MODEL_SOURCE=modelscope` 或 `HF_ENDPOINT=https://hf-mirror.com`
