# FunASR-API 部署指南

快速部署 FunASR-API 语音识别服务，支持 CPU/macOS 与 NVIDIA GPU 两种运行形态。

如果你正在继续验证本轮 CUDA 官方 vLLM 迁移，请同时参考：

- [PENDING_CUDA_VLLM_HANDOFF.md](./TODO/PENDING_CUDA_VLLM_HANDOFF.md)

依赖安装现已统一到 [pyproject.toml](/Users/quant/Documents/funasr-api/pyproject.toml)：

| 模式 | 命令 | 说明 |
|------|------|------|
| CPU | `uv sync --group cpu` | Linux/CPU 运行时 |
| GPU | `uv sync --group gpu` | Linux/NVIDIA 运行时，官方 vLLM nightly |

## 快速部署

### GPU 版本部署（推荐）

适用于生产环境，提供更快的推理速度：

**前置要求：**
- NVIDIA GPU (CUDA 12.6+)
- 已安装 NVIDIA Container Toolkit
- 显存 6GB+（推荐 16GB+ 以支持 Qwen3-ASR 1.7B）

```bash
# 使用 docker run（带模型挂载）
docker run -d --name funasr-api \
  --gpus all \
  -p 17003:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  -e DEVICE=auto \
  quantatrisk/funasr-api:gpu-latest

# 或使用 docker-compose（推荐）
docker-compose up -d
```

### 多 GPU 自动并行部署（推荐）

适用于并发量较高场景。该方案通过容器 entrypoint 自动完成：
- 根据 `CUDA_VISIBLE_DEVICES` 拉起多个 ASR 实例（每张卡 1 个实例）
- 容器内自动生成 Nginx upstream 并负载均衡到各实例
- 对外仍只暴露一个服务端口（默认 `8000`）

你不需要手工维护多个 `docker-compose` 服务块或手工维护 nginx upstream。

```bash
# 4 卡示例：GPU0,1,2,3 各启动 1 个实例
CUDA_VISIBLE_DEVICES=0,1,2,3 docker-compose up -d
```

常用组合：
- 单卡（保持默认）：`CUDA_VISIBLE_DEVICES=0`
- 双卡：`CUDA_VISIBLE_DEVICES=0,1`
- 四卡：`CUDA_VISIBLE_DEVICES=0,1,2,3`

**服务访问地址：**
- API 服务: `http://localhost:17003`
- API 文档: `http://localhost:17003/docs`

### CPU 版本部署

适用于开发测试或无 GPU 环境：

```bash
docker run -d --name funasr-api \
  -p 17003:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  -e DEVICE=cpu \
  quantatrisk/funasr-api:cpu-latest
```

**注意：** CPU 版本不使用 GPU/vLLM 路径。
当前 CPU 镜像已集成 QwenASR Rust backend，会自动选择 `qwen3-asr-0.6b`。
CUDA vLLM 与 CPU Rust 路径下，`word_timestamps=true` 会自动调用 forced aligner 返回字词级时间戳。
当前运行时 / 设备默认值以主 README 为准：

- [README.md](/Users/quant/Documents/funasr-api/README.md)
- [docs/README_zh.md](/Users/quant/Documents/funasr-api/docs/README_zh.md)
设计背景与实现思路可参考：

- 当前 Qwen3 后端：`CUDA -> vLLM`、`CPU/macOS -> vendored QwenASR Rust`
- 引用项目 [QwenASR](https://github.com/huanglizhuo/QwenASR)

### macOS / Apple Silicon 本地部署

适用于 M1/M2/M3/M4 机器上的本地 Qwen3-ASR 推理。当前 macOS 已统一走 vendored QwenASR Rust CPU backend。

```bash
uv sync --group cpu
uv run python start.py
```

### 验证部署

```bash
# 健康检查
curl http://localhost:17003/stream/v1/asr/health

# 查看可用模型
curl http://localhost:17003/stream/v1/asr/models

# 测试语音识别（阿里云协议）
curl -X POST "http://localhost:17003/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @test.wav

# 测试 OpenAI 兼容接口
curl -X POST "http://localhost:17003/v1/audio/transcriptions" \
  -H "Authorization: Bearer any" \
  -F "file=@test.wav" \
  -F "model=qwen3-asr-1.7b"
```

## 从源码构建镜像

### 使用构建脚本

项目提供了一个更薄的 `build.sh` 包装层，用于统一 `docker buildx` 参数：

```bash
# 构建所有版本（CPU + GPU）
./build.sh

# 仅构建 GPU 版本
./build.sh -t gpu

# 构建指定版本并推送
./build.sh -t all -v 1.0.0 -p

# 查看帮助
./build.sh -h
```

**构建脚本参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-a, --arch` | 目标架构: `amd64`, `arm64`, `multi` | `amd64` |
| `-t, --type` | 构建类型: `cpu`, `gpu`, `all` | `all` |
| `-v, --version` | 版本标签 | `latest` |
| `-p, --push` | 构建后推送到 Docker Hub | 否 |
| `-e, --export` | 导出单架构镜像为 tar.gz | 否 |
| `-o, --output` | 导出目录 | `.` |
| `-r, --registry` | 镜像仓库 | `quantatrisk` |
| `-n, --no-cache` | 禁用 Docker 构建缓存 | 否 |

### 手动构建

```bash
# 构建 CPU 版本
docker build -t funasr-api:cpu-latest -f Dockerfile.cpu .

# 构建 GPU 版本
docker build -t funasr-api:gpu-latest -f Dockerfile.gpu .
```

### 模型说明

服务支持以下 ASR 模型：

| 模型 | 说明 | 适用场景 |
|------|------|----------|
| Qwen3-ASR-1.7B ⭐ | 多语言 ASR（52种语言+方言，字级时间戳） | CUDA |
| Qwen3-ASR-0.6B | 轻量版多语言 ASR | CUDA / CPU Rust / macOS |
| Paraformer Large | WebSocket 实时识别能力 | CPU/GPU 均可 |

**运行时模型选择：**

系统根据机器资源自动选择合适的 Qwen3-ASR 模型：
- **显存 >= 32GB**: 自动加载 `qwen3-asr-1.7b`
- **显存 < 32GB**: 自动加载 `qwen3-asr-0.6b`
- **无 CUDA**: 自动加载基于 vendored Rust 的 `qwen3-asr-0.6b`
- **macOS / Apple Silicon**: 无论内存大小多少，默认都加载 `qwen3-asr-0.6b`；只有调用方显式指定时才使用 `qwen3-asr-1.7b`

`paraformer-large` realtime capability 会始终为 WebSocket 流式识别准备。

### 模型下载

当前默认启用 `HF_HUB_LOCAL_FILES_ONLY=1`，推荐在启动前先准备模型缓存。
推荐方式：

```bash
# 交互式导出当前运行计划所需模型
./scripts/prepare-models.sh

# 或直接使用项目 CLI
uv run python -m app.utils.download_models
uv run python -m app.utils.download_models --export-dir ./models
```

离线部署时，推荐目录结构：

```text
./models/
  modelscope/
  huggingface/
```

然后保持与 compose 文件一致的挂载：

```yaml
volumes:
  - ./models/modelscope:/root/.cache/modelscope
  - ./models/huggingface:/root/.cache/huggingface
```

## 环境变量配置

### 基础配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `HOST` | `0.0.0.0` | 服务绑定地址 |
| `PORT` | `8000` | 服务端口 |
| `DEBUG` | `false` | 调试模式（启用后可访问 /docs） |
| `LOG_LEVEL` | `INFO` | 日志级别：DEBUG, INFO, WARNING, ERROR |
| `WORKERS` | `1` | 工作进程数（多进程会复制模型，显存成倍增加） |
| `MAX_AUDIO_SIZE` | `2048` | 最大音频文件大小（MB，支持单位如 2GB） |
| `API_KEY` | - | 服务端统一鉴权密钥 |

### 设备配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DEVICE` | `auto` | 设备选择：`auto`, `cpu`, `cuda:0` |
| `CUDA_VISIBLE_DEVICES` | `0` | 可见的 GPU 设备，控制启动实例数量 |

### 内置 Nginx 与限流配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `NGINX_RATE_LIMIT_RPS` | `0` | 全局每秒请求上限，`0` 表示关闭 |
| `NGINX_RATE_LIMIT_BURST` | `0` | 全局突发请求数，`0` 时自动取 `NGINX_RATE_LIMIT_RPS` |

### ASR 模型配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ASR_ENABLE_REALTIME_PUNC` | `true` | 是否启用实时标点模型 |

### 性能优化配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ASR_BATCH_SIZE` | `4` | 长音频分段后的 ASR 批处理大小 |
| `INFERENCE_THREAD_POOL_SIZE` | 自动 | 推理线程池大小；默认按 CPU 核数自动设置 |
| `MAX_SEGMENT_SEC` | `30` | 音频分段最大时长（秒） |
| `WS_MAX_BUFFER_SIZE` | `160000` | WebSocket 音频缓冲区大小（样本数） |
| `QWEN_RUST_CPU_WORKERS` | `4` | CPU Rust backend worker 数；Rust ASR / forced align 默认 4 个 runtime |
| `QWEN_RUST_ASR_CONCURRENCY` | `0` | Rust ASR 阶段批内并行度；`0` 表示跟随 `QWEN_RUST_CPU_WORKERS` |
| `QWEN_RUST_ALIGN_CONCURRENCY` | `0` | Rust forced align 阶段批内并行度；`0` 表示跟随 `QWEN_RUST_CPU_WORKERS` |
| `QWENASR_CPU_NUM_THREADS` | 自动 / 安全 `1` | 覆盖单个 Rust runtime 的 CPU 线程数 |
| `QWENASR_LIBRARY_PATH` | 自动探测 | 覆盖 vendored Rust 动态库路径 |

### 远场过滤配置

流式 ASR 远场声音过滤功能，自动过滤远场声音和环境音：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ASR_ENABLE_NEARFIELD_FILTER` | `true` | 启用远场声音过滤 |
| `ASR_NEARFIELD_RMS_THRESHOLD` | `0.01` | RMS 能量阈值 |
| `LOG_LEVEL=DEBUG` | - | 需要观察过滤细节时打开调试日志 |

调优建议：

- `ASR_NEARFIELD_RMS_THRESHOLD=0.01` 是当前默认值，也是推荐起点
- 嘈杂环境可以适当调高，增强背景语音过滤
- 安静环境如果出现小声说话漏识别，可以适当调低
- 需要观察过滤行为时，可临时设置 `LOG_LEVEL=DEBUG`

### 鉴权配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `API_KEY` | - | 服务端统一鉴权密钥；同时兼容 `Authorization: Bearer` 和 `X-NLS-Token` |

**使用示例：**

```bash
# 使用 Token
curl -H "X-NLS-Token: your_token" http://localhost:8000/stream/v1/asr/health

# 使用 Bearer Token（OpenAI 兼容）
curl -H "Authorization: Bearer your_token" http://localhost:8000/v1/models
```

### 日志配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LOG_LEVEL` | `INFO` | 日志级别：`DEBUG`, `INFO`, `WARNING` |
| `LOG_FILE` | `logs/funasr-api.log` | 日志文件路径 |
| `LOG_MAX_BYTES` | `20971520` | 单个日志文件最大大小（20MB） |
| `LOG_BACKUP_COUNT` | `50` | 日志备份文件数量 |

## Docker Compose 配置

### 基础配置（GPU）

```yaml
services:
  funasr-api:
    image: quantatrisk/funasr-api:gpu-latest
    container_name: funasr-api
    ports:
      - "17003:8000"
    volumes:
      - ./models/modelscope:/root/.cache/modelscope
      - ./models/huggingface:/root/.cache/huggingface
      - ./temp:/app/temp
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - DEVICE=auto
      - ASR_BATCH_SIZE=4
      - WORKERS=1
      - INFERENCE_THREAD_POOL_SIZE=4
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### CPU 版本配置

```yaml
services:
  funasr-api:
    image: quantatrisk/funasr-api:cpu-latest
    container_name: funasr-api
    ports:
      - "17003:8000"
    volumes:
      - ./models/modelscope:/root/.cache/modelscope
      - ./temp:/app/temp
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - DEVICE=cpu
      - WORKERS=1
      - INFERENCE_THREAD_POOL_SIZE=1
    restart: unless-stopped
```

### 生产环境配置（内置 Nginx，推荐）

```yaml
services:
  funasr-api:
    image: quantatrisk/funasr-api:gpu-latest
    container_name: funasr-api
    ports:
      - "17003:8000"
    volumes:
      - ./models/modelscope:/root/.cache/modelscope
      - ./models/huggingface:/root/.cache/huggingface
      - ./temp:/app/temp
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - DEBUG=false
      - LOG_LEVEL=INFO
      - DEVICE=auto
      - CUDA_VISIBLE_DEVICES=0,1
      - NGINX_RATE_LIMIT_RPS=20
      - NGINX_RATE_LIMIT_BURST=40
      - WORKERS=1
      - INFERENCE_THREAD_POOL_SIZE=4
      - ASR_BATCH_SIZE=4
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 服务监控

### 健康检查

```bash
curl http://localhost:17003/stream/v1/asr/health
```

### 日志监控

```bash
# 实时查看日志
docker logs -f funasr-api

# 查看错误日志
docker logs funasr-api 2>&1 | grep -i error
```

### 资源监控

```bash
# 容器资源使用
docker stats funasr-api

# GPU 使用情况
docker exec -it funasr-api nvidia-smi
```

## 资源需求

### 最小配置（CPU 版本）

- CPU: 4 核
- 内存: 8GB
- 磁盘: 10GB

### 推荐配置（GPU 版本）

- CPU: 8 核
- 内存: 16GB
- GPU: NVIDIA GPU (6GB+ 显存，含说话人分离模型)
- 磁盘: 25GB

## 故障排除

### 常见问题

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| GPU 内存不足 | CUDA OOM 错误 | 设置 `DEVICE=cpu` 或使用更大显存的 GPU |
| 模型加载失败 / 缓慢 | 本地模型缓存缺失 | 先运行 `./scripts/prepare-models.sh` 或 `uv run python -m app.utils.download_models` 预准备模型 |
| 端口被占用 | 端口冲突错误 | 修改端口映射：`"8080:8000"` |
| 说话人分离失败 | CAM++ 模型错误 | 检查模型是否完整下载，显存是否充足 |

### 调试模式

```bash
# 启用调试模式
docker run -e DEBUG=true -e LOG_LEVEL=DEBUG ...

# 进入容器调试
docker exec -it funasr-api /bin/bash
```

## 更新服务

```bash
# 拉取最新镜像（GPU 版本）
docker pull quantatrisk/funasr-api:gpu-latest

# 拉取最新镜像（CPU 版本）
docker pull quantatrisk/funasr-api:cpu-latest

# 重启服务
docker-compose down && docker-compose up -d
```
