# Qwen3-ASR 部署指南

Ascend 910B 使用独立运行时和兼容矩阵，参见
[Ascend 910B 部署](./deployment-ascend.md)。

快速部署 Qwen3-ASR 语音识别服务，支持 CPU/macOS 与 NVIDIA GPU 两种运行形态。

依赖安装现在改成根目录默认 GPU，CPU 为单独特化环境：

| 模式 | 命令 | 说明 |
|------|------|------|
| GPU | `uv sync` | Linux/NVIDIA 运行时，默认锁定 CUDA 12.8/cu128 `torch/torchaudio/torchvision` + `vllm[audio]==0.19.0` |
| CPU | `./scripts/sync_cpu_env.sh` | Linux/CPU 运行时 |

## 快速部署

### GPU 版本部署（推荐）

适用于生产环境，提供更快的推理速度：

**前置要求：**
- NVIDIA GPU（默认镜像面向 CUDA 12.8+；CUDA 12.6 / 13.0 可通过构建参数覆盖）
- 已安装 NVIDIA Container Toolkit
- 显存 12GB+（推荐 16GB+ 以支持 Qwen3-ASR 1.7B）

```bash
# 使用 docker run（带模型挂载）
docker run -d --name qwen3-asr \
  --gpus all \
  -p 17003:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  quantatrisk/qwen3-asr:gpu-latest

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
docker run -d --name qwen3-asr \
  -p 17003:8000 \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  -v ./data:/app/data \
  -v ./logs:/app/logs \
  -v ./temp:/app/temp \
  quantatrisk/qwen3-asr:cpu-latest
```

CPU 镜像使用 QwenASR Rust，并自动选择 `qwen3-asr-0.6b`。x86_64 需要
`avx2` 与 `fma`；`word_timestamps=true` 会加载 forced aligner。构建目标与
运行时选择见主 README。

### macOS / Apple Silicon 本地部署

适用于 M1/M2/M3/M4 机器上的本地 Qwen3-ASR 推理。当前 macOS 已统一走 vendored QwenASR Rust CPU backend。

```bash
./scripts/sync_cpu_env.sh
source .venv/bin/activate
python start.py
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
./build.sh -t all -v 1.0.3 -p

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
docker build -t qwen3-asr:cpu-latest -f Dockerfile.cpu .

# 构建绑定当前机器指令集的 CPU 版本（仅适合同构部署）
docker build -t qwen3-asr:cpu-native -f Dockerfile.cpu \
  --build-arg QWENASR_RUST_TARGET_CPU=native \
  .

# 构建默认 GPU 版本（CUDA 12.8 / PyTorch cu128）
docker build -t qwen3-asr:gpu-latest -f Dockerfile.gpu .

# 构建 CUDA 12.6 版本
docker build -t qwen3-asr:gpu-cu126 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-12-6 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" \
  .

# 构建 CUDA 13.0 版本
docker build -t qwen3-asr:gpu-cu130 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-13-0 \
  --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  .
```

`Dockerfile.cpu` 可覆盖的 CPU 构建参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QWENASR_RUST_TARGET_CPU` | `x86-64-v2` | amd64 Rust backend 编译目标；可设为 `native` 构建绑定当前 CPU 的镜像 |

`Dockerfile.gpu` 可覆盖的 GPU 构建参数：

| 参数 | 默认值 | 用途 |
|------|--------|------|
| `PYTORCH_BASE_IMAGE` | `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` | 选择 PyTorch/CUDA 基础镜像 |
| `PYTORCH_CUDA_INDEX` | `https://download.pytorch.org/whl/cu128` | 选择 PyTorch wheel CUDA 后端 |
| `CUDA_NVCC_PACKAGE` | `cuda-nvcc-12-8` | 安装匹配的 nvcc，用于 vLLM/FlashInfer JIT |
| `TORCH_CUDA_ARCH_LIST` | `12.0+PTX` | 指定 JIT 编译目标架构 |
| `VLLM_PACKAGE` | `vllm[audio]==0.19.0` | 覆盖 vLLM 包版本或来源 |

### 模型下载

启动时会先检测当前运行计划所需模型；如果本地缓存缺失，会自动下载。离线部署可显式设置 `HF_HUB_OFFLINE=1` 并提前准备模型缓存。
手动准备方式：

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
  - ./data:/app/data
```

## 环境变量配置

复制 `.env.example` 为 `.env`，仅修改其中已说明的公开变量。GPU 和 CPU
Compose 文件是运行时配置的唯一来源；其余调优参数保留代码默认值。

## 服务监控

### 健康检查

```bash
curl http://localhost:17003/stream/v1/asr/health
```

### 日志监控

```bash
# 实时查看日志
docker logs -f qwen3-asr

# 查看错误日志
docker logs qwen3-asr 2>&1 | grep -i error
```

### 资源监控

```bash
# 容器资源使用
docker stats qwen3-asr

# GPU 使用情况
docker exec -it qwen3-asr nvidia-smi
```

## 资源需求

### 最小配置（CPU 版本）

- CPU: 4 核
- 内存: 16GB
- 磁盘: 20GB

### 推荐配置（GPU 版本）

- CPU: 4 核
- 内存: 16GB
- GPU: NVIDIA GPU (16GB+ 显存)
- 磁盘: 20GB

## 故障排除

### 常见问题

| 问题 | 症状 | 解决方案 |
|------|------|----------|
| GPU 内存不足 | CUDA OOM 错误 | 使用 `qwen3-asr-0.6b` 或部署 CPU 镜像 |
| 模型加载失败 / 缓慢 | 本地模型缓存缺失 | 先运行 `./scripts/prepare-models.sh` 或 `uv run python -m app.utils.download_models` 预准备模型 |
| 端口被占用 | 端口冲突错误 | 修改端口映射：`"8080:8000"` |
| 说话人分离失败 | CAM++ 模型错误 | 检查模型是否完整下载，显存是否充足 |

### 调试模式

```bash
# 启用调试模式
docker run -e DEBUG=true -e LOG_LEVEL=DEBUG ...

# 进入容器调试
docker exec -it qwen3-asr /bin/bash
```

## 更新服务

```bash
# 拉取最新镜像（GPU 版本）
docker pull quantatrisk/qwen3-asr:gpu-latest

# 拉取最新镜像（CPU 版本）
docker pull quantatrisk/qwen3-asr:cpu-latest

# 重启服务
docker-compose down && docker-compose up -d
```
