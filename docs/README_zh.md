<div align="center">

<h1>Qwen3-ASR</h1>
<h3>开箱即用的本地私有化部署语音识别服务</h3>

以 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 为核心的语音识别 API 服务，提供 CUDA vLLM 与 CPU Rust 两种后端，兼容阿里云语音 API 和 OpenAI Audio API，并保留 Paraformer realtime WebSocket 能力。

---

![Static Badge](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Static Badge](https://img.shields.io/badge/Torch-2.10.0-%23EE4C2C?logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/CUDA-12.8_default-%2376B900?logo=nvidia&logoColor=white)

</div>

## 在线演示站点

- **在线体验**: https://asr.vect.one

## 演示

[![演示](../demo/demo.png)](https://media.cdn.vect.one/qwenasr_client_demo.mp4)

## Release 1.0.1

> `v1.0.1` 是当前补丁版本。`v1.0.0` 相对于早期 `main` 分支引入了一轮大规模 breaking refactor。
> 如果你是从 `main` 升级过来，请先阅读 release 说明，再决定是否沿用旧的部署与运行时假设。
>
> 关键 breaking changes：
> - Python 依赖管理已经切到 `uv`（`pyproject.toml` + `uv.lock`），`requirements*.txt` 已移除
> - 运行时栈改成 `CUDA -> official vLLM`、`CPU/macOS -> vendored QwenASR Rust`
> - `MLX` / Apple Silicon GPU 路径已移除，`mps` 会归一化到 `cpu`
> - macOS / Apple Silicon 现在默认总是 `qwen3-asr-0.6b`，可通过 `QWEN3_ASR_MODEL` 覆盖
> - `ENABLED_MODELS` 已移除

## 主要特性

- **混合运行时栈** - 离线推理由自动选择的 Qwen3-ASR 提供，WebSocket 流式由 Paraformer realtime 能力提供
- **说话人分离** - 基于 CAM++ 模型自动识别多说话人，返回说话人标记
- **OpenAI API 兼容** - 支持 `/v1/audio/transcriptions` 端点，可直接使用 OpenAI SDK
- **阿里云 API 兼容** - 支持阿里云语音识别 RESTful API 和 WebSocket 流式协议
- **WebSocket 流式识别** - 支持实时流式语音识别，低延迟
- **智能远场过滤** - 流式 ASR 自动过滤远场声音和环境音，减少误触发
- **智能音频分段** - 基于 VAD 的贪婪合并算法，自动切分长音频，避免包含过长静音
- **GPU 批处理加速** - 支持批量推理，比逐个处理快 2-3 倍
- **资源感知运行时** - 根据当前机器资源自动选择合适的 Qwen3-ASR 模型

## 致谢

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 提供官方模型与多模态 / vLLM 使用方式
- [QwenASR](https://github.com/huanglizhuo/QwenASR) 提供本项目 vendored 的 CPU Rust backend

## 快速部署

### 1. Docker 部署(推荐)

```bash
# 复制并编辑配置
cp .env.example .env
# 编辑 .env 设置 API_KEY（可选）

# 启动服务（GPU 版本）
docker-compose up -d

# 或 CPU 版本
docker-compose -f docker-compose-cpu.yml up -d

# 多卡自动模式（每张可见卡自动拉起 1 个实例）
CUDA_VISIBLE_DEVICES=0,1,2,3 docker-compose up -d
```

服务访问地址：
- **API 端点**: `http://localhost:17003`
- **API 文档**: `http://localhost:17003/docs`

可选的内置限流参数：
- `NGINX_RATE_LIMIT_RPS`（全局每秒请求上限，`0` 表示关闭）
- `NGINX_RATE_LIMIT_BURST`（全局突发请求数，`0` 时自动使用 RPS）

**docker run 方式（替代）:**

```bash
# GPU 版本
docker run -d --name qwen3-asr \
  --gpus all \
  -p 17003:8000 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e API_KEY=your_api_key \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  quantatrisk/qwen3-asr:gpu-latest

# CPU 版本
docker run -d --name qwen3-asr \
  -p 17003:8000 \
  quantatrisk/qwen3-asr:cpu-latest
```

> **注意**: GPU 镜像默认使用 CUDA 12.8/cu128，以覆盖 Blackwell 等新架构 GPU。
> 开发者可通过 Docker build args 自行构建 CUDA 12.6、CUDA 13.0 或其他后端组合。
> 当前 CPU 镜像已通过内置 QwenASR Rust backend 支持 `qwen3-asr-0.6b`。默认 CPU 镜像使用可分发 Rust 构建目标；只有自建且构建机/部署机 CPU 同构时才建议设置 `QWENASR_RUST_TARGET_CPU=native`。
> CUDA vLLM 与 CPU Rust 路径下，`word_timestamps=true` 都会自动调用 forced aligner；当前实际后端为 `CUDA -> vLLM`、`CPU/macOS -> vendored QwenASR Rust`。
> Apple Silicon 上的 Qwen3-ASR 现已统一走 Rust CPU backend。

**自定义 GPU 后端构建：**

```bash
# 默认 GPU 构建：CUDA 12.8 / PyTorch cu128
docker build -t qwen3-asr:gpu-cu128 -f Dockerfile.gpu .

# CUDA 12.6 构建，用于旧部署环境
docker build -t qwen3-asr:gpu-cu126 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu126 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-12-6 \
  --build-arg TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" \
  .

# CUDA 13.0 构建，用于需要 CUDA 13 工具链的环境
docker build -t qwen3-asr:gpu-cu130 -f Dockerfile.gpu \
  --build-arg PYTORCH_BASE_IMAGE=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime \
  --build-arg PYTORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu130 \
  --build-arg CUDA_NVCC_PACKAGE=cuda-nvcc-13-0 \
  --build-arg TORCH_CUDA_ARCH_LIST="12.0+PTX" \
  .
```

**内网部署**：使用辅助脚本准备当前运行计划所需模型，然后复制到内网机器：

```bash
# 1. 准备模型
./scripts/prepare-models.sh

# 2. 复制到内网服务器
scp qwen3-asr-models-*.tar.gz user@server:/opt/qwen3-asr/

# 3. 在内网服务器解压并启动
tar -xzvf qwen3-asr-models-*.tar.gz
docker-compose up -d
```

> 详细部署说明请查看 [部署指南](./deployment.md)

### 本地开发

**系统要求:**

- Python 3.10+
- 默认 GPU 镜像要求 CUDA 12.8+；CUDA 12.6 / 13.0 可通过 Docker build args 自行构建
- FFmpeg (音频格式转换)

**安装步骤:**

运行时依赖现在改成“根目录默认 GPU，CPU 单独特化环境”：

| 模式 | 命令 | 说明 |
|------|------|------|
| GPU（默认） | `uv sync` | 同步根目录 [pyproject.toml](/opt/qwen3-asr/pyproject.toml) 和 [uv.lock](/opt/qwen3-asr/uv.lock) 到 `.venv`，包含 CUDA 12.8/cu128 `torch/torchaudio/torchvision` |
| CPU（特化） | `./scripts/sync_cpu_env.sh` | 同步 [environments/cpu/pyproject.toml](/opt/qwen3-asr/environments/cpu/pyproject.toml) 对应的 CPU lock 到 `.venv` |

```bash
# 克隆项目
cd qwen3-asr

# 安装依赖（Linux/CUDA）
uv sync

# 启动服务
source .venv/bin/activate
python start.py
```

macOS / Apple Silicon 本地开发：

```bash
./scripts/sync_cpu_env.sh
source .venv/bin/activate
python start.py
```

启动界面：

- `FUNASR_STARTUP_UI=auto` 是默认值，单 worker 且当前终端为交互 TTY 时会自动启用
- `auto` / `tui` 会启动 Textual 启动画面，由父进程接管终端；上方显示启动阶段进度，下方集中显示子进程日志
- `plain` 会关闭该界面，回退到普通终端输出
- 多 worker 模式始终回退为普通输出
- Docker / `docker-compose logs -f` 场景始终使用普通输出，因为容器日志流不是交互式 TTY

## 当前运行时默认值

当前主线代码的运行时行为如下：

- `DEVICE=auto`
  - 有 CUDA 时解析为 `cuda:0`
  - 否则解析为 `cpu`
- `DEVICE=mps` 会直接归一化为 `cpu`
- `Linux + CUDA` 使用官方 `vLLM`
- `Linux + CPU` 使用 vendored `QwenASR` Rust
- `macOS / Apple Silicon` 也使用 vendored `QwenASR` Rust
- macOS / Apple Silicon 默认总是 `qwen3-asr-0.6b`
- 在 macOS 上，只有设置 `QWEN3_ASR_MODEL=qwen3-asr-1.7b` 时才会使用 `qwen3-asr-1.7b`
- `word_timestamps=true` 在当前离线 CUDA 与 CPU Rust 路径下可用
- WebSocket 流式路径当前不返回词级时间戳
- CAM++ 说话人分离仍然必须保留，并继续跟随 `DEVICE`；在 CPU 上的主要热点仍是 speaker verification embedding

## API 接口

### OpenAI 兼容接口

| 端点                         | 方法 | 功能                    |
| ---------------------------- | ---- | ----------------------- |
| `/v1/audio/transcriptions` | POST | 音频转写（OpenAI 兼容） |
| `/v1/models`               | GET  | 离线模型列表                |

**请求参数:**

| 参数                           | 类型   | 默认值                | 说明                                  |
| ------------------------------ | ------ | --------------------- | ------------------------------------- |
| `file`                       | file   | 提供时优先使用         | 音频/视频文件                          |
| `audio_address`              | string | 可选                  | 音频/视频文件 URL（HTTP/HTTPS）；若同时提供 `file`，则忽略 |
| `language`                   | string | 自动检测              | 语言代码 (zh/en/ja)                   |
| `enable_speaker_diarization` | bool   | `true`              | 启用说话人分离                        |
| `word_timestamps`            | bool   | `false`             | 返回后端支持的字词级时间戳；Qwen CUDA vLLM 与 CPU Rust 在启用时会自动调用 forced aligner |
| `response_format`            | string | `verbose_json`      | 输出格式                              |
| `prompt`                     | string | -                     | 提示文本（保留兼容）                  |
| `temperature`                | float  | `0`                   | 采样温度（保留兼容）                  |

**音频/视频输入方式:**
- **文件上传**: 使用 `file` 参数上传音频文件或带音轨的视频容器
- **URL 下载**: 使用 `audio_address` 参数提供音频/视频 URL，服务将自动下载
- **优先级**: 如果同时提供 `file` 和 `audio_address`，服务会优先使用 `file`，并忽略 `audio_address`

**使用示例:**

```python
# 使用 OpenAI SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your_api_key")

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        file=f,
        response_format="verbose_json"  # 获取分段和说话人信息
    )
print(transcript.text)
```

```bash
# 使用 curl
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@audio.wav" \
  -F "model=qwen3-asr-0.6b" \
  -F "response_format=verbose_json" \
  -F "enable_speaker_diarization=true"
```

**支持的响应格式:** `json`, `text`, `srt`, `vtt`, `verbose_json`

### 阿里云兼容接口

| 端点                      | 方法      | 功能                   |
| ------------------------- | --------- | ---------------------- |
| `/stream/v1/asr`        | POST      | 语音识别（支持长音频） |
| `/stream/v1/asr/models` | GET       | 声明条目列表               |
| `/stream/v1/asr/health` | GET       | 健康检查               |
| `/ws/v1/asr`            | WebSocket | 流式语音识别（阿里云协议兼容） |
| `/ws/v1/asr/funasr`     | WebSocket | FunASR 流式识别（向后兼容）   |
| `/ws/v1/asr/qwen`       | WebSocket | Qwen3-ASR 流式识别 |

**请求参数:**

| 参数                           | 类型   | 默认值             | 说明                                  |
| ------------------------------ | ------ | ------------------ | ------------------------------------- |
| `audio_address`              | string | `https://media.cdn.vect.one/podcast_demo.mp4`（文档示例） | 音频/视频 URL（可选；若同时上传内容则忽略） |
| `sample_rate`                | int    | `16000`          | 采样率                                |
| `enable_speaker_diarization` | bool   | `true`           | 启用说话人分离                        |
| `word_timestamps`            | bool   | `false`          | 返回后端支持的字词级时间戳；Qwen CUDA vLLM 与 CPU Rust 在启用时会自动调用 forced aligner |
| `vocabulary_id`              | string | -                  | 热词（格式：`词1 权重1 词2 权重2`） |

**使用示例:**

```bash
# 基本用法
curl -X POST "http://localhost:8000/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav

# 带参数
curl -X POST "http://localhost:8000/stream/v1/asr?enable_speaker_diarization=true" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**响应示例:**

```json
{
  "task_id": "xxx",
  "status": 200,
  "message": "SUCCESS",
  "result": "说话人1的内容...\n说话人2的内容...",
  "duration": 60.5,
  "processing_time": 1.234,
  "segments": [
    {
      "text": "今天天气不错。",
      "start_time": 0.0,
      "end_time": 2.5,
      "speaker_id": "说话人1",
      "word_tokens": [
        {"text": "今天", "start_time": 0.0, "end_time": 0.5},
        {"text": "天气", "start_time": 0.5, "end_time": 0.9},
        {"text": "不错", "start_time": 0.9, "end_time": 1.3}
      ]
    }
  ]
}
```

## 说话人分离

基于 CAM++ 模型实现多说话人自动识别：

- **默认开启** - `enable_speaker_diarization=true`
- **自动识别** - 无需预设说话人数量，模型自动检测
- **说话人标记** - 响应中包含 `speaker_id` 字段（如 "说话人1"、"说话人2"）
- **智能合并** - 两层合并策略避免孤立短片段：
  - 第一层：小于10秒的同说话人片段累积合并
  - 第二层：连续片段累积合并至60秒上限
- **字幕支持** - SRT/VTT 格式输出包含说话人标记 `[说话人1] 文本内容`

关闭说话人分离：

```bash
# OpenAI API
-F "enable_speaker_diarization=false"

# 阿里云 API
?enable_speaker_diarization=false
```

## 音频处理

### 智能分段策略

长音频自动分段处理：

1. **VAD 语音检测** - 检测语音边界，过滤静音
2. **贪婪合并** - 累积语音段，确保每段不超过 `MAX_SEGMENT_SEC`（默认60秒）
3. **静音切分** - 语音段间静音超过3秒时强制切分，避免包含过长静音
4. **批处理推理** - 多片段并行处理，GPU 模式下性能提升 2-3 倍

### WebSocket 流式识别限制

**FunASR 模型限制**（使用 `/ws/v1/asr` 或 `/ws/v1/asr/funasr`）：
- ✅ 实时语音识别、低延迟
- ✅ 字句级时间戳
- ❌ **词级时间戳**（FunASR realtime 路径未实现）
- ❌ **置信度分数**（未实现）

**Qwen3-ASR 流式**（使用 `/ws/v1/asr/qwen`）：
- ✅ 支持多语言实时识别
- ✅ 当前支持 CUDA vLLM 与 CPU Rust 两条流式路径
- ❌ 当前流式路径不返回词级时间戳

### Qwen3 运行时矩阵

| 运行环境 | 后端 | 离线转写 | WebSocket 流式 | 离线词级时间戳 | 流式词级时间戳 | 成熟度 |
|---------|------|---------|----------------|----------------|----------------|--------|
| Linux + NVIDIA GPU | 官方 vLLM 0.19.0 | ✅ | ✅ | ✅ | ❌ | 面向生产 |
| CPU / macOS | QwenASR Rust | ✅ | ✅ | ✅（forced aligner） | ❌ | 推荐本地后端 |

## 支持离线的模型

| 模型 ID              | 名称              | 说明                                     | 特性      |
| -------------------- | ----------------- | ---------------------------------------- | --------- |
| `qwen3-asr-1.7b`   | Qwen3-ASR 1.7B    | 高性能多语言 ASR；CUDA 使用 vLLM | 离线/实时 |
| `qwen3-asr-0.6b`   | Qwen3-ASR 0.6B    | 轻量版多语言 ASR；CUDA 使用 vLLM，CPU/macOS 使用 Rust backend | 离线/实时 |

## 仅实时能力

| 能力 ID | 运行时 | 说明 |
| ------- | ------ | ---- |
| `paraformer-large` | FunASR realtime | 中文 WebSocket 实时识别栈，包含实时标点链路 |

**运行时选择:**
- **显存 >= 32GB**: 选择 `qwen3-asr-1.7b`
- **显存 < 32GB**: 选择 `qwen3-asr-0.6b`
- **无 CUDA**: 选择基于 vendored Rust 的 `qwen3-asr-0.6b`
- **macOS / Apple Silicon**: 无论内存大小多少，默认都选择 `qwen3-asr-0.6b`
- **环境变量覆盖**: 设置 `QWEN3_ASR_MODEL=qwen3-asr-1.7b` 或 `QWEN3_ASR_MODEL=qwen3-asr-0.6b` 可跳过自动选择
- `paraformer-large` 实时能力始终为 WebSocket 流式准备

启动时会先检测当前运行计划所需模型；如果本地缓存缺失，会自动下载。离线部署可显式设置 `HF_HUB_LOCAL_FILES_ONLY=1` 并提前准备模型缓存。

## 环境变量

推荐直接关心的公开配置：

| 变量                               | 默认值       | 说明                                            |
| ---------------------------------- | ------------ | ----------------------------------------------- |
| `API_KEY`                        | -            | API 认证密钥（可选，未配置时无需认证）        |
| `LOG_LEVEL`                      | `INFO`       | 日志级别（DEBUG/INFO/WARNING/ERROR）          |
| `MAX_AUDIO_SIZE`                 | `2048`       | 最大音频文件大小（MB，支持单位如 2GB）        |
| `ASR_BATCH_SIZE`                 | `4`          | 长音频分段后的 ASR 批处理大小 |
| `MAX_SEGMENT_SEC`                | `60`         | 音频分段最大时长（秒）                        |
| `ASR_ENABLE_NEARFIELD_FILTER`    | `true`       | 启用远场声音过滤                              |
| `QWEN3_ASR_MODEL`                | 自动选择      | 强制选择 `qwen3-asr-1.7b` 或 `qwen3-asr-0.6b` |

远场过滤调优建议：

- `ASR_NEARFIELD_RMS_THRESHOLD=0.01` 是当前默认值，也是推荐起点
- 嘈杂环境可以适当调高，增强背景语音过滤
- 安静环境如果出现小声说话漏识别，可以适当调低
- 需要观察过滤行为时，可临时设置 `LOG_LEVEL=DEBUG`

后端专项高级配置：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `QWEN_RUST_CPU_WORKERS` | `4` | CPU Rust backend worker 数（Rust ASR / forced align 默认 4 个 runtime） |
| `QWENASR_LIBRARY_PATH` | 自动探测 | 覆盖 vendored Rust 动态库路径 |

## 资源需求

**最小配置（CPU）:**

- CPU: 4 核
- 内存: 16GB
- 磁盘: 20GB

**推荐配置（GPU）:**

- CPU: 4 核
- 内存: 16GB
- GPU: NVIDIA GPU (16GB+ 显存)
- 磁盘: 20GB

## API 文档

启动服务后访问：

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 相关链接

- **部署指南**: [详细文档](./deployment.md)
- **Qwen3-ASR**: [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- **FunASR**: [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)
- **QwenASR**: [QwenASR GitHub](https://github.com/huanglizhuo/QwenASR)

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](../LICENSE) 文件了解详情。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=Quantatirsk/qwen3-asr&type=Date)](https://star-history.com/#Quantatirsk/qwen3-asr&Date)

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目!
