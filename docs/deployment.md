# 部署

Linux CUDA 标准入口为 `docker-compose.yml` 与 `Dockerfile.gpu`。镜像从 CUDA 13.0.2 开发镜像构建，使用 `uv.lock` 固定 Python 依赖，不依赖已有应用镜像。macOS Apple Silicon 使用原生 Rust CPU 进程，与 CUDA 共享服务协议和离线处理流程。

## 启动和生命周期

```bash
cp .env.example .env
./scripts/prepare-models.sh
docker compose up -d --build
docker compose logs -f asr
```

`start.py` 先检查所选设备、准备全部模型，再依次启动私有 R2T2 引擎和公共 API。私有进程持有唯一的 R2T2 实例，CUDA 使用 AsyncLLM，CPU 使用 Rust，同时处理实时和离线识别，只监听 `127.0.0.1:8001`。公共 API 持有 VAD、Nemotron 与独立的强制对齐模型，将非重叠语音区间交给私有引擎重新识别，并将对齐后的字词关联到 Nemotron 说话人时间轴，不加载第二份 R2T2。外部只暴露 8000。任一进程异常退出，启动器关闭全部子进程；停止时先关闭公共入口，再释放共享模型。

就绪检查同时检查两个进程的模型状态。模型下载、加载和首次编译需要时间，镜像健康检查给予 600 秒启动期。大型模型下载较慢时，建议提前准备模型缓存。

## macOS CPU

使用 Apple Silicon、Python 3.11–3.12、uv 和 Rust 工具链，在仓库根目录执行：

```bash
uv sync --frozen
./scripts/build-rust.sh
DEVICE=cpu R2T2_CPU_THREADS=8 uv run python start.py
```

访问 `http://localhost:8000`。依赖锁覆盖 macOS ARM64 与 Linux x86_64；macOS 从 PyPI 安装 PyTorch，不安装 vLLM 或 CUDA 依赖。Rust 动态库由源码构建；设置 `CARGO_TARGET_DIR` 时构建脚本沿用该目录，运行时通过 `R2T2_CPU_LIBRARY_PATH` 指定输出的动态库路径。

`DEVICE` 只接受 `cpu` 或 `cuda:0`，macOS 默认 `cpu`，Linux 默认 `cuda:0`。CUDA 不可用、模型缺失或 Rust 动态库缺失时直接报告错误，不改用其他后端。`R2T2_CPU_THREADS` 必须为正整数，默认 8，应结合目标设备的核心数和实际延迟调整。实时识别能否跟上音频输入、离线速度及量化后的质量须参考对应设备的实验结果，不以 CUDA 测试替代。

## 显存与并发

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `ASR_PORT` | `4174` | 宿主机 HTTP 端口 |
| `ASR_GPU` | `0` | 宿主机 GPU 编号 |
| `API_KEY` | 空 | 公共接口 Bearer 鉴权 |
| `R2T2_INTERNAL_TOKEN` | 空 | 容器内实时与离线接口鉴权 |
| `R2T2_GPU_MEMORY_UTILIZATION` | `0.30` | 共享 R2T2 显存比例 |
| `FORCED_ALIGNER_GPU_MEMORY_UTILIZATION` | `0.15` | 强制对齐显存比例 |
| `R2T2_MAX_MODEL_LEN` | `16384` | 共享引擎上下文长度上限 |
| `R2T2_MAX_SESSIONS` | `4` | 实时会话上限 |
| `R2T2_ENFORCE_EAGER` | `0` | 共享引擎禁用 CUDA graph 开关 |

两个显存比例都相对于整张 GPU；需要为 Nemotron FP32、CUDA 上下文和其他进程留余量。共享引擎只加载一份 R2T2 权重，额外请求仍占用 KV 缓存和工作空间；强制对齐模型继续独立加载。比例总和小于 1 并不保证可运行，必须按目标显卡进行加载、长音频和并发验收。

实时请求使用较高调度优先级，离线每次只提交一个最长 60 秒的片段，调度序列上限为实时会话数加一。实时仍使用最多 16 秒的滚动音频窗口。共享 GPU 上的离线音频编码和强制对齐仍会竞争计算资源，调度优先级不构成实时延迟保证。

## 模型缓存

启动前运行准备脚本下载全部模型，持久化目录如下：

- `models/huggingface`：固定版本 R2T2 和强制对齐模型。
- `models/modelscope`：FSMN VAD、CT-Transformer 标点模型。
- `models/nemotron-3-diarization`：固定 revision 的 Nemotron，容器中只读挂载到 `/app/models/nemotron-3-diarization`。原生进程可通过 `NEMOTRON_MODEL_PATH` 指定目录。
- `.cache/vllm`：vLLM 编译缓存。

联网的 Linux 开发环境可提前下载或导出：

```bash
uv sync --frozen
./scripts/prepare-models.sh
./scripts/prepare-models.sh --export-dir /tmp/r2t2-models
```

导出目录包含 `huggingface/`、`modelscope/` 和 `nemotron-3-diarization/`；复制为部署目录下的 `models/`，再设置 `HF_HUB_OFFLINE=1` 启动。离线模式缺少任何必需模型时启动失败，不会改用其他模型。

Nemotron revision 固定为 `f667ed73aee57d40cc39428eb768b4fd87a0a29e`，模型使用官方离线配置和 FP32，支持最多 8 个说话人。它输出说话人活跃时间区间，不分离干净音轨；重叠发言的文字仍受单路 ASR 能力限制。

Linux 依赖固定为 vLLM `0.30.0`、PyTorch `2.13.0+cu130`、torchaudio `2.11.0+cu130`、torchvision `0.28.0+cu130`、NCCL `2.29.7`。Transformers 使用 `5.18.0.dev0` 的已验证源码提交 `27166ea03f12c940f23176a904ab1d2ff1a3dcbb`，不是浮动开发版。macOS 保留 PyTorch/torchaudio `2.10.0` 与 torchvision `0.25.0`。所有模型共用一个 Python 环境；Nemotron 使用原生 Transformers，CUDA ASR 与强制对齐仍使用 vLLM。镜像预装匹配版本的 FlashInfer cubin，避免运行时下载内核；需要的 JIT 编译使用镜像自带的 nvcc。镜像安装时清除 `UV_OVERRIDE`，通过 `uv pip check` 检查依赖一致性。

`HF_ENDPOINT` 可配置 Hugging Face 镜像。两种识别模式使用相同 R2T2 revision：`185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`。

## 反向代理

浏览器录音需要 HTTPS 或 localhost；TLS 由外部反向代理提供。代理应允许 WebSocket Upgrade，并为离线长音频请求设置足够的请求体大小和超时时间。容器内没有额外的 nginx 或多 GPU 自动复制层。

## 验收

```bash
docker compose ps
curl http://localhost:4174/stream/v1/asr/health
curl http://localhost:4174/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=confucius4-r2t2 \
  -F word_timestamps=true \
  -F response_format=verbose_json
```

开启鉴权时添加 `Authorization: Bearer ...`。对同一份真实录音检查文本、说话人分配、时间戳以及同时运行实时会话时的延迟；不应把实时文字作为离线识别输入。

CPU 默认单会话、8 个 Rust 线程和 640ms 解码间隔，可分别通过 `R2T2_MAX_SESSIONS`、`R2T2_CPU_THREADS`、`R2T2_CHUNK_SECONDS` 调整。输入帧仍不得超过 1 秒；解码间隔与输入帧长度独立。CPU 同一时刻只执行一次原生推理，离线片段不可中途抢占；混合负载的实时延迟需单独验收。详见 [CPU PoC](../experiments/r2t2_cpu/README.md)。
