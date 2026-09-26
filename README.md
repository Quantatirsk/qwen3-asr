# R2T2 ASR

本分支提供基于 Confucius4-R2T2 的实时与离线语音识别服务，支持 Linux x86_64 / NVIDIA CUDA，以及 macOS Apple Silicon / Rust CPU。实时与离线共用一个 R2T2 推理实例，权重只加载一次；各请求保留独立的音频与解码状态。

离线流程为：完整录音 → CAM++ 说话人分离 → 按时间区间切段 → R2T2 重新识别 → 强制对齐 → 合并带说话人、时间戳的结果。离线识别不读取实时转写文本。保留 FSMN VAD 和 Qwen3-ForcedAligner-0.6B；强制对齐模型仅生成时间戳，不承担文字识别。

## 启动

安装 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit 后：

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f asr
```

默认使用 GPU 0，服务地址为 `http://localhost:4174`，录音页面 `/realtime`，API 文档 `/docs`。首次启动下载模型到 `models/`；后续可以设置 `HF_HUB_OFFLINE=1`。显存预算需要按硬件调整，详见 [部署说明](docs/deployment.md)。

macOS 使用原生进程，安装 Rust 工具链后：

```bash
uv sync --frozen
./scripts/build-rust.sh
DEVICE=cpu R2T2_MAX_SESSIONS=1 uv run python start.py
```

本地服务地址为 `http://localhost:8000`。macOS 默认选择 CPU，Linux 默认选择 `cuda:0`；显式选择 CUDA 但不可用时启动失败，不自动降级。CPU 线程数默认 8，可通过 `R2T2_CPU_THREADS` 调整。CPU 默认每 640ms 解码，CUDA 默认 160ms；用 `R2T2_CHUNK_SECONDS` 调整解码频率，停顿检测仍按小音频帧处理。完整对照数据和限制见 [CPU PoC](experiments/r2t2_cpu/README.md)。

## 文件转写

```bash
curl http://localhost:4174/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=confucius4-r2t2 \
  -F word_timestamps=true \
  -F response_format=verbose_json \
  -F enable_speaker_diarization=true
```

模型列表仅返回 `confucius4-r2t2`。文件转写请求中的 `model` 参数可填写任意值，服务忽略该参数并始终使用 R2T2，不按名称选择或路由模型。支持 `/v1/audio/transcriptions` 和 `/stream/v1/asr` 下的离线接口；完整参数以 `/docs` 为准。

实时接口为 `/v1/stream`，使用 16 kHz 单声道 PCM，通过 WebSocket 返回追加式文本；协议见 [实时转写](docs/realtime.md)。实时不提供说话人标签和词级时间戳。

## 开发与验证

```bash
uv sync --frozen
uv run python start.py
uv run pytest tests
```

依赖分别为 Linux CUDA 和 macOS ARM64 锁定，macOS 不安装 vLLM 或 CUDA 包。模型质量、内存及吞吐量必须在对应硬件上使用真实录音验收；本地逻辑测试不能替代模型推理验证。

## 上游

- [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2)：实时与离线识别；源码归属见 `deploy/R2T2-NOTICE`，权重遵循上游独立 MODEL_LICENSE。
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)：保留其中的 Qwen3-ForcedAligner-0.6B 强制对齐能力。
- [FunASR](https://github.com/modelscope/FunASR)：FSMN VAD 与 CAM++ 说话人模型。
