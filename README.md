<div align="center">

<h3>Ready-to-use Local Speech Recognition API Service</h3>

Speech recognition API service powered by [FunASR](https://github.com/alibaba-damo-academy/FunASR) and [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), supporting 52 languages, compatible with OpenAI API and Alibaba Cloud Speech API.

[简体中文](./docs/README_zh.md)

---

![Static Badge](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Static Badge](https://img.shields.io/badge/Torch-2.9.0-%23EE4C2C?logo=pytorch&logoColor=white)
![Static Badge](https://img.shields.io/badge/CUDA-12.6+-%2376B900?logo=nvidia&logoColor=white)

</div>

## Live Demo Site

- **Web Demo**: https://asr.vect.one

## Demo

[![Demo](./demo/demo.png)](https://media.cdn.vect.one/qwenasr_client_demo.mp4)

## Release 1.0

> `v1.0.0` is a large breaking refactor relative to the current `main` branch.
> If you are upgrading from `main`, read the release notes before reusing old deployment assumptions.
>
> Key breaking changes:
> - Python dependency management is now `uv`-based (`pyproject.toml` + `uv.lock`); `requirements*.txt` are gone
> - Runtime stack changed to `CUDA -> official vLLM`, `CPU/macOS -> vendored QwenASR Rust`
> - `MLX` / Apple Silicon GPU path has been removed; `mps` is normalized to `cpu`
> - macOS / Apple Silicon now defaults to `qwen3-asr-0.6b` unless the caller explicitly selects `qwen3-asr-1.7b`
> - Offline `model` / `model_id` are now compatibility-only parameters and no longer switch the active offline model
> - `ENABLED_MODELS` has been removed

## Features

- **Hybrid Runtime Stack** - Uses auto-selected Qwen3-ASR for offline inference and Paraformer realtime for websocket streaming
- **Speaker Diarization** - Automatic multi-speaker identification using CAM++ model
- **OpenAI API Compatible** - Supports `/v1/audio/transcriptions` endpoint, works with OpenAI SDK
- **Alibaba Cloud API Compatible** - Supports Alibaba Cloud Speech RESTful API and WebSocket streaming protocol
- **WebSocket Streaming** - Real-time streaming speech recognition with low latency
- **Smart Far-Field Filtering** - Automatically filters far-field sounds and ambient noise in streaming ASR
- **Intelligent Audio Segmentation** - VAD-based greedy merge algorithm for automatic long audio splitting
- **GPU Batch Processing** - Batch inference support, 2-3x faster than sequential processing
- **Resource-Aware Runtime** - Auto-selects the appropriate Qwen3-ASR model for the current machine

## Acknowledgements

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) provides the official model family and multimodal/vLLM usage guidance
- [QwenASR](https://github.com/huanglizhuo/QwenASR) provides the CPU Rust backend vendored by this project

## Quick Deployment

### 1. Docker Deployment (Recommended)

```bash
# Copy and edit configuration
cp .env.example .env
# Edit .env to set API_KEY (optional)

# Start service (GPU version)
docker-compose up -d

# Or CPU version
docker-compose -f docker-compose-cpu.yml up -d

# Multi-GPU auto mode (one instance per visible GPU)
CUDA_VISIBLE_DEVICES=0,1,2,3 docker-compose up -d
```

Service URLs:
- **API Endpoint**: `http://localhost:17003`
- **API Docs**: `http://localhost:17003/docs`

Optional built-in rate limit settings:
- `NGINX_RATE_LIMIT_RPS` (global requests/sec, `0` = disabled)
- `NGINX_RATE_LIMIT_BURST` (global burst, `0` = auto use RPS)

**docker run (alternative):**

```bash
# GPU version
docker run -d --name funasr-api \
  --gpus all \
  -p 17003:8000 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e API_KEY=your_api_key \
  -v ./models/modelscope:/root/.cache/modelscope \
  -v ./models/huggingface:/root/.cache/huggingface \
  quantatrisk/funasr-api:gpu-latest

# CPU version
docker run -d --name funasr-api \
  -p 17003:8000 \
  quantatrisk/funasr-api:cpu-latest
```

> **Note**: CPU images now support `qwen3-asr-0.6b` via the bundled QwenASR Rust backend.
> On CUDA vLLM and CPU Rust, `word_timestamps=true` now triggers the forced aligner automatically.
> On macOS / Apple Silicon, Qwen3-ASR now runs through the Rust CPU backend.

**Offline Deployment**: Use the helper script to prepare the current runtime model package, then copy to the offline machine:

```bash
# 1. Prepare models
./scripts/prepare-models.sh

# 2. Copy the package to offline server
scp funasr-models-*.tar.gz user@server:/opt/funasr-api/

# 3. On offline server, extract and start
tar -xzvf funasr-models-*.tar.gz
docker-compose up -d
```

> Detailed deployment instructions: [Deployment Guide](./docs/deployment.md)

### Local Development

**System Requirements:**

- Python 3.10+
- CUDA 12.6+ (optional, for GPU acceleration)
- FFmpeg (audio format conversion)

**Installation:**

Runtime dependency locks now default to the GPU stack at the repo root, with CPU kept as a specialized environment:

| Mode | Command | Notes |
|------|---------|-------|
| GPU (default) | `uv sync` | Syncs the root [pyproject.toml](/opt/funasr-api/pyproject.toml) and [uv.lock](/opt/funasr-api/uv.lock) into `.venv`, including CUDA `torch/torchaudio/torchvision` |
| CPU (specialized) | `./scripts/sync_cpu_env.sh` | Syncs the dedicated CPU lock in [environments/cpu/pyproject.toml](/opt/funasr-api/environments/cpu/pyproject.toml) into `.venv` |

```bash
# Clone project
cd FunASR-API

# Install dependencies (Linux/CUDA)
uv sync

# Start service
source .venv/bin/activate
python start.py
```

macOS / Apple Silicon local development:

```bash
./scripts/sync_cpu_env.sh
source .venv/bin/activate
python start.py
```

Startup UI:

- `FUNASR_STARTUP_UI=auto` is the default for single-worker interactive terminals
- `auto` / `tui` starts a Textual dashboard that owns the terminal, captures child stdout/stderr, shows startup phase progress on top, and streams logs in a dedicated log pane
- `plain` disables the dashboard and falls back to normal terminal output
- multi-worker mode always falls back to plain output
- Docker / `docker-compose logs -f` always use plain output because the container log stream is not an interactive TTY

## Runtime Defaults

Current runtime behavior on the mainline codebase:

- `DEVICE=auto` resolves to `cuda:0` when CUDA is available, otherwise `cpu`
- `DEVICE=mps` is normalized to `cpu`
- `Linux + CUDA` uses official `vLLM`
- `Linux + CPU` uses vendored `QwenASR` Rust
- `macOS / Apple Silicon` also uses vendored `QwenASR` Rust
- macOS / Apple Silicon defaults to `qwen3-asr-0.6b`
- `qwen3-asr-1.7b` on macOS is only used when the caller explicitly selects it
- `word_timestamps=true` works on the current offline CUDA and CPU Rust paths
- WebSocket streaming does not currently return word-level timestamps
- CAM++ speaker diarization remains required and still follows `DEVICE`; on CPU its main hotspot is speaker verification embedding

## API Endpoints

### OpenAI Compatible API

| Endpoint | Method | Function |
|----------|--------|----------|
| `/v1/audio/transcriptions` | POST | Audio transcription (OpenAI compatible) |
| `/v1/models` | GET | Offline model list |

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | Preferred when provided | Audio/video file |
| `audio_address` | string | Optional | Audio/video URL (HTTP/HTTPS). Ignored when `file` is also provided |
| `model` | string | ignored | Compatibility parameter; ignored for offline requests |
| `language` | string | Auto-detect | Language code (zh/en/ja) |
| `enable_speaker_diarization` | bool | `true` | Enable speaker diarization |
| `word_timestamps` | bool | `false` | Return word-level timestamps when the backend supports them. Qwen CUDA vLLM and CPU Rust automatically use the forced aligner when enabled. |
| `response_format` | string | `verbose_json` | Output format |
| `prompt` | string | - | Prompt text (reserved) |
| `temperature` | float | `0` | Sampling temperature (reserved) |

**Audio / Video Input Methods:**
- **File Upload**: Use `file` parameter to upload an audio file or a video container with an audio track
- **URL Download**: Use `audio_address` parameter to provide an audio/video URL, service will download automatically
- **Precedence**: If both `file` and `audio_address` are provided, the service uses `file` and ignores `audio_address`

**Usage Examples:**

```python
# Using OpenAI SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="your_api_key")

with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",  # Compatibility field; ignored by offline routing
        file=f,
        response_format="verbose_json"  # Get segments and speaker info
    )
print(transcript.text)
```

```bash
# Using curl
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@audio.wav" \
  -F "model=qwen3-asr-0.6b" \
  -F "response_format=verbose_json" \
  -F "enable_speaker_diarization=true"
```

**Supported Response Formats:** `json`, `text`, `srt`, `vtt`, `verbose_json`

### Alibaba Cloud Compatible API

| Endpoint | Method | Function |
|----------|--------|----------|
| `/stream/v1/asr` | POST | Speech recognition (long audio support) |
| `/stream/v1/asr/models` | GET | Declared model/capability entries |
| `/stream/v1/asr/health` | GET | Health check |
| `/ws/v1/asr` | WebSocket | Streaming ASR (Alibaba Cloud protocol compatible) |
| `/ws/v1/asr/funasr` | WebSocket | FunASR streaming (backward compatible) |
| `/ws/v1/asr/qwen` | WebSocket | Qwen3-ASR streaming |

**Request Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | string | ignored | Compatibility parameter; ignored for offline requests |
| `audio_address` | string | `https://media.cdn.vect.one/podcast_demo.mp4` (docs example) | Audio/video URL (optional; ignored when body content is uploaded) |
| `sample_rate` | int | `16000` | Sample rate |
| `enable_speaker_diarization` | bool | `true` | Enable speaker diarization |
| `word_timestamps` | bool | `false` | Return word-level timestamps when the backend supports them. Qwen CUDA vLLM and CPU Rust automatically use the forced aligner when enabled. |
| `vocabulary_id` | string | - | Hotwords (format: `word1 weight1 word2 weight2`) |

**Usage Examples:**

```bash
# Basic usage
curl -X POST "http://localhost:8000/stream/v1/asr" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav

# With parameters
curl -X POST "http://localhost:8000/stream/v1/asr?enable_speaker_diarization=true" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @audio.wav
```

**Response Example:**

```json
{
  "task_id": "xxx",
  "status": 200,
  "message": "SUCCESS",
  "result": "Speaker1 content...\nSpeaker2 content...",
  "duration": 60.5,
  "processing_time": 1.234,
  "segments": [
    {
      "text": "Today is a nice day.",
      "start_time": 0.0,
      "end_time": 2.5,
      "speaker_id": "Speaker1",
      "word_tokens": [
        {"text": "Today", "start_time": 0.0, "end_time": 0.5},
        {"text": "is", "start_time": 0.5, "end_time": 0.7},
        {"text": "a nice day", "start_time": 0.7, "end_time": 1.5}
      ]
    }
  ]
}
```

## Speaker Diarization

Multi-speaker automatic identification based on CAM++ model:

- **Enabled by Default** - `enable_speaker_diarization=true`
- **Automatic Detection** - No preset speaker count needed, model auto-detects
- **Speaker Labels** - Response includes `speaker_id` field (e.g., "Speaker1", "Speaker2")
- **Smart Merging** - Two-layer merge strategy to avoid isolated short segments:
  - Layer 1: Accumulate merge same-speaker segments < 10 seconds
  - Layer 2: Accumulate merge continuous segments up to 60 seconds
- **Subtitle Support** - SRT/VTT output includes speaker labels `[Speaker1] text content`

Disable speaker diarization:

```bash
# OpenAI API
-F "enable_speaker_diarization=false"

# Alibaba Cloud API
?enable_speaker_diarization=false
```

## Audio Processing

### Intelligent Segmentation Strategy

Automatic long audio segmentation:

1. **VAD Voice Detection** - Detect voice boundaries, filter silence
2. **Greedy Merge** - Accumulate voice segments, ensure each segment does not exceed `MAX_SEGMENT_SEC` (default 30s)
3. **Silence Split** - Force split when silence between voice segments exceeds 3 seconds
4. **Batch Inference** - Multi-segment parallel processing, 2-3x performance improvement in GPU mode

### WebSocket Streaming Limitations

**FunASR Model Limitations** (using `/ws/v1/asr` or `/ws/v1/asr/funasr`):
- ✅ Real-time speech recognition, low latency
- ✅ Sentence-level timestamps
- ❌ **Word-level timestamps** (not implemented on the FunASR realtime path)
- ❌ **Confidence scores** (not implemented)

**Qwen3-ASR Streaming** (using `/ws/v1/asr/qwen`):
- ✅ Multi-language real-time recognition
- ✅ CUDA vLLM and CPU Rust both support the current streaming path
- ❌ Word-level timestamps are not available in the current streaming path

### Qwen3 Runtime Matrix

| Runtime | Backend | Offline | WebSocket Streaming | Word Timestamps Offline | Word Timestamps Streaming | Maturity |
|---------|---------|---------|---------------------|-------------------------|---------------------------|----------|
| Linux + NVIDIA GPU | Official vLLM 0.19.0 | ✅ | ✅ | ✅ | ❌ | Production-oriented |
| CPU / macOS | QwenASR Rust | ✅ | ✅ | ✅ (forced aligner) | ❌ | Recommended local fallback |

## Offline-Capable Models

| Model ID | Name | Description | Features |
|----------|------|-------------|----------|
| `qwen3-asr-1.7b` | Qwen3-ASR 1.7B | High-performance multilingual ASR, 52 languages + dialects; CUDA uses vLLM | Offline/Realtime |
| `qwen3-asr-0.6b` | Qwen3-ASR 0.6B | Lightweight multilingual ASR; CUDA uses vLLM, CPU/macOS uses Rust backend | Offline/Realtime |

## Realtime-Only Capability

| Capability ID | Runtime | Description |
|---------------|---------|-------------|
| `paraformer-large` | FunASR realtime | Chinese websocket realtime stack with realtime punctuation |

**Runtime selection:**
- **VRAM >= 32GB**: Select `qwen3-asr-1.7b`
- **VRAM < 32GB**: Select `qwen3-asr-0.6b`
- **No CUDA**: Select the vendored Rust-backed `qwen3-asr-0.6b`
- **macOS / Apple Silicon**: Always default to `qwen3-asr-0.6b`, regardless of memory size, unless a caller explicitly selects `qwen3-asr-1.7b`
- `paraformer-large` realtime capability is always prepared for websocket streaming

## Environment Variables

Recommended public settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | - | API authentication key (optional, unauthenticated if not set) |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG/INFO/WARNING/ERROR) |
| `MAX_AUDIO_SIZE` | `2048` | Max audio file size (MB, supports units like 2GB) |
| `ASR_BATCH_SIZE` | `4` | ASR batch size for long-audio segment processing |
| `MAX_SEGMENT_SEC` | `30` | Max audio segment duration (seconds) |
| `ASR_ENABLE_NEARFIELD_FILTER` | `true` | Enable far-field sound filtering |

Far-field filter notes:

- `ASR_NEARFIELD_RMS_THRESHOLD=0.01` is the current default and recommended starting point
- raise it in noisy rooms to filter more background speech
- lower it in quiet rooms if soft speech is being dropped
- use `LOG_LEVEL=DEBUG` temporarily when you need to inspect filter behavior

Advanced backend-specific settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_RUST_CPU_WORKERS` | `4` | CPU Rust backend worker count (Rust ASR / forced align default to 4 runtimes) |
| `QWENASR_LIBRARY_PATH` | auto-detect | Override vendored Rust dylib/so path |

## Resource Requirements

**Minimum (CPU):**

- CPU: 4 cores
- Memory: 16GB
- Disk: 20GB

**Recommended (GPU):**

- CPU: 4 cores
- Memory: 16GB
- GPU: NVIDIA GPU (16GB+ VRAM)
- Disk: 20GB

## API Documentation

After starting the service:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Links

- **Deployment Guide**: [Detailed Docs](./docs/deployment.md)
- **FunASR**: [FunASR GitHub](https://github.com/alibaba-damo-academy/FunASR)
- **Chinese README**: [中文文档](./docs/README_zh.md)

## License

This project uses the MIT License - see [LICENSE](LICENSE) file for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Quantatirsk/funasr-api&type=Date)](https://star-history.com/#Quantatirsk/funasr-api&Date)

## Contributing

Issues and Pull Requests are welcome to improve the project!
