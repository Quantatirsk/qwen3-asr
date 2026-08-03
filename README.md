# Qwen3-ASR for Ascend 910B

Offline speech transcription service packaged as one manually operated Ascend container.

The single image contains two isolated Python environments:

- the base image environment runs Ascend vLLM and `Qwen/Qwen3-ASR-1.7B`;
- `/opt/qwen3-asr-venv` runs FastAPI, audio decoding, FSMN VAD, CAM++ speaker diarization, and ITN on CPU.

Both processes run in the same container and communicate over `127.0.0.1`. The image starts in `/bin/bash`; it never starts a service automatically. The repository contains no realtime WebSocket, Paraformer, punctuation-model, local CUDA, or Rust Qwen runtime.

## Capabilities

- Alibaba-compatible `POST /stream/v1/asr`
- OpenAI-compatible `POST /v1/audio/transcriptions`
- long-audio segmentation with FSMN VAD
- optional CAM++ speaker diarization
- JSON, text, SRT, and VTT responses
- estimated compatibility word timestamps

When `word_timestamps=true`, tokens are distributed uniformly inside each effective VAD or diarization segment. Responses include `word_timestamp_method: "uniform_fallback"`; these values are not acoustic forced alignment.

## Image Build

The image build downloads and embeds the CPU support models. Qwen weights remain outside the image and are supplied by the customer platform.

```bash
docker build -f Dockerfile.ascend -t qwen3-asr:ascend-910b .
```

## Manual Container Start

After the customer platform starts the image and opens its shell, stage the preloaded Qwen snapshot:

```bash
SRC=/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
/workspace/qwen3-asr/scripts/stage-qwen-model.sh "$SRC"
```

Then start vLLM and the API in the foreground:

```bash
export API_KEY=replace-me
export QWEN_ASCEND_TENSOR_PARALLEL_SIZE=1
/workspace/qwen3-asr/scripts/start-ascend-services.sh
```

The public API listens on `0.0.0.0:17003`. Ascend vLLM listens only on `127.0.0.1:17004`. See [Ascend deployment](docs/deployment-ascend.md) for the complete operating procedure.

## Example

```bash
curl -X POST "http://localhost:17003/v1/audio/transcriptions" \
  -H "Authorization: Bearer ${API_KEY}" \
  -F "file=@audio.wav" \
  -F "model=qwen3-asr-1.7b" \
  -F "response_format=verbose_json" \
  -F "word_timestamps=true"
```

Only `qwen3-asr-1.7b` is a valid model ID in this branch.
