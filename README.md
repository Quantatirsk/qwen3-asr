# Qwen3-ASR for Ascend 910B

Offline speech transcription service specialized for Huawei Ascend 910B. The deployment has two processes:

- `api`: FastAPI plus CPU audio decoding, FSMN VAD, CAM++ speaker diarization, and ITN.
- `qwen-npu`: pinned vLLM Ascend image serving `Qwen/Qwen3-ASR-1.7B`.

The repository intentionally contains no realtime WebSocket, Paraformer, punctuation-model, local CUDA, or Rust Qwen runtime.

## Capabilities

- Alibaba-compatible `POST /stream/v1/asr`
- OpenAI-compatible `POST /v1/audio/transcriptions`
- long-audio segmentation with FSMN VAD
- optional CAM++ speaker diarization
- JSON, text, SRT, and VTT responses
- optional estimated word timestamps

When `word_timestamps=true`, tokens are distributed uniformly inside each effective VAD/diarization segment. Responses include `word_timestamp_method: "uniform_fallback"`; these values are compatibility estimates, not acoustic forced alignment.

## Start

Prepare the pinned Qwen model and CPU support models:

```bash
uv sync --frozen
./scripts/prepare-models.sh
```

On an Ascend host with the matching driver, firmware, CANN, and container runtime:

```bash
npu-smi info
docker compose config
docker compose build
docker compose up -d
docker compose logs -f qwen-npu api
```

The API listens on port `17003` by default. See [Ascend deployment](docs/deployment-ascend.md) for the compatibility boundary and acceptance checks.

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
