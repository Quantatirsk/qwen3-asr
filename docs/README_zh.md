# Qwen3-ASR Ascend 910B 离线服务

本分支只保留华为 Ascend 910B 部署需要的离线链路：

- NPU 容器通过 vLLM Ascend 运行 `Qwen/Qwen3-ASR-1.7B`；
- API 容器在 CPU 上完成音频解码、FSMN VAD、CAM++ 说话人分离和 ITN；
- API 容器通过 OpenAI-compatible Transcriptions API 调用 NPU 容器。

实时 WebSocket、Paraformer、PUNC、本地 CUDA Qwen、Rust Qwen 和 0.6B 模型均不属于本分支。

## 离线能力

- `POST /stream/v1/asr`
- `POST /v1/audio/transcriptions`
- 长音频自动切分
- 可选说话人分离
- JSON、文本、SRT、VTT
- 兼容型词级时间戳

传入 `word_timestamps=true` 时，服务会在每个有效 VAD/说话人片段中按文本单元均匀分配时间，并返回：

```json
{
  "word_timestamp_method": "uniform_fallback"
}
```

这些时间戳用于保持前端协议兼容，不代表 forced aligner 的声学对齐精度。

## 启动

```bash
uv sync --frozen
./scripts/prepare-models.sh
npu-smi info
docker compose config
docker compose build
docker compose up -d
```

默认公开端口为 `17003`，详细要求见 [Ascend 910B 部署说明](deployment-ascend.md)。
