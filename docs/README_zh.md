# Qwen3-ASR Ascend 910B 离线服务

本分支交付为一个人工运维的 Ascend 镜像。客户平台启动镜像后只进入 `/bin/bash`，不会自动启动任何服务。

同一容器内包含两个隔离的 Python 环境：

- 基础镜像环境运行 Ascend vLLM 和 `Qwen/Qwen3-ASR-1.7B`；
- `/opt/qwen3-asr-venv` 在 CPU 上运行 FastAPI、音频解码、FSMN VAD、CAM++ 说话人分离和 ITN。

两个进程只通过 `127.0.0.1` 通信。实时 WebSocket、Paraformer、PUNC、本地 CUDA/Rust Qwen 和 0.6B 模型均不属于本分支。

## 离线能力

- `POST /stream/v1/asr`
- `POST /v1/audio/transcriptions`
- 长音频自动切分
- 可选说话人分离
- JSON、文本、SRT、VTT
- 兼容型词级时间戳

传入 `word_timestamps=true` 时，服务会在每个有效 VAD 或说话人片段中按文本单元均匀分配时间，并返回 `word_timestamp_method: "uniform_fallback"`。这些时间戳不代表 forced aligner 的声学对齐精度。

## 人工启动

客户平台进入镜像 shell 后，先暂存平台预置的 Qwen 模型：

```bash
SRC=/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
/workspace/qwen3-asr/scripts/stage-qwen-model.sh "$SRC"
```

再启动同容器内的 vLLM 与 API：

```bash
export API_KEY=replace-me
export QWEN_ASCEND_TENSOR_PARALLEL_SIZE=1
/workspace/qwen3-asr/scripts/start-ascend-services.sh
```

API 默认监听 `0.0.0.0:17003`，vLLM 仅监听 `127.0.0.1:17004`。完整要求见 [Ascend 910B 部署说明](deployment-ascend.md)。
