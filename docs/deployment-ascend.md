# Ascend 910B 部署

## 支持边界

当前适配采用拆分运行时：

- `api` 使用 CPU PyTorch，负责 FastAPI、音频处理、Paraformer、CAM++ 和聚类。
- `qwen-npu` 使用官方 vLLM Ascend 镜像，独占一张 64 GB Ascend 910B，负责 Qwen3-ASR-1.7B 离线识别。
- API 通过 vLLM OpenAI-compatible Transcriptions API 调用 NPU 服务。

首阶段不声明以下能力已经通过 910B 验证：

- Qwen WebSocket 实时增量识别；
- Qwen3 Forced Aligner 和词级时间戳；
- Qwen context/hotword hints；
- Paraformer、VAD、PUNC 或 CAM++ 在 NPU 上运行。

请求 `word_timestamps=true` 时会显式返回错误，不会静默生成伪时间戳。Qwen 实时能力也不会出现在模型能力声明中。

## 前置条件

开始部署前必须从服务器供应方取得完整兼容 BOM：

- Atlas 产品型号、910B revision、每卡 HBM 和 CPU 架构；
- 宿主 OS、内核、driver 和 firmware；
- 与所选 vLLM Ascend 镜像匹配的商用 CANN/HDK 支持包；
- `/dev/davinci0`、管理设备、DCMI 和 driver 文件可由容器读取。

默认 PoC 镜像固定为 `v0.22.1rc1` 对应的多架构 manifest digest。生产环境仍需记录客户实际拉取的平台镜像 digest，并使用客户取得的商用软件版本重新闭合兼容矩阵。

## 模型准备

默认部署使用离线缓存。将 Qwen 和 ModelScope 模型预置到以下目录：

```text
models/
  huggingface/
  modelscope/
```

Qwen 容器内模型路径可通过 `QWEN_ASCEND_MODEL_PATH` 覆盖。若使用 Hugging Face 缓存 ID，保持默认值：

```dotenv
QWEN_ASCEND_MODEL_PATH=Qwen/Qwen3-ASR-1.7B
QWEN_ASCEND_MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
HF_HUB_OFFLINE=1
```

联网准备机可按固定 revision 下载 Qwen 权重；CPU 辅助模型继续使用项目现有模型准备流程：

```bash
QWEN_VLLM_BASE_URL=http://qwen-npu:8000 \
  uv run --project environments/cpu python -m app.utils.download_models \
  --export-dir models
uv run --project environments/cpu hf download Qwen/Qwen3-ASR-1.7B \
  --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --cache-dir models/huggingface/hub
```

## 启动

先确认宿主驱动正常：

```bash
npu-smi info
docker compose -f docker-compose-ascend.yml config
docker compose -f docker-compose-ascend.yml build
docker compose -f docker-compose-ascend.yml up -d
docker compose -f docker-compose-ascend.yml logs -f qwen-npu api
```

默认公开端口为 `17003`。vLLM 服务只暴露在 Compose 内部网络，不直接映射到宿主。

## 配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLLM_ASCEND_IMAGE` | `quay.io/ascend/vllm-ascend@sha256:9008...` | `v0.22.1rc1` 多架构 manifest digest |
| `QWEN_ASCEND_MODEL_PATH` | `Qwen/Qwen3-ASR-1.7B` | 容器可读取的模型 ID 或本地路径 |
| `QWEN_ASCEND_MODEL_REVISION` | `7278e1e...` | 固定的 Hugging Face 模型 revision |
| `QWEN_ASCEND_MAX_MODEL_LEN` | `4096` | 首轮按官方保守值启动 |
| `QWEN_ASCEND_MEMORY_UTILIZATION` | `0.9` | vLLM 设备内存预算 |
| `QWEN_VLLM_TIMEOUT_SEC` | `3600` | API 调用 NPU 服务的超时 |
| `SPEAKER_DIARIZATION_DEVICE` | `cpu` | 避免 ModelScope pipeline 接收不支持的 `npu` 设备 |

## 验收顺序

1. `qwen-npu` 健康检查通过，容器内可执行 `npu-smi info`。
2. 使用官方示例音频验证 vLLM Transcriptions API。
3. 验证项目 OpenAI transcription API，且 `enable_speaker_diarization=false`。
4. 打开 CPU CAM++，验证多说话人结果。
5. 使用业务金标语料比较 CUDA 与 NPU 的 CER/WER、RTF、延迟、吞吐和 HBM。
6. 完成并发与稳定性门槛后，再评估 Forced Aligner、实时链路和 FunASR NPU worker。

完整研究、风险和验收指标见 [Ascend 910B 可行性研究](./research/ascend-910b-feasibility.md)。
