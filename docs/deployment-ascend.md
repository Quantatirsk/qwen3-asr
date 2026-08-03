# Ascend 910B 部署

## 架构边界

部署由两个容器组成：

| 容器 | 设备 | 职责 |
|---|---|---|
| `api` | CPU | FastAPI、音频解码、FSMN VAD、CAM++、ITN、结果格式化 |
| `qwen-npu` | Ascend 910B | vLLM Ascend、Qwen3-ASR-1.7B 离线推理 |

本分支不包含实时 WebSocket、Paraformer、PUNC、本地 CUDA/Rust Qwen 或 Qwen3-ASR-0.6B。

## 词级时间戳

Ascend 运行时不加载 Qwen3 Forced Aligner。前端传入 `word_timestamps=true` 时，API 在每个 VAD 或说话人片段内均匀分配文本单元时间：

- 中文汉字各自成为一个单元；
- 英文和数字按连续单词成为一个单元；
- 标点附加到前一个单元；
- 时间为片段内绝对时间，最后一个单元严格结束于片段终点；
- 响应包含 `word_timestamp_method: "uniform_fallback"`。

该结果只用于接口兼容，不应作为字幕精对齐或声学分析依据。

## 前置条件

- 已确认 Atlas 型号、910B revision、每卡 HBM 和服务器 CPU 架构；
- driver、firmware、CANN/HDK 与所选 vLLM Ascend 镜像匹配；
- 容器可访问 `/dev/davinci0`、管理设备、DCMI 和 driver 文件；
- `npu-smi info` 在宿主机正常；
- 模型缓存已预置，生产环境建议 `HF_HUB_OFFLINE=1`。

默认 vLLM Ascend 镜像和 Qwen 权重均以 digest/revision 固定。客户环境升级其中任一项时，需要重新验证兼容矩阵。

## 模型准备

```bash
uv sync --frozen
./scripts/prepare-models.sh
```

导出目录结构：

```text
models/
  huggingface/   # Qwen3-ASR-1.7B
  modelscope/    # FSMN VAD + CAM++
```

默认 Qwen revision：

```text
7278e1e70fe206f11671096ffdd38061171dd6e5
```

## 启动

```bash
npu-smi info
docker compose config
docker compose build
docker compose up -d
docker compose logs -f qwen-npu api
```

默认公开地址为 `http://服务器地址:17003`。NPU vLLM 仅在 Compose 内部网络暴露。

## 关键配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `VLLM_ASCEND_IMAGE` | 固定 manifest digest | vLLM Ascend 基础镜像 |
| `QWEN_ASCEND_MODEL_PATH` | `Qwen/Qwen3-ASR-1.7B` | 模型 ID 或容器内路径 |
| `QWEN_ASCEND_MODEL_REVISION` | `7278e1e...` | 固定权重 revision |
| `QWEN_ASCEND_MAX_MODEL_LEN` | `4096` | vLLM 最大上下文 |
| `QWEN_ASCEND_MEMORY_UTILIZATION` | `0.9` | NPU 内存预算 |
| `QWEN_VLLM_TIMEOUT_SEC` | `3600` | API 到 vLLM 的请求超时 |

## 验收

1. `qwen-npu` 和 `api` 健康检查通过。
2. 短音频 REST 与 OpenAI API 返回正确文本。
3. 长音频输出连续且有序的片段时间戳。
4. 说话人分离开关两种模式均通过。
5. `word_timestamps=true` 返回单调、片段内的 token 和 fallback 标识。
6. JSON、text、SRT、VTT 格式通过。
7. 静音、无效音频、NPU 不可用和超时返回明确错误。
8. 使用业务金标语料验证 CER/WER、RTF、并发和稳定性。

研究依据和仍需客户 BOM 闭合的风险见 [Ascend 910B 可行性研究](research/ascend-910b-feasibility.md)。
