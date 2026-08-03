# Ascend 910B 单镜像部署

## 交付边界

客户平台只启动一个镜像，并向维护者提供容器内的交互式 shell。本镜像因此遵循以下约束：

- 默认命令是 `/bin/bash`，镜像启动时不自动拉起服务；
- vLLM 和 API 位于同一容器，但使用两个隔离的 Python 环境；
- Ascend vLLM 使用基础镜像自带的 `torch_npu` 软件栈；
- API 使用 `/opt/qwen3-asr-venv` 中的 CPU PyTorch，承载 FSMN VAD、CAM++ 和 ITN；
- vLLM 只监听 `127.0.0.1:17004`，API 对外监听 `0.0.0.0:17003`；
- Qwen 权重由客户平台预置，FSMN/CAM++ 辅助模型已经固化在镜像中；
- 不依赖 Compose、第二个容器、容器 DNS 或目标机联网下载。

本分支不包含实时 WebSocket、Paraformer、PUNC、本地 CUDA/Rust Qwen 或 Qwen3-ASR-0.6B。

## 制品构建

在可访问软件源和 ModelScope 的构建环境中生成唯一交付镜像：

```bash
docker build -f Dockerfile.ascend -t qwen3-asr:ascend-910b .
docker save qwen3-asr:ascend-910b -o qwen3-asr-ascend-910b.tar
```

构建过程会下载并校验 FSMN VAD 与 CAM++ 资产，但不会下载或嵌入 Qwen3-ASR-1.7B。基础镜像、CANN、`torch_npu`、vLLM 和目标服务器 driver/firmware 必须属于同一条兼容版本线。

## 平台前置条件

这些配置必须由客户容器平台完成，维护者进入 shell 后无法补救：

- 将需要使用的 `/dev/davinci*`、`/dev/davinci_manager`、`/dev/devmm_svm` 和 `/dev/hisi_hdc` 注入容器；
- 挂载宿主 DCMI、`npu-smi`、driver 库、driver 版本文件和 Ascend 安装信息；
- 为容器分配至少 `1 GiB` 共享内存；
- 让容器内可以读取平台预置的 Qwen 模型缓存；
- 对外放通 API 端口 `17003`，无需放通 vLLM 端口 `17004`。

## 维护者启动流程

### 1. 检查硬件和模型源

```bash
npu-smi info
ls -lah /root/.cache/huggingface/hub
```

标准 Hugging Face snapshot 的默认源路径为：

```text
/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
```

客户平台使用其他缓存布局时，直接把实际目录作为第一个参数传入。

### 2. 暂存 Qwen 模型

```bash
SRC=/root/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5
DST=/workspace/hf_models/Qwen3-ASR-1.7B
/workspace/qwen3-asr/scripts/stage-qwen-model.sh "$SRC" "$DST"
```

脚本会显示复制字节数、进度和速率。它使用临时目录完成复制，解引用 Hugging Face cache symlink，校验 `config.json` 与 safetensors 权重后再原子发布到目标路径。目标已完整存在时重复执行会直接跳过。

### 3. 启动全部服务

单卡默认配置：

```bash
export API_KEY=replace-me
export QWEN_ASCEND_TENSOR_PARALLEL_SIZE=1
/workspace/qwen3-asr/scripts/start-ascend-services.sh
```

多卡容器必须先确认平台已经把对应设备全部注入，再调整张量并行数：

```bash
export QWEN_ASCEND_TENSOR_PARALLEL_SIZE=8
/workspace/qwen3-asr/scripts/start-ascend-services.sh
```

额外 vLLM 参数可以直接附在启动命令后：

```bash
/workspace/qwen3-asr/scripts/start-ascend-services.sh \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8
```

脚本首先运行 `vllm serve`，等待 `/health` 成功，再启动 API。两个进程由当前前台 shell 共同托管；按 `Ctrl+C` 会一起退出。

## 关键配置

| 变量 | 默认值 | 说明 |
|---|---|---|
| `QWEN_MODEL_SOURCE` | 固定 revision 的标准 HF snapshot | 模型暂存源目录 |
| `QWEN_ASCEND_MODEL_PATH` | `/workspace/hf_models/Qwen3-ASR-1.7B` | vLLM 本地模型目录 |
| `QWEN_ASCEND_TENSOR_PARALLEL_SIZE` | `1` | vLLM 张量并行 NPU 数 |
| `QWEN_ASCEND_MAX_MODEL_LEN` | `4096` | vLLM 最大上下文 |
| `QWEN_ASCEND_MEMORY_UTILIZATION` | `0.9` | vLLM NPU 内存预算 |
| `QWEN_VLLM_STARTUP_TIMEOUT_SEC` | `600` | vLLM 健康检查等待上限 |
| `QWEN_VLLM_TIMEOUT_SEC` | `3600` | API 推理请求超时 |
| `QWEN3_ASR_API_PORT` | `17003` | 对外 API 端口 |
| `QWEN_VLLM_PORT` | `17004` | 容器内 vLLM 端口 |

## 健康检查

启动日志出现 `vLLM is ready.` 后，在容器内执行：

```bash
curl --fail http://127.0.0.1:17004/health
curl --fail http://127.0.0.1:17003/health
curl --fail http://127.0.0.1:17003/v1/models
```

日志目录：

```text
/workspace/qwen3-asr/logs/vllm.log
/workspace/qwen3-asr/logs/qwen3-asr.log
```

## 词级时间戳

Ascend 运行时不加载 Qwen3 Forced Aligner。前端传入 `word_timestamps=true` 时，API 在每个 VAD 或说话人片段内均匀分配文本单元时间，并返回 `word_timestamp_method: "uniform_fallback"`。结果只用于接口兼容，不应用作字幕精对齐或声学分析依据。

## 验收

1. 容器启动后只出现 shell，没有自动服务进程。
2. `npu-smi info` 正常，暂存模型包含完整、非悬空的权重文件。
3. 启动脚本先报告 vLLM ready，再报告 API 启动。
4. 目标机只运行一个容器，只有端口 `17003` 对外开放。
5. 短音频 REST 与 OpenAI API 返回正确文本。
6. 长音频输出连续且有序的片段时间戳。
7. 说话人分离开关两种模式均通过。
8. `word_timestamps=true` 返回单调、片段内 token 和 fallback 标识。
9. 静音、无效音频、NPU 不可用和超时返回明确错误。
10. 使用业务金标语料验证 CER/WER、RTF、并发和稳定性。
