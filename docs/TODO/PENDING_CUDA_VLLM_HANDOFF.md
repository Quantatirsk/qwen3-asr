# CUDA Qwen3-ASR Official vLLM Migration Handoff

## 目的

这份文档用于交接本轮 **CUDA Qwen3-ASR -> 官方 vLLM** 的迁移进度。

当前目标不是宣称迁移已经在 NVIDIA 服务器上完成验证，而是：

- 记录本轮已经落地的代码改动
- 明确当前待验证范围
- 为后续在 NVIDIA 服务器上的联调与性能测试提供 checklist

## 本轮已完成

### 1. CUDA Qwen 适配层已切到官方 vLLM

新增文件：

- [app/services/asr/qwen3_vllm.py](/Users/quant/Documents/funasr-api/app/services/asr/qwen3_vllm.py)

职责：

- 使用官方 `vllm` 做离线 Qwen3-ASR 推理
- 使用官方 `vllm` 做 forced aligner `encode(..., pooling_task="token_classify")`
- 在项目内保留一层很薄的 streaming state adapter

当前这层已经覆盖：

- `transcribe_text(...)`
- `transcribe_raw(...)`
- `transcribe_batch(...)`
- `align_transcript(...)`
- `init_streaming_state(...)`
- `feed_stream(...)`
- `finish_stream(...)`

### 2. Qwen3 CUDA engine 已切换到新适配层

文件：

- [app/services/asr/qwen3_engine.py](/Users/quant/Documents/funasr-api/app/services/asr/qwen3_engine.py)

当前后端语义：

- `CUDA -> official vLLM`
- `MPS -> MLX`
- `CPU -> vendored QwenASR Rust`

这次移除了 CUDA 下对 `qwen-asr[vllm]` 的直接依赖。

### 3. CUDA 路径从“独占串行”改成“共享 runtime + 容量控制”

文件：

- [app/services/asr/runtime/router.py](/Users/quant/Documents/funasr-api/app/services/asr/runtime/router.py)

当前行为：

- `QWEN_VLLM` 不再走 `LocalEnginePool(size=1)` 的独占 lease
- 改成单个共享 engine
- 通过 `asyncio.Semaphore` 做容量控制

当前容量上限是硬编码：

- `8`

这意味着：

- CUDA 离线请求不再服务层串行排队
- CUDA Qwen WebSocket 连接也不再独占唯一 engine
- vLLM 现在有机会在共享 runtime 内部做请求调度

### 4. 依赖和文档已同步

文件：

- [pyproject.toml](/Users/quant/Documents/funasr-api/pyproject.toml)
- [README.md](/Users/quant/Documents/funasr-api/README.md)
- [docs/README_zh.md](/Users/quant/Documents/funasr-api/docs/README_zh.md)
- [docs/deployment.md](/Users/quant/Documents/funasr-api/docs/deployment.md)
- [app/api/v1/asr.py](/Users/quant/Documents/funasr-api/app/api/v1/asr.py)

当前说明已经更新为：

- CUDA 使用官方 `vLLM` nightly
- `word_timestamps=true` 在 CUDA vLLM / CPU Rust / Apple MLX 下都会自动调用 forced aligner

## 本轮已完成的静态验证

本轮已实际执行：

```bash
python -m compileall app
uvx pyright
```

结果：

- `compileall` 通过
- `pyright` 通过

## 当前仍未完成的验证

以下内容 **尚未在真实 NVIDIA 服务器上验证**：

### 1. 官方 vLLM nightly 依赖可安装性

待确认：

- 服务器上的 Python / CUDA / torch / driver 版本
- `vllm[audio]` nightly 是否能稳定安装
- 是否需要额外 pin 某个具体 commit / nightly 版本

### 2. CUDA 离线推理可用性

待确认：

- `qwen3-asr-0.6b`
- `qwen3-asr-1.7b`

至少验证：

- `/stream/v1/asr`
- `/v1/audio/transcriptions`

### 3. CUDA forced align 可用性

待确认：

- `word_timestamps=true`
- `Qwen/Qwen3-ForcedAligner-0.6B`

重点验证：

- 英文音频
- 中文音频
- 字词级时间戳是否落在合理范围

### 4. CUDA realtime streaming 可用性

当前 websocket 仍然依赖：

- [app/api/v1/websocket_asr.py](/Users/quant/Documents/funasr-api/app/api/v1/websocket_asr.py)
- [app/services/asr/qwen3_engine.py](/Users/quant/Documents/funasr-api/app/services/asr/qwen3_engine.py)
- [app/services/asr/qwen3_vllm.py](/Users/quant/Documents/funasr-api/app/services/asr/qwen3_vllm.py)

待确认：

- `/ws/v1/asr/qwen`
- `start -> bytes -> stop` 全链路
- partial / final 语义是否稳定
- 长连接下共享 runtime 是否会互相污染

### 5. CUDA 并发收益是否真实成立

虽然服务层已经改成 shared runtime，但收益仍需实测。

待确认：

- 单请求延迟
- 2 路并发
- 4 路并发
- 8 路并发
- 离线 REST
- Qwen websocket realtime

## NVIDIA 服务器验证 checklist

建议按这个顺序验证：

### 第一步：环境与依赖

```bash
python -c "import torch; print(torch.__version__)"
python -c "import vllm; print(vllm.__version__)"
nvidia-smi
```

确认：

- CUDA 可用
- `vllm` 可导入
- GPU 显存大小与预期一致

### 第二步：服务启动

建议先单卡启动：

```bash
DEVICE=cuda:0 uv run python start.py
```

观察日志里是否出现：

- `Loading Qwen3-ASR (official vLLM)`

### 第三步：离线识别

验证：

- `/stream/v1/asr`
- `/v1/audio/transcriptions`

重点检查：

- 英文音频
- 中文音频
- 长音频
- `word_timestamps=false`
- `word_timestamps=true`

### 第四步：Qwen websocket realtime

验证：

- `/ws/v1/asr/qwen`

重点检查：

- partial 文本是否稳定累计
- final 文本是否正确
- 多连接同时进行时是否串话

### 第五步：并发 benchmark

建议至少测：

- 1 / 2 / 4 / 8 路
- 长音频
- 离线路径
- websocket 路径

重点观察：

- 总耗时
- 单请求耗时
- GPU 利用率
- 显存占用
- 是否出现 OOM / CUDA graph / scheduler 异常

## 预计最可能出现的问题

### 1. vLLM nightly 版本漂移

这是当前最大的外部风险。

如果服务器上出现：

- 安装失败
- runtime 行为变化
- multimodal/audio API 变动

优先策略：

- 固定到一个可工作的 nightly commit
- 不要继续依赖“永远最新”

### 2. streaming 行为和 `qwen-asr[vllm]` 不完全一致

这次我们把 streaming 语义收回到了项目内。

风险点：

- partial 文本回滚策略
- context/hotwords 生效方式
- language 字段解析

如果出现差异，优先修：

- [app/services/asr/qwen3_vllm.py](/Users/quant/Documents/funasr-api/app/services/asr/qwen3_vllm.py)

### 3. shared runtime 并发上限不一定等于 8

当前 `8` 只是一个保守起点，不是验证后的最优值。

如果服务器显存较小或请求较长，可能需要：

- 下调到 `4`
- 或做成正式配置项

## 建议的后续处理顺序

如果 NVIDIA 服务器验证通过，建议下一步按这个顺序继续：

1. 固定可用的 `vllm` nightly 版本
2. 做一轮 CUDA 离线 benchmark
3. 做一轮 CUDA websocket benchmark
4. 根据结果决定是否把 shared runtime 容量从硬编码改成配置项
5. 最后再清理 CUDA 路径里可能残留的旧 wording / 兼容注释

## 当前结论

当前状态可以理解为：

- **代码改造已完成到“可联调”阶段**
- **静态检查已通过**
- **真实 NVIDIA 运行验证尚未完成**

因此这份迁移目前应视为：

- **pending handoff**
- 而不是“已经生产验证完成”
