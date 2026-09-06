# VibeVoice 流式 ASR 替换可行性研究

研究日期：2026-09-06。范围：上游文档、模型元数据、实现代码与本项目接入链路；未下载权重，未进行模型推理、精度或性能实测。

- 本地版本：`3263fee409a144568e5fb04b2c2c27b9e061ebd7`。
- 上游版本：[`1541f590c7099820f10ea012f48d2399282df69f`](https://github.com/microsoft/VibeVoice/commit/1541f590c7099820f10ea012f48d2399282df69f)，提交日期 2026-09-03。
- 7B 权重版本：`60d858b518b4e19d404af3737f848fc185b30177`；1.5B 权重版本：`4262d23d8a539a6530cf64fbd0b1751ef9a30853`。

## 结论

值得做独立对照验证，但目前不建议直接替换。新模型确实接受持续到达的音频，并保留跨块语音和文本上下文；主要增益方向是多语言、热词与在线说话人归属。对当前中文实时字幕链路，最先需要验证的是能否接受约 2.93 秒的文本更新节奏，以及更大的模型资源成本。

本项目离线识别已经使用 Qwen3-ASR；Paraformer 主要承担 `/ws/v1/asr`、`/funasr` 的实时识别。因此，这次评估应针对实时链路，而不是替换全部 ASR。建议先比较现有 Paraformer、已有 Qwen3 流式路径和 VibeVoice 1.5B；只有小模型精度不够时再测 7B。1.5B 首先用于资源可行性验证，不能继承 7B 榜单结果；上游对比显示两者精度有明显差距。[论文 Table 5](https://arxiv.org/html/2609.02812v1)。

## 模型与能力边界

| 项目 | 核实结果 |
| --- | --- |
| 新发布模型 | 2026-09-03 发布 `VibeVoice-ASR-Streaming`，有 1.5B 和 7B 两种骨干规模 |
| 流式语言 | 中文、英语、法语、德语、意大利语、日语、韩语、葡萄牙语、俄语、西班牙语 |
| 模型输出 | 带 `Speaker N:` 标签的分块文本；支持 `context_info` 热词/上下文 |
| 授权标记 | 仓库与两个流式模型卡均标注 MIT |
| 离线版本 | `microsoft/VibeVoice-ASR` 是另一条路径，其 50 多语言、60 分钟能力不能直接归给新流式模型 |
| TTS 版本 | `VibeVoice-Realtime-0.5B` 是语音合成，不是本次候选 ASR |

来源：[仓库介绍](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/README.md)、[7B 模型卡](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B)、[1.5B 模型卡](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B)。

## 实际流式行为

两个发布 checkpoint 均配置为 24 kHz、每潜在帧 3200 个采样点、22 帧推进、4 帧前瞻，即每块推进 2.933 秒，读取窗口 3.467 秒。分块与前瞻是训练约定，不应通过改参数将其压缩成 240/600 毫秒。[7B 配置](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B/blob/60d858b518b4e19d404af3737f848fc185b30177/preprocessor_config.json)、[1.5B 配置](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B/blob/4262d23d8a539a6530cf64fbd0b1751ef9a30853/preprocessor_config.json)。

持续输入且没有结束信号时，第一块需先收到约 3.47 秒音频，再执行解码；后续每约 2.93 秒返回一个文本块。短语音可以发送 `end` 提前补零刷新，所以 3.47 秒不是短句最终结果的硬性等待时间。论文的 2.00 秒是稳态平均算法延迟，不是端到端首字延迟，还需加上解码、排队和网络时间。[实时输入实现](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vllm_plugin/asr_streaming_server.py)、[论文 §4、§6](https://arxiv.org/html/2609.02812v1)。

发布模型目标会话长度为 8 分钟。vLLM 文档允许增大上下文运行更长会话，但这是服务配置能力，不代表更长语音已有精度保证。模型保留历史且资源随时长增长，不能用无限连接等同无限可靠识别。[论文 §3.1、§6](https://arxiv.org/html/2609.02812v1)、[部署文档](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/docs/vibevoice-vllm-asr-streaming.md)。

说话人标签在当前会话内按首次出现分配，不是声纹注册身份。时间戳也需区分：流式模型直接输出说话人与内容；上游展示层按块边界和字符占比估算 `Start/End`，再生成 SRT，并非词级强制对齐。多个说话人持续重叠也是论文明确的限制。[分段实现](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vllm_plugin/asr_streaming.py)、[论文 §6](https://arxiv.org/html/2609.02812v1)。

## 部署与资源

| 候选 | 总参数量，包括语音组件 | BF16 权重字节量 | 用途 |
| --- | ---: | ---: | --- |
| `microsoft/VibeVoice-ASR-Streaming-1.5B` | 2,814,116,321 | 5.63 GB | 优先验证资源与基本体验 |
| `microsoft/VibeVoice-ASR-Streaming-7B` | 8,674,021,857 | 17.35 GB | 精度与说话人能力对照 |

参数量来自官方模型仓库 API，权重字节量来自 safetensors 索引，使用十进制 GB。它们不是推理最低显存要求；还需 KV cache、语音编码、激活和运行时开销。上游未给出可据此承诺的最低 GPU 显存。[1.5B 元数据](https://huggingface.co/api/models/microsoft/VibeVoice-ASR-Streaming-1.5B)、[7B 元数据](https://huggingface.co/api/models/microsoft/VibeVoice-ASR-Streaming-7B)、[1.5B 索引](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-1.5B/blob/4262d23d8a539a6530cf64fbd0b1751ef9a30853/model.safetensors.index.json)、[7B 索引](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B/blob/60d858b518b4e19d404af3737f848fc185b30177/model.safetensors.index.json)。

官方性能证据是 A100 80GB PCIe、vLLM、BF16、batch size 1，使用论文中的 15 帧配置：每块约 146–208 毫秒，RTF 约 0.073–0.104。它不能直接代表已发布 22 帧模型、其他 GPU 或多用户并发。[论文 Table 8](https://arxiv.org/html/2609.02812v1)。

部署需要注意：

- 上游 Python 包要求 Python ≥3.10、`transformers>=4.51.3,<5.0.0`；推荐 NVIDIA PyTorch 容器，vLLM 示例固定为 `vllm/vllm-openai:v0.14.1`。参考推理使用仓库自带模型类，不能把 HF 页面自动生成的导入示例当作当前环境可用性的证明。
- 本项目当前为 Torch 2.10、Transformers 4.57 系列和 vLLM 0.19.0。Transformers 范围相交，但不能据此推定 vLLM 插件跨版本兼容；首轮使用上游容器隔离依赖。
- 裸 FastAPI 示例按块用全局 `gpu_lock` 串行推理；正式多会话验证使用 vLLM 路径，其引擎支持连续批处理。Tensor Parallel 可用；Data Parallel 参数被拒绝，多实例必须按完整会话固定路由。
- 前缀 KV cache 和多模态处理缓存需要容量预算；文档的近似恒定每块成本不表示内存不增长，缓存不足还可能触发音频重发或失败。
- 独立进程不会隔离显存。本项目 Qwen3 引擎已有显存预算，第一轮应顺序运行模型或使用不同 GPU，再验证最终共存。
- 本次核实到的是 NVIDIA/CUDA 官方部署路径；CPU/MPS/XPU 仅在参考脚本中有设备选项，未获得其性能保证，也未核实昇腾 NPU 支持。

来源：[上游依赖](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/pyproject.toml)、[参考脚本](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/demo/vibevoice_asr_streaming_inference_from_file.py)、[裸 FastAPI 示例](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/demo/vibevoice_asr_streaming_fastapi_demo.py)、[vLLM 服务](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vllm_plugin/asr_streaming_server.py)、[启动器](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vllm_plugin/scripts/start_streaming_server.py)、本地 `pyproject.toml` 和 `app/services/asr/qwen3_engine.py`。

## 与本项目的接入差异

| 边界 | 当前实现 | VibeVoice 需要处理 |
| --- | --- | --- |
| 实时入口 | `/ws/v1/asr`、`/funasr`，阿里云风格事件 | 对接上游 `/v1/stream`，映射现有事件 |
| 输入 | 16 kHz，内部 3840/9600 样本块，即 240/600 ms | 24 kHz 单声道 float32 little-endian PCM，持续重采样并保持音频连续 |
| 引擎调用 | 直接调用 FunASR `realtime_model.generate`，传 `cache/is_final/chunk_size` | 每连接保留 VibeVoice 会话，上游积累音频和 KV 历史 |
| 断句 | 静音/空结果触发句末 flush，然后清空 `audio_cache` | 上游没有同会话句末 flush/reset，`end` 会结束会话 |
| 文本 | 增量拼接，中文标点与中文 ITN | 上游块内文本带说话人标签；适配增量，按语言决定后处理 |
| 说话人/时间 | 当前实时响应没有在线说话人或词级时间戳 | 需要时明确扩展协议；不能把上游估算时间当词级对齐 |
| 近场过滤 | 部分音频帧丢弃，但本地时钟继续 | 评估过滤对连续语音上下文、时间轴和说话人判断的影响 |
| 运行时管理 | 非 Qwen 模型进入 FunASR runtime，默认一个独占 worker | 新 runtime/会话生命周期；不能只改模型 ID |

输入分块长度不是现有系统实测首字延迟，上表只比较协议和调度边界。

现有模型 ID 是 `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online`。关键接入点：[模型配置](/Users/quant/Documents/qwen3-asr/app/services/asr/models.json:3)、[实时路由](/Users/quant/Documents/qwen3-asr/app/api/v1/websocket_asr.py:18)、[实时识别服务](/Users/quant/Documents/qwen3-asr/app/services/websocket_asr.py:184)、[运行时选择](/Users/quant/Documents/qwen3-asr/app/services/asr/runtime/router.py:91)、[部署计划](/Users/quant/Documents/qwen3-asr/app/services/asr/model_plan.py:115)、[模型资源声明](/Users/quant/Documents/qwen3-asr/app/services/asr/model_capabilities.py)、[中文文本处理](/Users/quant/Documents/qwen3-asr/app/utils/text_processing.py)。目前部署计划同时准备 Qwen 与 Paraformer，真实替换还需更新资源预热；FunASR 同时提供 VAD/CAM++ 等组件，移除 Paraformer 不等于可以删除 FunASR 依赖。

上游 WebSocket 的第一条消息是 JSON 配置，之后发送二进制 PCM，最后发送字符串 `end`。普通消息的 `text` 是本块增量，最终 `done` 消息的 `text` 是全文。新连接创建新会话；没有现成的在线热词更新、会话恢复或保留说话人身份的重置接口。[协议与会话源码](https://github.com/microsoft/VibeVoice/blob/1541f590c7099820f10ea012f48d2399282df69f/vllm_plugin/asr_streaming_server.py)。

本地 `/qwen` 已有另一套 `start/result/segment_end/final` 流式协议，默认 2 秒块，并在 2 秒静音或 60 秒后重置。它可作为已有的多语言候选基线，但不能宣称其精度优于 VibeVoice，也不能与阿里云协议客户端直接互换。

## 推荐验证顺序

1. 固定上述上游版本，在独立 GPU 环境加载 1.5B，先验证启动、连续输入、1–2 秒短句 `end` 刷新以及非整块尾音。此阶段不改生产链路、不做通用 provider 抽象。
2. 对同一批真实录音进行按真实时钟重放，比较 Paraformer、已有 Qwen 流式和 VibeVoice；必要时追加 7B。覆盖中文近场、中英混合、领域热词、十语言中的业务目标语言、静音噪声、多人交替与重叠。
3. 分别记录首字时间、稳定文本延迟、停讲话后最终结果延迟、CER/WER、峰值显存、纯推理 RTF。若需要说话人功能，再测说话人归属错误和跨块身份一致性。
4. 验证 1/2/4 等目标并发，以及接近 8 分钟和超过 8 分钟时的截断/重连方案；重连会失去上下文与说话人编号连续性，必须把该代价纳入验收。
5. 只有当业务准确率收益、数秒级更新节奏和资源成本均可接受时，再实现现有实时协议到 VibeVoice 的适配，并删除 Paraformer 专属路径。

现有[实时 benchmark 客户端](/Users/quant/Documents/qwen3-asr/scripts/benchmark/clients/asr_client.py:186)的重放传输可复用，但其 `SentenceEnd` 处理会覆盖已保存文本，直接算 CER/WER 会只评价最后一句；需要先修正累计。其计时包含实时音频发送节奏，不能与论文纯推理 RTF 直接比较。验收数据应同时保留完整 transcript 和原始分块事件。

## 尚未验证

没有本项目语料上的 VibeVoice/Paraformer 对比，论文也没有对当前 Paraformer checkpoint 的直接比较；不能以会议榜单推断中文短句识别一定提升。目标 GPU、长期并发容量、现有 vLLM 0.19 插件兼容性、8 分钟以上的实用策略和多语言后处理均需上述 PoC 给出证据。
