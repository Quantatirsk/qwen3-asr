# R2T2 的 macOS / CPU / Rust 适配可行性

调研日期：2026-09-26。本文件保留静态研究阶段的一手证据；后续真实 CPU 实验与重构已完成，最新结果见 [CPU PoC](../../experiments/r2t2_cpu/SUMMARY.md)。下文“尚未验收”描述的是研究当时的状态。

## 外部 / 上游发现

**结论：模型并非只能依赖 CUDA，离线复用 Qwen3-ASR 推理具有直接代码依据；能否零修改替换本项目 Rust 权重，仍要核对本地实现及实际输出。实时质量不能只靠更换权重保证。**

版本边界：上游 GitHub 源码固定到 `26d55a54ce5670cff9947a167d8ed95d569fd4d9`；已核对部署模型的 Hugging Face revision 为 `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`。两者属于不同仓库，不能把源码提交当作权重提交。

### 离线架构与流式解码

`R2T2ASRModel` 直接继承 `Qwen3ASRModel`，其 `transcribe()` 将全部参数转发给父类。这比“基于 Qwen 训练”更直接地表明官方离线复用了原模型推理路径。流式路径则额外管理累计音频、已确认前缀、token 回退、管道符截断及收尾；复用底层矩阵计算不等于复现这些流式语义。实际代码还会重新编码当前累计音频，不能按“每个音频块只处理一次”的方式估计 CPU 成本。[源码](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/r2t2/r2t2_asr.py#L176)

官方 llama 包继续使用 Qwen 的 config/processor，并特意保留 HF tokenizer 保证前缀 token 边界与 vLLM 一致。`R2T2LlamaASRModel.transcribe()` 自身并不支持离线，离线应走 `LlamaNativeOnetime`；不能仅凭类名假设接口相同。[流式包装](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/r2t2_llama/model.py#L123)、[离线包装](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/r2t2_llama/llama_native_backend.py#L65)

### CPU / macOS 的实际支持边界

| 官方路径 | 音频编码 | 文本解码 | 对本任务的意义 |
| --- | --- | --- | --- |
| `stream_llama_hybrid` | PyTorch / vLLM GPU | llama.cpp | 仍需要 GPU，不是纯 CPU 方案 |
| `stream_llama` | llama.cpp | llama.cpp | 有全原生流式推理基础 |
| `onetime_llama` | llama.cpp | llama.cpp | 有全原生离线推理基础 |

上游推荐 hybrid，并明确解释原生与 PyTorch encoder 在较轻的音频起始部分鲁棒性不同；这是另一个后端的已知质量风险，不能自动推定现有 Rust 同样受影响。预编译产物仅覆盖 Linux x86_64 / CUDA / CPython 3.12；官方说明可重编 CPU 版本。[上游后端说明](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/r2t2_llama/README.md)

C++ 参数将 `use_gpu` 用于 encoder 和 KQV offload，将 `n_gpu_layers` 单独用于 decoder；纯 CPU 应同时关闭 offload 并设置零 GPU 层。生成循环以模型 EOG 终止；`|`、`#` 在另一个辅助函数中作为可选 stop-bias 候选，并不是该文件所有解码路径的硬编码停止条件。[原生实现](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/r2t2_llama/native_ext.cpp#L247)

其固定版本 llama.cpp `ad6c66839af3c5646fba8c6c2e2087a1e4e38948` 支持 macOS Accelerate 和 Metal；macOS 默认启用 Metal，纯 CPU 可编译时设 `GGML_METAL=OFF`。这证明底层运行库有平台基础，尚不等于 R2T2 包经过 Mac 验收。[llama.cpp 构建文档](https://github.com/ggml-org/llama.cpp/blob/ad6c66839af3c5646fba8c6c2e2087a1e4e38948/docs/build.md#metal-build)

直接安装官方 Python 包仍会引入 `qwen-asr[vllm]`，打包配置只包含 `.so`；流式控制代码指定生成 token 数时也会导入 vLLM 的 `SamplingParams`。因此纯 llama 内核运算可行与整个 Python 应用脱离 vLLM 是两件事。Mac 适配仍需要调整依赖和动态库打包，或复用本项目已有 Rust 执行层。[依赖配置](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/pyproject.toml)、[CMake 配置](https://github.com/netease-youdao/Confucius4-R2T2/blob/26d55a54ce5670cff9947a167d8ed95d569fd4d9/r2t2_llama/CMakeLists.txt)

官方另行发布 decoder 的 F16 / Q8_0 / Q4_K_M GGUF 与音频 projector 的 F16 / Q8_0 GGUF。此资源支持新增 llama 后端的备选路线，不能把 GGUF 文件直接当作已有 safetensors Rust loader 的输入。[官方 GGUF 模型卡](https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF)

## 本项目 main 代码 / 实际 checkpoint 验证

主研究对服务器已缓存 checkpoint 的配置文件及 safetensors header 作只读比对，没有加载 GPU 模型：

- R2T2 revision：`185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`；Qwen3-ASR-1.7B revision：`7278e1e70fe206f11671096ffdd38061171dd6e5`。
- `config.json` 结构完全一致：`Qwen3ASRForConditionalGeneration`，24 层 audio encoder、28 层 decoder、hidden size 2048、16 个 attention heads / 8 个 KV heads，`tie_word_embeddings=true`。
- R2T2 为 707 个张量，原版为 708 个；707 个共有张量的名称、shape、dtype 全部一致。唯一缺少 `thinker.lm_head.weight`。
- `vocab.json`、`merges.txt`、`tokenizer_config.json`、`preprocessor_config.json`、`chat_template.json`、`generation_config.json` 逐字节一致。R2T2 额外包含原版缓存没有的 `tokenizer.json`、`added_tokens.json`、`special_tokens_map.json`。
- 关键形状：embedding `[151936, 2048]`、decoder gate projection `[6144, 2048]`、audio encoder layer 18 权重 `[1024, 1024]`；比较依据是模型头部元数据，不是权重数值相等。

缓存快照与公开来源对应：[R2T2 固定版本](https://huggingface.co/netease-youdao/Confucius4-R2T2/tree/185ce639118ad1362d049ca0d8ed04b6ec5cd6c9)、[原版固定版本](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/7278e1e70fe206f11671096ffdd38061171dd6e5)。

main 中的 vendored Rust revision 为 `4e85a19b05f034e106a345d279c68f50df718ab8`。现有 loader 根据权重形状检测架构；decoder 在 `classify_num > 0` 时要求独立 lm_head，普通 ASR 尝试读取独立 lm_head，缺失时使用共享 embedding。因此上述缺少的 lm_head **不构成该加载路径的静态阻碍**。来源：[config.rs](/Users/quant/Documents/qwen3-asr/vendor/qwenasr/crates/qwen-asr/src/config.rs)、[decoder.rs](/Users/quant/Documents/qwen3-asr/vendor/qwenasr/crates/qwen-asr/src/decoder.rs:235)。

**目前结论是静态兼容性较强，尚未完成 R2T2 的 Rust 加载和音频推理验收。** 本机 M5 Pro / 64 GiB 有已构建的 Rust 动态库，但没有 R2T2 或原版 1.7B 权重，现有本地缓存主要为 0.6B 和对齐模型。不能据此报告 R2T2 在 Mac 上的正确率、延迟或内存数值。

主研究另确认四项实施边界：

1. ARM64 的 Rust decoder 选择 INT8 路径，不能假定与 GPU BF16 数值及文字输出完全相同。[decoder.rs](/Users/quant/Documents/qwen3-asr/vendor/qwenasr/crates/qwen-asr/src/decoder.rs:60)
2. 旧流式默认 2 秒 chunk、回退 5 tokens、前 2 chunks 不固定、最多新生成 32 tokens，与 R2T2 示例的 160 ms、lookahead 和 1 token 回退明显不同；更换权重不会自动改变实时语义。[context.rs](/Users/quant/Documents/qwen3-asr/vendor/qwenasr/crates/qwen-asr/src/context.rs:215)
3. 旧离线生成最多 2048 tokens，停止条件仅 EOT / IM_END，未处理 R2T2 包装中的 `|`；这是输出验收和最小适配的检查点，不能在无实际样本时声称必然出错。[transcribe.rs](/Users/quant/Documents/qwen3-asr/vendor/qwenasr/crates/qwen-asr/src/transcribe.rs:195)
4. 应用层 model plan / 模型清单显式限定旧模型名称。因此“底层 loader 很可能能读”不等于“整个服务只改环境变量就能运行”，仍要调整模型注册和后端路由。[model_plan.py](/Users/quant/Documents/qwen3-asr/app/services/asr/model_plan.py)、[models.json](/Users/quant/Documents/qwen3-asr/app/services/asr/models.json)

## 最小验证范围

静态兼容性检查已完成。下一步在隔离目录直接调用主分支 `QwenASRRustBackend(model_path=...)`，用 R2T2 原始 safetensors 目录及相同中英文音频验证离线输出、结束标记和耗时，无需先恢复整套 CPU 服务。R2T2 单文件权重为 4,076,191,640 字节；当前本机未缓存，后续实验需要下载或从已有服务器缓存复制。若目标还包括实时，则再核对前缀回退与收尾协议，并量测实际 chunk 延迟，不能只看离线实时率。说话人分离和 ForcedAligner 的 CPU 能力属于独立组件，不能由 ASR 权重兼容性推出。
