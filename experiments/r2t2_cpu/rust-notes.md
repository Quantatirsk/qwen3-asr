# Rust CPU 实验

实验运行在 Apple M5 Pro / 64 GiB / macOS 27.2；仅使用 CPU Accelerate/vDSP，不使用 Metal。原始结果在 `rust-*.json`，记录库与音频 SHA256、加载耗时、进程峰值 RSS、首次与两次热推理；“首次”仅指进程内第一次调用，没有清理操作系统文件缓存。每个独立子进程限制 120 秒，超时保留已经完成的结果。

`rust-builds.json` 固定来源为主分支 `dd53fc0` 的 Rust 源码及每个文件 SHA256。源码只读复制到缓存目录再编译，未修改主分支。两种构建共享同一份 R2T2 safetensors 权重文件：

- `rust-int8`：原 ARM64 decoder INT8 权重量化实现，encoder / prefill 仍有浮点计算，不应称为全模型 INT8。
- `rust-bf16`：只将 `decode_kernel_mode()` 的 ARM64 分支改为 `Bf16`。BF16 权重经转换后进行 FP32 CPU 计算，不是原生 BF16 CPU 算术。

```sh
cargo build --release -p qwen-asr --features macos-ffi
python experiments/r2t2_cpu/rust_probe_check.py
python experiments/r2t2_cpu/rust_probe.py \
  --library "$HOME/.cache/r2t2-poc/rust-int8/target/release/libqwen_asr.dylib" \
  --model "$HOME/.cache/r2t2-poc/models/Confucius4-R2T2" \
  --audio "$HOME/.cache/r2t2-poc/audio/zh.wav" \
  --mode offline --output /tmp/rust-int8-zh.json
```

`--mode stream --chunk-seconds 2.0` 和 `0.16` 都以 160ms 音频块无等待回放；改变内部 chunk 大小。它们测试的是旧 Qwen 流式控制算法搭配新权重，**不能代表原生 R2T2 流式算法**。事件记录每次调用耗时、累积墙钟时间、当前音频时刻和增量文本；离线回放墙钟首字时间不等于真人录音首字延迟。

`raw` 是原 C API 暴露的文本（该 API 已去掉语言头），保留 `|`；`clean` 与现有 GPU 解析一致，只取第一个 `|` 前的文本。性能比较应使用 `clean`，同时审查 `raw` 中标记之后的异常生成。

## 对齐与共享

ForcedAligner 的中英文原生 CPU 调用已返回非空字/词序列，时间区间单调且不超过音频长度；存在零时长字词，与允许非递减的约束一致。最初两次对齐使用 CUDA 参考转写，不是人工逐字标注，也不是对对齐准确率的评测。其耗时可能受到同时编译影响。随后又用 INT8 R2T2 实际识别出的中英文原文调用对齐，结果见 `rust-align-recognized-{zh,en}.json`：英文通过；中文末尾句号被分配到 6800ms，超出 6740ms 音频，因此原始 native bounds 检查失败。生产 Python 对齐层必须沿用去标点的统一分字规则与时长修正，不能直接透传 native 时间戳。

现有 Rust `SharedQwenModel` 已经用 `Arc` 和按模型路径缓存实现进程内模型权重共享。不同 context 拥有独立 KV cache 与临时缓冲区，支持同进程统一模型、独立会话状态；跨进程不会共享动态生成的 INT8 / FP32 权重。

## 原生 R2T2 流式需要的接口

现有 C API 没有任意 assistant prefix 的生成接口；旧 `stream_push` 有自己的 rollback/cache/reset 规则，不能直接取代现有服务的 16 秒窗口、8 秒滑动、动态 token budget 和稳定前缀规则。

最小可复用边界是 Rust PCM 生成接口：接收 system token IDs、assistant prefix token IDs 和 max_new_tokens，返回生成 token IDs 与 stop/length 状态。Python 保留当前 Session 控制逻辑，并用 Hugging Face tokenizer 编码及按 token 扣留末尾；模型权重仍只加载一次。

不应把已有 Rust `tokenizer.encode()` 当作完整 HF tokenizer：它对整串做 BPE，缺少官方 regex 预分词与特殊 token 的识别。特别是 `language ...<asr_text>` 前缀必须通过精确 token IDs 传入；原始无 context 离线 prompt 的固定 token 不涉及这个差异。

## 已完成结果

以下是单进程两次热推理的平均耗时；进程内首次推理及加载耗时保留在 JSON 中。首次 INT8 模型加载为 11.30 秒，随后新进程借助系统文件缓存通常不到 1 秒，因此不把“冷加载”结果泛化为固定启动耗时。

| 音频 | INT8 热推理 / 秒 | BF16 权重路径 / 秒 |
|---|---:|---:|
| 中文 6.74 秒 | 0.795 | 0.923 |
| 英文 7 秒 | 0.897 | 1.083 |
| 拼接中英文 | 1.157 | 1.385 |
| 纯静音 | 0.519 | 0.499 |
| 降低音量中文 | 0.800 | 0.956 |
| 长音频 | 3.726 | 4.195 |

两种精度均完成所有 6 段音频的首次与两次热推理，无超时。INT8 峰值 RSS 约 8.15–8.49 GiB，BF16 权重路径约 10.41–10.74 GiB。中文、英文、降低音量中文和静音的主要内容一致；BF16 并未在这些样本上体现质量收益。拼接中英文整段离线时两种精度都丢失前半段中文，但 CUDA 参考也存在该现象，不能将它归因为 CPU 或量化回归。长录音实验保留全部原始输出，未做人工逐字标注，不能据此宣称正式准确率。

旧 Rust 流式算法的结果说明“只换权重”对实时还不够：

| 旧算法内部步长 | 中文耗时 | 英文耗时 | 主要现象 |
|---|---:|---:|---|
| 2 秒 | 2.004 秒 | 2.117 秒 | 单语言可用，中英拼接明显丢词 |
| 160ms | 14.790 秒 | 16.639 秒 | 无法实时追上音频，中文重复“也没”，拼接英文缺失 |

共享权重也做了实测：同进程第二个 context 加载约 26 微秒，峰值 RSS 只增加约 0.11 MiB，日志明确出现 shared weights reuse。这个数据是尚未推理的第二个 context，不能当成活跃并发请求的额外内存成本。

## 已落地原生生成接口

`vendor/qwenasr` 已恢复为仅库的 workspace，没有恢复原版模型配置或 CLI。新增 `generate.rs` 和 C ABI `qwen_asr_generate_pcm`：输入 PCM、audio placeholder 前后的精确 token IDs 及生成预算；输出 token IDs 与 stop/length 原因。逐项校验音频长度、有限值、token 范围及预算，FFI 捕获可展开的 Rust panic。共享权重结构沿用既有实现。

`rust-native-generate-{zh,en,prefix,budget}.json` 记录了新接口的真实加载/推理：中英文与原 offline 输出一致，assistant prefix 续写不会重发已给出的前缀，预算 1 返回 length，英文末尾 pipe ID 91 被保留供公共控制逻辑处理。这里的耗时包含 PoC 的 tokenizer 构建开销，不与前表作纯内核性能比较。PoC 直接使用 HF `tokenizers` 和模型自带 Jinja 模板，其空 prompt、中英 assistant prefix 与热词 context 三组 token IDs 已与 `AutoTokenizer(fix_mistral_regex=True)` 精确比对一致。

```sh
cargo build --release -p qwen-asr --features ffi
cargo test -p qwen-asr --features ffi --lib
CARGO_TARGET_DIR=/tmp/r2t2-bf16 cargo build --release -p qwen-asr --features ffi,bf16-decoder
```

上述两个 feature 构建均通过，库单元测试 10 项通过；新增输入校验测试属于其中一项。生产实时是否达到目标延迟由复用 Session 控制逻辑后的整体实验判断，不能由旧 stream 或单次离线速度代替。
