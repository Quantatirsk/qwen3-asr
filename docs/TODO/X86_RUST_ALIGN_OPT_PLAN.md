# x86 Rust Align Optimization Plan

## 当前结论

- 当前工作区的 vendored Rust backend 已经收敛到更接近 upstream `huanglizhuo/QwenASR` 的 Linux/x86_64 路径：
  - `release`
  - `RUSTFLAGS="-C target-cpu=native"`
  - `BLAS/OpenBLAS`
  - x86_64 默认 `BF16` decode
- 保留的有意偏离只有两类：
  - `SharedQwenModel` / shared model cache
  - 中性的 `ffi` feature（`macos-ffi` 仅作为兼容别名保留）

## upstream 参考

- upstream repo: `https://github.com/huanglizhuo/QwenASR`
- inspected commit: `4e85a19b05f034e106a345d279c68f50df718ab8`

## 本机环境

- CPU: `Intel Core i5-13600KF`
- visible CPUs: `14`
- memory: user reported `G.SKILL DDR5-6400`

## 已验证 benchmark

音频：

- `/opt/funasr-api/temp/test_assets/podcast_demo_2min_16k.wav`
- duration: `120s`

### decode 路径对照（runtime concurrency = 4）

`INT8 decode`

- total: `173.42s`
- asr: `54.69s`
- align: `118.74s`
- rtf: `1.4452`

来源：

- `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_int8.json`

`BF16 decode`

- total: `125.87s`
- asr: `46.40s`
- align: `79.47s`
- rtf: `1.0489`

来源：

- `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_bf16.json`

结论：

- 在当前这台 x86_64 机器上，`BF16 decode` 明显优于 `INT8 decode`
- 因此 x86_64 默认 decode 路径应保持 `BF16`

### 收敛后的默认路径 benchmark

来源：

- `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_default_after_converge.json`

结果：

- `runtime concurrency = 4`
  - total: `131.77s`
  - asr: `51.54s`
  - align: `80.23s`
  - rtf: `1.0981`
- `runtime concurrency = 14`
  - total: `128.25s`
  - asr: `57.12s`
  - align: `71.13s`
  - rtf: `1.0688`

结论：

- 当前主瓶颈仍然在 `align`
- `align_sec` 明显大于或接近 `asr_sec`
- `runtime concurrency` 的最优值并不稳定，说明问题不在单纯线程数，而在具体阶段的访问模式和 kernel 行为

## 为什么不继续默认走 INT8

- upstream README 对 Linux/x86_64 的主路径描述是 `BLAS + AVX2/FMA`
- 当前本机实测中，`INT8 decode` 明显慢于 `BF16 decode`
- 说明这台机器上的主瓶颈不只是权重带宽，更多是：
  - x86_64 上 INT8 kernel 的有效带宽利用率
  - cache / 数据布局
  - 实现成熟度差异

## 当前仍需保留的偏离

### 1. Shared model cache

文件：

- `/opt/funasr-api/vendor/qwenasr/crates/qwen-asr/src/context.rs`

目的：

- 多 runtime / 多 worker 场景下复用只读模型权重
- 避免每个 runtime 重复 mmap / 持有整套权重

### 2. ffi feature

文件：

- `/opt/funasr-api/vendor/qwenasr/crates/qwen-asr/Cargo.toml`
- `/opt/funasr-api/vendor/qwenasr/crates/qwen-asr/src/lib.rs`
- `/opt/funasr-api/Dockerfile.cpu`

目的：

- 让 Linux CPU 集成不再依赖命名不准确的 `macos-ffi`
- 同时保留兼容别名，避免已有脚本立即失效

## align 热点拆解计划

### Phase 1: 阶段级 profiling

状态：已完成

目标：

- 先确认 `align` 的主耗时究竟在哪一段

位置：

- `/opt/funasr-api/vendor/qwenasr/crates/qwen-asr/src/align.rs`
- `/opt/funasr-api/vendor/qwenasr/crates/qwen-asr/src/decoder.rs`

需要拆出的阶段：

- `mel_spectrogram`
- `encoder.forward`
- `input_embeds build`
- `decoder_prefill_logits`
- `timestamp argmax extract`
- `fix_timestamps`

验收：

- 2 分钟样本上输出稳定的阶段级耗时表

实际结果：

- 产物：
  - `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_align_profile.json`
  - `/opt/funasr-api/temp/test_logs/qwen_rust_runtime_concurrency_2min_align_profile.stderr`
- 结论：
  - `align` 的主热点明确落在 `decoder_prefill_logits`
  - `final rms_norm` 和 `lm_head projection` 不是主矛盾

### Phase 2: decoder_prefill_logits 内部分解

状态：已完成

如果 `decoder_prefill_logits` 是主热点，则继续拆分：

- `decoder_prefill`
- final `rms_norm`
- `lm_head projection`

目的：

- 判断到底是 decoder prefill 慢，还是最后分类头 projection 慢

实际结果：

- 产物：
  - `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_align_breakdown.json`
  - `/opt/funasr-api/temp/test_logs/qwen_rust_runtime_concurrency_2min_align_breakdown.stderr`
- 结论：
  - 真正的大头是 `decoder_prefill`
  - 长段（`seq_len=1801`）时，`attention_ms` 占 `decoder_prefill` 的绝大部分
  - 关键样本：
    - `decoder_prefill total_ms=79280.78`
    - `attention_ms=73769.91`
    - `qkv_ms=1336.33`
    - `gate_up_ms=1866.68`
    - `down_proj_ms=955.94`

### Phase 2.5: 失败尝试记录

状态：已完成并回退

尝试：

- 针对 x86_64 预先物化 prefill 用 F32 权重，避免每次 `align` 反复做 `BF16 -> F32`

结果：

- 长段 `decoder_prefill` 没有稳定收益，反而出现回归
- 这条路径已经回退，不保留在主线代码里

结论：

- 当前瓶颈不是简单的 BF16 权重转换
- 更直接的问题是 multi-token causal attention 的算法路径

### Phase 3: 对热点段做针对性优化

状态：第一轮已完成

根据 profiling 结果，按优先级选一个方向：

1. 如果热点在 `input_embeds build`
   - 复用固定 prefix/suffix embeddings
   - 减少逐 token 小块 copy
   - 降低每段 align 的重复构造开销

2. 如果热点在 `decoder_prefill`
   - 检查 `BF16 matvec / attention / swiglu` 的实际热点
   - 优化并行粒度或数据布局

3. 如果热点在 `lm_head projection`
   - 优先优化 `BF16` classify head 路径
   - 避免无价值的整块 materialize
   - 但只有在 profiling 证明是主热点后才动

4. 如果热点在后处理
   - 精简 `fix_timestamps`
   - 减少 `Vec/String` 分配

### Phase 3 实施结果

本轮实际落地的是：

- 文件：
  - `/opt/funasr-api/vendor/qwenasr/crates/qwen-asr/src/kernels/mod.rs`
- 改动：
  - 对 BLAS multi-token causal attention 增加长序列专用 batched 路径
  - 仅在 `seq_q >= 256` 时启用
  - 从“每个 head、每一行 2 次小 GEMM”改为：
    - 每个 head 1 次 `Q @ K^T`
    - 行级 causal softmax
    - 每个 head 1 次 `softmax @ V`

### Phase 3 回归结果

来源：

- 优化后 benchmark：
  - `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_after_attention_opt.json`
- 优化后 profiling：
  - `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_attention_opt_profile.json`
  - `/opt/funasr-api/temp/test_logs/qwen_rust_runtime_concurrency_2min_attention_opt_profile.stderr`

关键对比：

- 之前默认路径（2 分钟样本，基线文件）：
  - `total=128.25s`
  - `asr=57.12s`
  - `align=71.13s`
- 优化后：
  - `total=76.46s`
  - `asr=45.32s`
  - `align=31.14s`
  - `rtf=0.6372`

attention 热点变化：

- 长段 `seq_len=1801`
  - 优化前：
    - `decoder_prefill total_ms=79280.78`
    - `attention_ms=73769.91`
  - 优化后：
    - `decoder_prefill total_ms=16625.88`
    - `attention_ms=9981.88`

结论：

- 当前这台 i5 上，`align` 的主矛盾已经从“attention 明显失控”收敛到了“attention 仍是第一热点，但已降到可接受量级”
- 这一轮优化是有效的，应该保留

## Phase 4: FFN 路径继续收敛

状态：已完成一轮，并保留有效部分

本轮动作：

- 文件：
  - `/opt/funasr-api/vendor/qwenasr/crates/qwen-asr/src/decoder.rs`
- 改动：
  - 仅对 x86_64 `BF16 prefill` 的 FFN 路径物化共享 F32 权重
  - 范围只包括：
    - `gate_up_fused`
    - `down_weight`
  - `QKV` 和 `O-proj` 暂不纳入

回归数据：

- 不带 profiling：
  - `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_after_ffn_opt.json`
  - `total=58.61s`
  - `asr=29.50s`
  - `align=29.11s`
- 带 profiling：
  - `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_ffn_opt_profile.json`
  - `total=62.07s`
  - `asr=32.15s`
  - `align=29.92s`

与上一轮 attention-only profiling 对比：

- attention-only：
  - `total=64.58s`
  - `align=30.86s`
- FFN-opt：
  - `total=62.07s`
  - `align=29.92s`

长段热点对比（`seq_len=1801`）：

- FFN-opt 之前：
  - `attention_ms=9981.88`
  - `gate_up_ms=2080.10`
  - `down_proj_ms=1183.64`
- FFN-opt 之后：
  - `attention_ms=9660.15`
  - `gate_up_ms=2210.39`
  - `down_proj_ms=1166.96`

结论：

- FFN 这轮不是“大收益”，但 `align_sec` 仍然有小幅下降
- 收益不像 attention 优化那样压倒性，更像是小幅收敛
- 当前可以保留，但不值得继续在同一方向上扩大复杂度

## Phase 5: QKV / O-proj 试验与回退

状态：已完成并回退

尝试：

- 在 prefill 中进一步物化 `QKV` 和 `O-proj` 的共享 F32 权重

回归数据：

- 试验版本：
  - `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_after_qkv_opt.json`
  - `total=63.12s`
  - `align=30.07s`

结论：

- 相比 FFN-opt 版本，没有形成净收益
- 因此这条路径已回退，不保留在主线代码里

## Phase 6: attention 继续细化的两次试验

状态：已完成并回退

### 试验 A：query-block batched attention

尝试：

- 将长序列 batched causal attention 从“整段一次性 `QK^T` / `SV`”改成按 query block 分块执行

结果：

- `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_after_attention_block_opt.json`
- `total=79.18s`
- `align=33.40s`

结论：

- 这条路径在当前 i5 + OpenBLAS 组合下没有收益
- 增加 GEMM 次数带来的额外调度开销，超过了小块缓存收益
- 已回退

### 试验 B：提高 batched 切换阈值到 512

尝试：

- 只改 `BATCHED_CAUSAL_ATTENTION_THRESHOLD`
- 让中等长度序列继续走 row-wise 路径，只把更长的序列交给 batched 路径

结果：

- `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_after_attention_threshold512.json`
- `total=74.49s`
- `align=30.42s`

结论：

- 相比当前主线最优版本也没有收益
- 说明当前阈值 `256` 不是主要问题
- 已恢复回 `256`

## Phase 7: K/V block + online softmax 试验

状态：已完成并回退

尝试：

- 仅替换长序列 batched attention 路径
- 改为按 `K/V` block 流式累积的 online softmax
- 短序列和单 token 路径完全不动

目标：

- 不再一次性物化整块 `scores`
- 降低大矩阵内存压力
- 观察是否能进一步压低长段 `attention_ms`

结果：

- `/opt/funasr-api/temp/test_assets/qwen_rust_runtime_concurrency_2min_after_kvblock_online_softmax.json`
- `total=74.11s`
- `align=30.33s`

结论：

- 当前实现下没有优于主线最优版本
- 在这台机器上，额外的 block 循环和 online softmax 合并开销，超过了减少大 `scores` 矩阵带来的收益
- 已回退

## 当前判断

- 当前最值钱、且已验证有效的优化仍然是：
  - 长序列 batched causal attention
  - FFN selective F32 物化
- 继续扩大到 `QKV/O-proj` 这一步暂时不划算
- query-block 化和 batched 阈值调优目前也不划算
- `K/V block + online softmax` 在当前实现形态下也不划算
- 后续判断应优先看：
  - `align_sec`
  - `decoder_prefill` profiling
  - 尤其是长段 `seq_len` 下的热点变化

补充：

- `asr_sec` 在多次回归中波动明显大于 `align_sec`
- 因此后续评估优化效果时，不应只盯总耗时，应优先以 `align` profiling 为准

## 暂不做的事

- 不再把 x86_64 默认路径改回 `INT8 decode`
- 不继续做没有 profiling 支撑的 `align` 结构性改写
- 不围绕 `runtime concurrency` 数量盲调

## 下一步执行顺序

1. 保留当前 batched causal attention 路径，继续观察不同长段下的稳定性
2. 保留 FFN selective F32 物化，继续观察其稳定收益
3. 如需继续优化，优先看：
   - 更激进的 `attention` 算法级改动，例如按 `K/V` block 的 online softmax
   - 再其次才是 `qkv_ms + gate_up_ms + down_proj_ms`
4. 如果后续继续深挖，再考虑：
   - batched attention 的 block 化，降低大 score matrix 的瞬时内存
   - `decoder_prefill` 内的投影层进一步收敛
5. 保持同一份 2 分钟样本持续回归，避免再次把回归误当成优化
