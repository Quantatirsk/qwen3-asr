# 当前运行时说明

这份文档只记录**当前主线代码真实成立的运行时事实**，用于减少 README / 部署文档的语义漂移。

如果本页与其他文档冲突，以本页和对应源码为准。

## 0. M5 Pro 转写实时率参考

以下数据来自当前仓库在 Apple Silicon `M5 Pro` 上的实测结果，目的是给出**量级参考**，不是跨机器 SLA。

测试口径：

- 后端：`QwenASR Rust`
- 设备：`macOS / Apple Silicon -> CPU`
- workers：`18`
- `ASR_BATCH_SIZE=4`
- 包含：`VAD + ASR + forced align`
- 不包含：`speaker diarization`

### 0.1 默认 `qwen3-asr-0.6b`

| 样本 | 时长 | total_sec | RTF | asr_sec | align_sec |
|------|------|-----------|-----|---------|-----------|
| 2 分钟样本 | `120s` | `17.78s` | `0.1481` | `9.54s` | `8.23s` |
| 10 分钟样本 | `600s` | `36.04s` | `0.0601` | `17.76s` | `18.28s` |

### 0.2 显式指定 `qwen3-asr-1.7b`

| 样本 | 时长 | total_sec | RTF | asr_sec | align_sec |
|------|------|-----------|-----|---------|-----------|
| 2 分钟样本 | `120s` | `18.72s` | `0.1560` | `10.48s` | `5.42s` |
| 10 分钟样本 | `600s` | `57.24s` | `0.0954` | `36.21s` | `19.40s` |

说明：

- 长样本的 RTF 明显更低，说明当前 Rust 路径在长音频上更容易摊薄初始化和调度成本
- 在当前本机 `Rust CPU + 18 workers` 口径下，`qwen3-asr-1.7b` 的 RTF 明显高于 `qwen3-asr-0.6b`
- 这也是 macOS / Apple Silicon 当前默认仍然选择 `qwen3-asr-0.6b` 的直接性能依据之一
- 这些值适合拿来做“本机参考基线”，不适合直接外推到 Linux x86_64 或 CUDA 环境

## 1. 设备解析规则

源码依据：

- [app/core/device.py](/Users/quant/Documents/funasr-api/app/core/device.py)

当前设备解析规则如下：

- `DEVICE=auto`
  - `CUDA 可用 -> cuda:0`
  - 否则 `cpu`
- `DEVICE=cuda`
  - 归一化为 `cuda:0`
- `DEVICE=mps`
  - 直接归一化为 `cpu`

这意味着：

- 当前仓库已经**没有独立的 Apple Silicon GPU / MLX 路径**
- Apple Silicon 现在统一走 **CPU / Rust** 路径

## 2. Qwen 离线路径默认选择

源码依据：

- [app/services/asr/model_plan.py](/Users/quant/Documents/funasr-api/app/services/asr/model_plan.py)

当前默认模型选择规则：

- `macOS / Apple Silicon`
  - 默认始终选择 `qwen3-asr-0.6b`
  - 不再根据内存大小自动漂到 `qwen3-asr-1.7b`
  - 只有调用方**显式指定**时，才会使用 `qwen3-asr-1.7b`
- `Linux / CPU`
  - 默认选择 `qwen3-asr-0.6b`
- `Linux / CUDA`
  - `显存 >= 32GB` 默认选择 `qwen3-asr-1.7b`
  - `显存 < 32GB` 默认选择 `qwen3-asr-0.6b`

## 3. 当前运行时矩阵

### 3.1 离线 Qwen

| 环境                  | 默认模型                                          | 后端                      |
| --------------------- | ------------------------------------------------- | ------------------------- |
| Linux + NVIDIA GPU    | `qwen3-asr-1.7b` / `qwen3-asr-0.6b`（按显存） | 官方 `vLLM` nightly     |
| Linux + CPU           | `qwen3-asr-0.6b`                                | vendored `QwenASR` Rust |
| macOS / Apple Silicon | `qwen3-asr-0.6b`                                | vendored `QwenASR` Rust |

### 3.2 词级时间戳

| 环境        | 实现                         |
| ----------- | ---------------------------- |
| CUDA        | 官方 `vLLM` forced aligner |
| CPU / macOS | Rust forced aligner          |

说明：

- `word_timestamps=true` 在当前离线路径下可用
- WebSocket 流式路径当前**不返回**词级时间戳

### 3.3 WebSocket realtime

| 能力                 | 当前实现                                | 设备行为                                    |
| -------------------- | --------------------------------------- | ------------------------------------------- |
| `paraformer-large` | FunASR realtime stack                   | 跟随 `DEVICE`，`mps` 最终仍落到 `cpu` |
| `qwen` WebSocket   | `CUDA -> vLLM`，`CPU/macOS -> Rust` | 跟随 `DEVICE`，`mps` 最终仍落到 `cpu` |

## 4. 当前 CPU Rust 并发参数

源码依据：

- [app/core/config.py](/Users/quant/Documents/funasr-api/app/core/config.py)

当前默认值：

| 环境变量                        | 默认值            | 当前语义                                                               |
| ------------------------------- | ----------------- | ---------------------------------------------------------------------- |
| `QWEN_RUST_CPU_WORKERS`       | `4`             | Rust runtime 数量                                                      |
| `QWEN_RUST_ASR_CONCURRENCY`   | `0`             | Rust ASR 阶段并行度；`0` 表示跟随 `QWEN_RUST_CPU_WORKERS`          |
| `QWEN_RUST_ALIGN_CONCURRENCY` | `0`             | Rust forced align 阶段并行度；`0` 表示跟随 `QWEN_RUST_CPU_WORKERS` |
| `QWENASR_CPU_NUM_THREADS`     | 自动 / 安全 `1` | 单个 runtime 的 CPU 线程数                                             |

补充事实：

- vendored Rust backend 已支持**同进程共享只读模型权重**
- 因此增加 runtime 数量时，不再按旧版本那样线性复制整套模型权重

## 5. 当前 CAM++ 说话人分离事实

源码依据：

- [app/utils/speaker_diarizer.py](/Users/quant/Documents/funasr-api/app/utils/speaker_diarizer.py)

当前事实：

- 说话人分离功能必须保留，当前主链仍然使用 CAM++
- CAM++ pipeline 跟随 `DEVICE`
  - `cuda:*` 时可跑 GPU
  - `cpu` 时跑 CPU
  - `mps` 会先归一化为 `cpu`
- 当前已经启用了 batched speaker verification
- batched SV 的默认 `max_batch_size` 当前是 `32`

已确认的性能结论：

- CPU 上的主要瓶颈是 `speaker verification embedding`
- 不是 `merge` / `clustering` / `change locator`
- 当前本机 profiling 下，把 batched SV 从 `16` 提到 `32` 有明显收益

## 6. 当前 x86 Rust 路径的边界

详细记录见：

- [docs/TODO/X86_RUST_ALIGN_OPT_PLAN.md](/Users/quant/Documents/funasr-api/docs/TODO/X86_RUST_ALIGN_OPT_PLAN.md)

当前结论：

- x86_64 Rust 路径已经从“不能跑”推进到“能跑”
- 但当前仍然更接近 **correctness-first fallback**
- Apple Silicon / aarch64 路径仍然是当前更成熟、更快的 Rust 主路径

## 7. 使用建议

如果只是想快速判断“现在到底会跑哪条路径”，优先记住这 4 条：

1. macOS / Apple Silicon 现在默认总是 `qwen3-asr-0.6b + Rust CPU`
2. 当前没有独立 MLX / MPS 路径
3. CUDA 才会走官方 `vLLM`
4. CAM++ 在 CPU 上可用，但热点主要在 `SV embedding`
