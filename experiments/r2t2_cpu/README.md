# R2T2 CPU 实验包

结论与限制见 [SUMMARY.md](SUMMARY.md)。所有音频来自公开上游样例，不包含用户会议录音。实验比较 Rust safetensors、官方 llama.cpp GGUF、CUDA 服务，以及旧流式算法与复用 R2T2 控制器后的 CPU 服务。

## 数据与环境

- macOS 27.2、Apple M5 Pro、64 GiB；CPU 实验使用 8 线程，未使用 Metal。
- `corpus.json` 固定来源、16 kHz PCM 与 WAV SHA256：中文 6.74s、英文 7s、中英拼接 14.24s、静音 5s、首 1.5s 降低音量的中文、三轮拼接 44.22s。
- `rust-builds.json` 固定旧 Rust 来源和文件散列；`llama-build.md` 固定官方源码、GGUF、依赖与构建方式。R2T2 safetensors revision 为 `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`。
- 项目依赖由 `uv.lock` 锁定；SciPy 在 macOS 为 1.16.3，Linux 为 1.15.3。原生库构建需要 Rust；音频准备需要 FFmpeg。
- 没有人工逐字真值。CUDA 输出只供跨后端对照；不计算或宣称准确率。各结果中的路径是实验机位置，复现时可替换。

## 最小复现

从仓库根目录执行，模型按项目启动流程准备；直接内核探针的 `--model` 需指向同一 R2T2 权重目录。为保留原始记录，重跑写入 `/tmp`。

```bash
uv sync --frozen
uv run python experiments/r2t2_cpu/prepare.py
./scripts/build-rust.sh
uv run python experiments/r2t2_cpu/rust_probe_check.py
DEVICE=cpu uv run python start.py
```

服务启动后，在另一个终端检查完整离线 API（说话人分离、独立识别、时间戳）：

```bash
uv run python experiments/r2t2_cpu/api_probe.py \
  --url http://127.0.0.1:8000 \
  --audio-dir "$HOME/.cache/r2t2-poc/audio" \
  --output /tmp/r2t2-cpu-api.json
```

共享引擎已启动时，使用实际内部 HTTP 与 WebSocket 地址；需要鉴权则通过 `R2T2_INTERNAL_TOKEN`、`API_KEY` 环境变量提供：

```bash
uv run python experiments/r2t2_cpu/service_probe.py \
  --backend rust-cpu --mode both --samples zh en mixed silence quiet long \
  --audio-dir "$HOME/.cache/r2t2-poc/audio" \
  --engine-url http://127.0.0.1:8001 \
  --stream-url ws://127.0.0.1:8000 \
  --output /tmp/r2t2-cpu-service.json
```

同一探针改用 CUDA 服务地址并设置 `--backend vllm-cuda` 即可对照。`service_probe.py` 的离线模式为引擎转写，不含说话人分离与对齐；完整链路用 `api_probe.py`。

## 记录索引

| 记录 | 用途 |
| --- | --- |
| `rust-notes.md`、`rust-*-offline-*.json` | 两种 Rust 精度、首次与热推理、峰值 RSS |
| `rust-int8-stream-*.json` | 旧 Rust 流式控制器的 2s / 160ms 对照 |
| `rust-native-generate-*.json`、`rust-int8-shared.json` | 精确 token IDs、前缀续写、预算及权重共享 |
| `rust-align*.json` | ForcedAligner 原始与识别文本对齐，包括失败边界 |
| `llama-build.md`、`llama-findings.md`、`llama-*.json` | 官方 Q8 decoder / F16 encoder 的离线与流式 |
| `cuda_results.json` | CUDA 引擎离线与实时回放参考 |
| `rust_service_results.json`、`rust_service_640ms.json` | 新控制器修正前的 2s / 640ms 失败记录 |
| `rust_service_640ms_fixed.json` | 160ms 停顿检测、640ms 解码的真实节奏回放 |
| `cpu_api_results.json` | 完整 CPU API 初轮结果，含静音失败 |

内核级复现参数与构建差异见 `rust-notes.md`；官方 GGUF 独立环境的复现命令见 `llama-build.md`。探针将失败写入 JSON，不能仅凭脚本退出判断全部通过。

最终整合后的接口验收见 `cpu_api_final.json`；补充的 llama 自动语言结果见 `llama-auto.json`。Mac 默认使用 Rust INT8、640ms 解码间隔、单实时会话，并按最多 160ms 的输入帧检测停顿。
