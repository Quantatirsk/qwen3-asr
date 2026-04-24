# Qwen3-ASR 并发性能测试脚本

测试 ASR/TTS WebSocket 服务在不同并发级别下的性能表现。

## 依赖

```bash
./scripts/sync_cpu_env.sh
```

## 快速开始

### 1. 启动服务

```bash
# 在项目根目录
source .venv/bin/activate
python start.py
```

### 2. 运行测试

```bash
# 完整测试 (ASR + TTS)
.venv/bin/python -m scripts.benchmark.run --audio-file /path/to/audio.wav

# 仅测试 TTS (无需音频文件)
.venv/bin/python -m scripts.benchmark.run --test-type tts

# 仅测试 ASR
.venv/bin/python -m scripts.benchmark.run --audio-file /path/to/audio.wav --test-type asr

# Qwen Rust CPU 固定配置跑测（固定 VAD 分段）
.venv/bin/python -m scripts.benchmark.qwen_rust_sensitivity \
  --audio-file /path/to/audio.wav
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | localhost | 服务器主机名 |
| `--port` | 8000 | 服务器端口 |
| `--audio-file` | - | ASR 测试音频文件路径 (测试 ASR 时必需) |
| `--test-type` | both | 测试类型: `asr` / `tts` / `both` |
| `--concurrency` | 5 10 20 50 | 并发级别列表 |
| `--output` | ./benchmark_results | 报告输出目录 |
| `--timeout` | 120 | 请求超时时间 (秒) |
| `--voice` | 中文女 | TTS 测试音色 |

## Qwen Rust CPU 固定配置跑测

用于对 vendored Rust backend 做固定 workload 的端到端对比：

- 同一组 VAD 分段
- Rust ASR
- Rust forced align
- 跑当前环境变量指定的 Rust 并发配置

示例：

```bash
# 直接跑当前配置
.venv/bin/python -m scripts.benchmark.qwen_rust_sensitivity \
  --audio-file temp/podcast_demo_10min_16k.wav \
  --json-out temp/qwen_rust_runtime_config.json

# 通过环境变量切换不同并发配置后再跑
QWEN_RUST_CPU_WORKERS=8 \
QWEN_RUST_ASR_CONCURRENCY=8 \
QWEN_RUST_ALIGN_CONCURRENCY=4 \
.venv/bin/python -m scripts.benchmark.qwen_rust_sensitivity \
  --audio-file temp/podcast_demo_10min_16k.wav \
  --json-out temp/qwen_rust_runtime_config.json
```

说明：

- 脚本只跑当前一组配置
- 并发度由 `QWEN_RUST_CPU_WORKERS`、`QWEN_RUST_ASR_CONCURRENCY`、`QWEN_RUST_ALIGN_CONCURRENCY` 控制
- 跑完后会输出一条进度日志，并写入 `--json-out`
- 如果传了 `--json-out`，脚本还会自动生成同名 `.md` 中文对比报告
- 也可以显式指定 Markdown 路径：

```bash
.venv/bin/python -m scripts.benchmark.qwen_rust_sensitivity \
  --audio-file temp/podcast_demo_10min_16k.wav \
  --json-out temp/qwen_rust_runtime_config.json \
  --markdown-out temp/qwen_rust_runtime_config_report.md
```

### TTS 流式模拟配置 (config.py)

TTS 测试模拟 LLM 流式输出场景，按标点符号（逗号、句号、顿号等）分割文本逐步发送：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `tts_chunk_interval` | 0.05 | 发送间隔秒数 (模拟 LLM 生成速度) |

## 使用示例

```bash
# 自定义并发级别
.venv/bin/python -m scripts.benchmark.run \
  --audio-file test.wav \
  --concurrency 5 10 20 50 100

# 连接远程服务器
.venv/bin/python -m scripts.benchmark.run \
  --host 192.168.1.100 \
  --port 8000 \
  --test-type tts

# 使用不同音色测试 TTS
.venv/bin/python -m scripts.benchmark.run \
  --test-type tts \
  --voice 中文男
```

## 测试指标

### ASR 指标
- **首次响应延迟**: 从开始到收到第一个识别结果的时间
- **总处理时间**: 从开始到识别完成的总时间
- **RTF**: 处理时间 / 音频时长 (小于 1.0 表示快于实时)

### TTS 指标
- **首包延迟**: 从发送文本到收到第一个音频块的时间
- **总合成时间**: 从开始到合成完成的总时间
- **RTF**: 合成时间 / 生成音频时长

### 统计维度
每个指标计算: 平均值 (Avg)、P50、P95、P99、最大值 (Max)

## 输出文件

测试完成后在输出目录生成:

```
benchmark_results/
├── benchmark_report_20241202_143000.md   # Markdown 报告
├── first_latency_20241202_143000.png     # 首次响应延迟图
├── rtf_20241202_143000.png               # RTF 对比图
├── throughput_20241202_143000.png        # 吞吐量图
└── total_time_20241202_143000.png        # 总时间图
```

## 报告示例

```markdown
## ASR 性能测试结果

### 延迟指标 (毫秒)

| 并发数 | 首次响应 (Avg) | 首次响应 (P95) | 总时间 (Avg) | 总时间 (P95) |
|--------|---------------|---------------|-------------|-------------|
| 5      | 245.3         | 312.5         | 62345.2     | 63521.4     |
| 10     | 289.7         | 425.3         | 63245.8     | 65123.6     |

### RTF 和吞吐量

| 并发数 | RTF (Avg) | RTF (P95) | 吞吐量 (req/s) | 成功率 |
|--------|----------|----------|---------------|--------|
| 5      | 1.04     | 1.06     | 0.08          | 100%   |
| 10     | 1.05     | 1.08     | 0.16          | 100%   |
```

## 目录结构

```
scripts/benchmark/
├── run.py              # 主入口脚本
├── config.py           # 测试配置
├── clients/
│   ├── base_client.py  # WebSocket 客户端基类
│   ├── asr_client.py   # ASR 测试客户端
│   └── tts_client.py   # TTS 测试客户端
├── metrics/
│   ├── models.py       # 指标数据类
│   └── statistics.py   # 统计计算
├── reporters/
│   ├── markdown_reporter.py  # Markdown 报告生成
│   └── chart_generator.py    # 图表生成
└── utils/
    ├── audio_utils.py  # 音频文件处理
    └── text_generator.py  # 测试文本生成
```

## 注意事项

1. **ASR 测试需要音频文件**: 建议使用 1 分钟左右的音频，格式支持 wav/mp3 等常见格式
2. **TTS 测试自动生成文本**: 使用内置的中文随机句子生成器，无需额外准备
3. **TTS 模拟流式输入**: 测试会按标点符号（逗号、句号、顿号等）分割文本逐步发送，模拟 LLM 流式输出场景
4. **并发测试会占用资源**: 高并发测试时请确保服务器有足够资源
5. **RTF 解读**:
   - RTF < 1.0: 处理速度快于实时，性能良好
   - RTF ≈ 1.0: 刚好实时处理
   - RTF > 1.0: 处理速度慢于实时，可能出现延迟累积
