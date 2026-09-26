# R2T2 实时转写

离线文件与实时音频均使用 Confucius4-R2T2，各自独立推理。实时只提供一个 WebSocket 协议，默认自动识别语言，支持中英文及混合输入。没有旧协议别名，也没有 Chat 音频适配层。

## 架构

一个容器、一个 Python 环境、两个进程：公共 API 提供离线接口和 WebSocket 转发；私有 R2T2 进程持有一个 vLLM AsyncLLM 引擎。多个连接的解码请求由 vLLM 连续批处理，各自独立保存音频、提示词和已确认文本。私有进程统一限制接入数量，公共 API 增加 worker 不会绕过限额。

核心代码：`app/services/realtime/engine.py`（解码），`server.py`（接入与生命周期），`protocol.py`（音频协议与队列），`client.py` / `gateway.py`（转发）。`deploy/entrypoint.py` 管理两个进程，任一进程退出即关闭容器，由容器策略重启。

## 启动

```bash
cp .env.example .env
docker compose up -d --build
```

默认使用 GPU 0，公共端口 4174；模型缓存、显存预算和鉴权配置见 [部署说明](deployment.md)。浏览器麦克风需要 HTTPS 或 localhost。实时和离线引擎共享 GPU 计算资源，需要用真实并发工作负载验收延迟。

## 协议

连接 `ws://host:4174/v1/stream`，鉴权可用 `Authorization: Bearer ...` 或浏览器 `?token=...`。首条消息为 `{}`，可选热词上下文 `{"context":"网易有道，Qwen，R2T2"}`。不接受语言选择或采样参数，始终自动识别和确定性解码。

服务返回 `ready:true` 后发送 **16 kHz、单声道、int16 little-endian PCM** 二进制帧。建议每帧 160 ms（5120 字节），单帧最多 1 秒。结束时发送文本 `end`，继续接收至 `done:true`。

```json
{"delta":"Hello","audio_ms":960,"inference_ms":45.2,"done":false}
{"delta":".","audio_ms":1280,"inference_ms":48.1,"done":true,"text":"Hello."}
```

客户端直接追加 `delta`；已发布文字不改写。结束事件包含完整 `text`。空增量表示已处理一块音频，可作为进度心跳。`audio_ms` 是已处理音频位置，**不是文字时间戳**；`inference_ms` 包括该解码请求的调度和推理时间，不等于从说话到出字的完整延迟。

错误为 `{"code":"capacity_exceeded","error":"..."}` 并关闭连接。每会话最长 1 小时，输入队列最多 10 秒；超时、积压、断线会取消该连接的 vLLM 请求并释放名额。没有实时说话人标签或词级时间戳，文件级标注仍使用离线接口。

## 解码与版本

- 模型 `netease-youdao/Confucius4-R2T2`，revision `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`。
- 按 2026-09-25 上游最新 commit `26d55a54ce5670cff9947a167d8ed95d569fd4d9` 的滚动窗口方法适配为原生异步调用。
- vLLM `0.19.0`、PyTorch `2.10.0`、Transformers `4.57.x`，使用 vLLM 内置 Qwen3-ASR processor。无需额外 Python 依赖或模型插件。
- 首次解码收集 320 ms 音频，以后每 160 ms 解码。实际出字还取决于语音上下文和推理时间，320 ms 不是首字延迟保证。
- 音频窗口最多 16 秒，超过后移除最早 8 秒及相应文本前缀；按采样位置对齐，避免长录音无限重算。保留英文空格；结束时即使恰好对齐完整块，也刷新最后一个未确认 token。
- 语音之后检测到连续至少 320 ms 的低能量音频（RMS < 0.004）时，补齐当前语句并清空识别窗口，下一句重新自动识别语言。此边界检测不丢弃音频、不关闭连接，不依赖额外 VAD 模型。持续背景噪声可能使边界不触发；无停顿的任意语言切换仍受模型能力限制。
- 长音频仍会重复编码当前窗口，模型本身不是完全增量的音频编码器。自动语言识别不构成对任意语言切换、口音或噪声的准确率保证。

算法源自 [NetEase Youdao R2T2](https://github.com/netease-youdao/Confucius4-R2T2)，代码 Apache-2.0，权重使用上游独立 MODEL_LICENSE。源码归属见 `deploy/R2T2-NOTICE`。

## 验证

```bash
python -m unittest discover -s tests
python -m scripts.benchmark.realtime_smoke zh.wav en.wav \
  --url http://127.0.0.1:4174 --concurrency 4 \
  --output benchmark_results/r2t2-acceptance.json
```

验收脚本比较单路与多路文本，并测试双向语言切换、长录音、满载拒绝与断线后名额回收。报告分别记录首个文字时间、帧处理延迟和结束补齐延迟。贪心解码在不同 GPU batch 下也可能产生少量文字差异：脚本记录是否逐字一致，要求归一化字符编辑距离不超过 5%，并且比其他输入的结果更接近本路基线；会话状态隔离另外由回归测试覆盖。这不是语料级准确率评测。
