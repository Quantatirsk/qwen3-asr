# R2T2 验收脚本

服务启动后，用自己的中文、英文录音回放验证文字、延迟与会话隔离：

```bash
uv run python -m scripts.benchmark.realtime_smoke zh.wav en.wav \
  --url http://127.0.0.1:4174 --concurrency 4 \
  --output benchmark_results/r2t2-acceptance.json
```

脚本需要 FFmpeg；启用鉴权时设置 `API_KEY` 或传入 `--api-key`。它检查单路与并发文字差异、语言切换、长音频、满载拒绝和断线后的名额回收；结果不能替代语料级准确率评测。

浏览器录音验收：

```bash
npm install --prefix .cache/browser-tests playwright
./.cache/browser-tests/node_modules/.bin/playwright install chromium
NODE_PATH="$PWD/.cache/browser-tests/node_modules" \
  TEST_WAV="$PWD/recording.wav" \
  node scripts/benchmark/realtime_browser.cjs
```

原有通用压测工具保留 ASR 客户端，可指定端口和并发级别：

```bash
uv run python -m scripts.benchmark.run --test-type asr \
  --port 4174 --audio-file recording.wav --concurrency 1 2 4
```

通用报告工具额外需要 `matplotlib`，不属于服务运行依赖。离线转写验收直接调用 `/v1/audio/transcriptions`，检查说话人、识别文本、时间戳及真实 GPU 资源占用。
