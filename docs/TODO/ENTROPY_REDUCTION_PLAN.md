# 熵减执行清单

本文档记录本轮已执行的熵减工作。目标是移除不可达路径、兼容占位、隐藏 fallback、重复请求流水线和无用依赖。

## 范围

- 主范围：`app/`、根运行配置、公开运行文档。
- 不处理：`vendor/qwenasr` 内部实现、仅 benchmark 使用且不阻塞主链路的代码。
- 原则：开发中项目不保留废弃接口、旧字段、兼容层或 fallback 逻辑。

## P0

- [x] 移除 Qwen3 `transformers` 后端残留路径。
  - 删除后端选择里的 `"transformers"` fallback。
  - 删除仅服务该路径的 batch/segment 转换死代码。
  - 不支持的设备显式失败。
- [x] 启动预加载改为 fail-fast。
  - 模型完整性检查或预加载失败时停止 worker。
  - 不再静默降级到首次请求加载。
- [x] 移除被忽略的离线模型兼容参数。
  - 删除 REST `model_id` 兼容处理。
  - 删除 OpenAI transcription `model` 兼容处理。
  - 运行时模型选择统一由部署计划和 `QWEN3_ASR_MODEL` 控制。

## P1

- [x] 抽出共享离线转写服务。
  - API 层只处理协议输入和响应格式。
  - 音频准备、`OfflineASRRequest`、runtime 调用和清理边界集中到服务层。
- [x] 合并音频字节处理逻辑。
  - `process_from_request` 和 `process_upload_file` 共享私有 byte 处理 helper。

## P2

- [x] 将 `.tsscale` sidecar 隐式耦合改为显式结构化元数据。
- [x] 拆分 WebSocket 路由和 Qwen3 协议服务。
  - Qwen3 websocket 状态机移出 API route。
  - route 模块仅保留端点注册和 service delegation。
- [x] 删除首轮发现的无用 helper。
- [x] 审计并移除无用直接依赖。
  - 根环境和 CPU 环境移除直接依赖 `pydub`、`httpx`。
  - 保留 ModelScope/FunASR 动态 runtime 依赖。
- [x] 继续压缩阿里协议 WebSocket service。
  - 删除大段注释、未使用状态、未使用参数和死函数。
  - 合并重复响应构造。
  - 删除重复音频转换。

## 验证

- [x] `uv run python -m py_compile $(find app -name '*.py' -not -path '*/__pycache__/*') start.py`
- [x] `uvx pyright`
- [x] 变更模块 import smoke check。
- [x] 手工 API smoke plan 已记录：
  - `/stream/v1/asr`
  - `/v1/audio/transcriptions`
  - `/ws/v1/asr/funasr`
  - `/ws/v1/asr/qwen`

## 手工 Smoke Plan

启动服务并准备模型后执行：

1. `POST /stream/v1/asr`，使用小 WAV request body，确认返回 `result`、`segments`、`duration`、`processing_time`。
2. `POST /v1/audio/transcriptions`，使用 multipart `file` 和 `response_format=verbose_json`，确认返回 OpenAI 风格 `text` 和 `segments`。
3. 连接 `/ws/v1/asr/funasr`，发送阿里兼容 start/audio/stop 消息，确认 sentence 事件仍正常返回。
4. 连接 `/ws/v1/asr/qwen`，发送 start/audio/stop 消息，确认 partial/final 事件仍正常返回。
