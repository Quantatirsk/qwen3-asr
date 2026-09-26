# CUDA 部署

标准入口为 `docker-compose.yml` 与 `Dockerfile.gpu`。镜像从 CUDA 12.8 开发镜像构建，使用 `uv.lock` 固定 Python 依赖，不依赖已有应用镜像。运行平台为 Linux x86_64，使用 NVIDIA GPU；不提供其他推理后端。

## 启动和生命周期

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f asr
```

`start.py` 先检查 CUDA、准备全部模型，再依次启动私有实时引擎和公共 API。公共 API 持有离线 R2T2 与强制对齐引擎，私有实时引擎只监听容器内 `127.0.0.1:8001`；外部只暴露 8000。任一进程异常退出，启动器关闭全部子进程；停止时先关闭公共入口，再释放实时模型。

就绪检查同时检查两个进程的模型状态。模型下载、加载和首次编译需要时间，镜像健康检查给予 600 秒启动期。大型模型下载较慢时，建议提前准备模型缓存。

## 显存与并发

| 配置 | 默认值 | 用途 |
| --- | --- | --- |
| `ASR_PORT` | `4174` | 宿主机 HTTP 端口 |
| `ASR_GPU` | `0` | 宿主机 GPU 编号 |
| `API_KEY` | 空 | 公共接口 Bearer 鉴权 |
| `R2T2_INTERNAL_TOKEN` | 空 | 容器内实时接口鉴权 |
| `R2T2_GPU_MEMORY_UTILIZATION` | `0.30` | 实时 R2T2 显存比例 |
| `R2T2_OFFLINE_GPU_MEMORY_UTILIZATION` | `0.30` | 离线 R2T2 显存比例 |
| `FORCED_ALIGNER_GPU_MEMORY_UTILIZATION` | `0.15` | 强制对齐显存比例 |
| `R2T2_MAX_SESSIONS` | `4` | 实时会话上限 |
| `R2T2_ENFORCE_EAGER` | `0` | 实时引擎禁用 CUDA graph 开关 |
| `R2T2_OFFLINE_MAX_NUM_SEQS` | `4` | 离线 vLLM 同时调度序列数 |
| `R2T2_OFFLINE_MAX_MODEL_LEN` | `16384` | 离线上下文长度上限 |
| `R2T2_OFFLINE_ENFORCE_EAGER` | `1` | 离线引擎禁用 CUDA graph 开关 |

三个引擎的显存比例都相对于整张 GPU；需要为 CAM++、CUDA 上下文和其他进程留余量。比例总和小于 1 并不保证可运行，各引擎还必须装得下权重和工作空间；按目标显卡进行加载、长音频和并发验收。实时与离线同时推理会竞争计算资源，不能仅通过提高会话数保证实时性。

## 模型缓存

默认首次启动下载全部模型，持久化目录如下：

- `models/huggingface`：固定版本 R2T2 和强制对齐模型。
- `models/modelscope`：FSMN VAD、CAM++。
- `.cache/vllm`：vLLM 编译缓存。

联网的 Linux 开发环境可提前下载或导出：

```bash
uv sync --frozen
./scripts/prepare-models.sh
./scripts/prepare-models.sh --export-dir /tmp/r2t2-models
```

导出目录包含 `huggingface/` 和 `modelscope/`；复制为部署目录下的 `models/`，再设置 `HF_HUB_OFFLINE=1` 启动。离线模式缺少任何必需模型时启动失败，不会改用其他模型。

`HF_ENDPOINT` 可配置 Hugging Face 镜像。两种识别模式使用相同 R2T2 revision：`185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`。

## 反向代理

浏览器录音需要 HTTPS 或 localhost；TLS 由外部反向代理提供。代理应允许 WebSocket Upgrade，并为离线长音频请求设置足够的请求体大小和超时时间。容器内没有额外的 nginx 或多 GPU 自动复制层。

## 验收

```bash
docker compose ps
curl http://localhost:4174/stream/v1/asr/health
curl http://localhost:4174/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=confucius4-r2t2 \
  -F word_timestamps=true \
  -F response_format=verbose_json
```

开启鉴权时添加 `Authorization: Bearer ...`。对同一份真实录音检查文本、说话人分配、时间戳以及同时运行实时会话时的延迟；不应把实时文字作为离线识别输入。
