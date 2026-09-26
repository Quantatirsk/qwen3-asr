# 开发脚本

运行环境为 Linux x86_64 / NVIDIA CUDA，依赖统一由根目录 `pyproject.toml` 和 `uv.lock` 管理。

```bash
./scripts/sync_gpu_env.sh
./scripts/prepare-models.sh
uv run python start.py
```

`prepare-models.sh` 下载当前服务的全部模型到 `models/`，可传 `--export-dir /path/to/models` 导出供离线部署。`build.sh` 构建本地 CUDA 镜像，默认标签为 `local/r2t2-asr:dev`，可用 `IMAGE_TAG` 覆盖。

实时音频回放与并发验收见 [benchmark/README.md](benchmark/README.md)。`analyze_audio_rms.py` 可用于检查录音能量分布；`retire_realtime_models.py` 是独立的历史缓存清理工具，默认只列出候选目录，不参与服务启动。
