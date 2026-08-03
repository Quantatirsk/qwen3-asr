# Single-container Operations

`stage-qwen-model.sh` copies a customer-provided Hugging Face snapshot into `/workspace/hf_models/Qwen3-ASR-1.7B`. It dereferences cache symlinks, reports progress, validates the copied files, and publishes the destination atomically.

`start-ascend-services.sh` starts Ascend vLLM, waits for its health endpoint, then starts the API from the isolated CPU environment. It keeps both processes under one foreground shell and cleans them up together.

```bash
/workspace/qwen3-asr/scripts/stage-qwen-model.sh /path/to/Qwen3-ASR-1.7B
/workspace/qwen3-asr/scripts/start-ascend-services.sh
```
