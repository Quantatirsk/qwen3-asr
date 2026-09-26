# llama.cpp 纯 CPU 实验复现

只操作 `/Users/quant/.cache/r2t2-poc/llama-*`；无需安装 CUDA/vLLM，不改部署中的服务。当前机器为 Apple M5 Pro / 64 GiB，原生 C++ 扩展未修改，使用 Accelerate，编译和运行均禁用 GPU。

## 固定来源

- [R2T2 源码](https://github.com/netease-youdao/Confucius4-R2T2/tree/26d55a54ce5670cff9947a167d8ed95d569fd4d9)：`26d55a54ce5670cff9947a167d8ed95d569fd4d9`。
- [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/ad6c66839af3c5646fba8c6c2e2087a1e4e38948)：`ad6c66839af3c5646fba8c6c2e2087a1e4e38948`。
- [官方 GGUF](https://huggingface.co/netease-youdao/Confucius4-R2T2-GGUF/tree/86ff0251cb9f456b63aeef5f80137f104e22869a)：`86ff0251cb9f456b63aeef5f80137f104e22869a`。
- HF tokenizer/processor：`netease-youdao/Confucius4-R2T2` 的 `185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`，只下载 JSON/词表。

| 文件 | 本地缓存名称 | 字节数 | 实际 SHA256（与官方 LFS oid 相同） |
| --- | --- | ---: | --- |
| Confucius4-R2T2-Q8_0.gguf | llama-decoder-Q8_0.gguf | 1834422208 | 151097e43957a19984ea7de66e8144ce69b95039eb31c93da4f58db367e455c3 |
| mmproj-Confucius4-R2T2-f16.gguf | llama-mmproj-f16.gguf | 641773984 | 0057b28b8814e431a28e3d1002343eef2d1676bbf8b0625351ae4550ac4e595d |

## 构建命令

源码解压到 `llama-src`、`llama-cpp` 后：

```bash
uv venv --python 3.12 "$HOME/.cache/r2t2-poc/llama-venv"
uv pip install --python "$HOME/.cache/r2t2-poc/llama-venv/bin/python" cmake==4.4.3 pybind11==3.1.0 numpy==2.5.3 qwen-asr==0.0.6
"$HOME/.cache/r2t2-poc/llama-venv/bin/cmake" \
  -S "$HOME/.cache/r2t2-poc/llama-src/r2t2_llama" \
  -B "$HOME/.cache/r2t2-poc/llama-build" \
  -DLLAMA_CPP_DIR="$HOME/.cache/r2t2-poc/llama-cpp" \
  -DPython_EXECUTABLE="$HOME/.cache/r2t2-poc/llama-venv/bin/python" \
  -Dpybind11_DIR="$HOME/.cache/r2t2-poc/llama-venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11" \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
"$HOME/.cache/r2t2-poc/llama-venv/bin/cmake" \
  --build "$HOME/.cache/r2t2-poc/llama-build" --target qwen3asr_native -j 6
```

宿主未安装 cmake，隔离环境安装解决。官方 CMake 在 macOS 生成 `.dylib` 和 `qwen3asr_native.cpython-312-darwin.so`，BUILD_RPATH 可找到构建目录，无需改源码。官方 release 是 Linux CUDA 产物，不适用，但本机源码构建成功。安装 `qwen-asr==0.0.6` 而非带 `[vllm]` 的完整 R2T2 包，使官方状态机和 HF processor 在 macOS 可导入。完整版本快照保存在缓存 `llama-requirements.txt`，构建日志为 `llama-configure.log`、`llama-build.log`。

## 运行命令与计量

```bash
"$HOME/.cache/r2t2-poc/llama-venv/bin/python" experiments/r2t2_cpu/llama_probe.py \
  --samples zh en mixed silence quiet \
  --output experiments/r2t2_cpu/llama-results.json
"$HOME/.cache/r2t2-poc/llama-venv/bin/python" experiments/r2t2_cpu/llama_probe.py \
  --samples zh en mixed silence quiet --mode stream --step-ms 2000 \
  --output experiments/r2t2_cpu/llama-stream-2s.json
"$HOME/.cache/r2t2-poc/llama-venv/bin/python" experiments/r2t2_cpu/llama_probe.py \
  --samples zh en mixed silence quiet long zh en mixed silence quiet long --mode offline \
  --output experiments/r2t2_cpu/llama-offline-warm.json
```

全部使用 8 threads、4096 context、2048 batch、`use_gpu=False`、`n_gpu_layers=0`。离线与流式共用一个 native 模型实例。流式直接调用官方 `init_streaming_state` / `streaming_transcribe` / `finish_streaming_transcribe`，沿用示例的首块 lookahead、单 token 回退和 token budget 更新。把预算放在 `sampling_params.max_tokens`，而不是传可选 `max_new_tokens`，避免后者触发 vLLM import；官方 llama adapter 读取的数值相同。固定源码示例的 `total_new_asr_tokens` 始终为空，因此其中文额外倍增分支从不触发，本 PoC 保持实际行为。

结果中的 RTF 是离线重放计算秒数 / 音频秒数，未插入真实录音等待。`first_text_compute_seconds` 不是端到端采集延迟。每个 `calls` 记录输入截至时间、单次计算耗时、当前文字和确认文字。RSS 是 macOS `ru_maxrss` 字节数，包含 Python/torch/HF 和 native 模型，属于进程累计峰值；不是独立的 GGUF 权重体积。音频 SHA256 和来源见共享 corpus manifest，GPU 模型输出仅作跨后端文字差异参照，不能作为人工准确率标注。
