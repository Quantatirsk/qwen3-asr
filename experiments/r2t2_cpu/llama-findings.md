# macOS 原生 llama.cpp CPU 结果

2026-09-26，M5 Pro / 64 GiB，固定官方 R2T2 源码与 GGUF。构建与复现说明见 `llama-build.md`，原始记录见 `llama-*.json`。

**可运行。** 不必改 R2T2 C++ 数学内核，原始官方扩展在 macOS 编译成功；Q8 decoder + F16 audio encoder 使用同一常驻模型完成离线和流式。编译关闭 Metal/CUDA，运行关闭 encoder/KV GPU offload 且 decoder GPU layers=0，日志仅 CPU/BLAS。8 threads、4096 context、2048 batch；首次加载计时约 1.49 秒（不包含 Python 首次导入）。

## 短样本性能

单次顺序测试；数字仅描述本机这些音频，不作为全量准确率或吞吐保证。

| 样本 | 音频秒 | 离线 RTF | 160ms 流式 RTF | 640ms 流式 RTF | 2s 流式 RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 中文 | 6.74 | 0.099 | 1.750 | 0.485 | 0.308 |
| 英文 | 7.00 | 0.104 | 1.905 | 0.741 | 0.609 |
| 中英拼接 | 14.24 | 0.079 | 3.463 | 1.707 | 0.668 |
| 静音 | 5.00 | 0.094 | 1.713 | 未测 | 0.256 |
| 低音量中文 | 6.74 | 0.108 | 2.413 | 未测 | 0.389 |

本表每档 lookahead=该档 chunk，首块分别需要 320ms / 1280ms / 4s 音频；大块速度以响应延迟为代价。160ms 流式即使短样本也慢于音频生成；640ms 中文可实时，14秒混合样本不行；2s 样本内可实时。官方状态机每步重新编码累计音频，不能从短片段外推长会话。无滚动窗口的本次短流式实验没有验证长会话服务策略。

160ms 流式单步 P95：中文 0.441s、英文 0.487s、混合 0.991s；640ms 为 0.431s、0.863s、1.579s；2s 为 0.667s、1.689s、1.794s。各采样行完整延迟及文本在 JSON 的 `calls`，便于检验首包及返修。首次完整矩阵进程峰 RSS 约 5.38 GiB，包括 HF/torch 等 Python 依赖；两个 GGUF 文件共 2.31 GiB。

## 质量边界

- 中文离线与流式均保留“之前有顾客自己带酒水，也没加收钱或者不让喝”。低音量样本也完整，不能仅根据上游已知起始轻音问题断言本实现必坏。
- 英文主要内容完整，结尾多出 `his`；样本自身截断，未人工标注，不能把 GPU 结果直接视为真值。
- **混合音频自动语言离线明显丢掉前面的中文**：原始输出 `language English<asr_text>Before, he wasn't even that big when I started listening to him, but and his`。这是此配置的真实失败样本，应阻止凭短中文成功就宣称离线可全面替换。
- 同一混合文件经官方流式状态机能够保留中文与英文；三档均如此，说明流式前缀与完整离线质量不能互相代替。
- 静音离线仅返回 `language None<asr_text>` 元数据，流式文本为空。服务必须解析协议头，不能把 metadata 原样展示。

## 最小接入选择

如果选择 llama 路径，只需适配当前 shared engine 的单次推理与 tokenize/detokenize 操作；官方原生扩展已证明 CPU 计算路径成立。HF tokenizer 能保留官方回退边界，CPU 使用显式低频 chunk/滚动窗口，离线仍独立重识别。不要安装含 vLLM 的整个官方默认包，也不要把 GGUF 塞给 safetensors Rust loader。

当前优先保留为独立 PoC：部署需要同时解决混合离线缺字、长期累计编码成本、macOS 动态库分发及依赖缩减。若现有 Rust safetensors 内核质量与性能合格，直接恢复它更少一套推理依赖。CPU/GPU 可共享 R2T2 流式控制和模型语义，但不能据此要求相同 chunk 延迟。

## 重复与诊断补充

同一进程两轮顺序离线：中文 RTF 0.091 / 0.112，英文 0.098 / 0.149，混合 0.072 / 0.126；混合漏中文输出逐字稳定。44.22 秒拼接文件 RTF 0.078 / 0.111，峰值 RSS 约 5.39 GiB；输出包含两轮中英内容，而输入有三轮，故长样本也不能仅凭速度通过验收。第二轮没有变快，不能把它包装为稳定吞吐提升。

同一14.24秒混合文件额外强制 `Chinese`，离线 RTF 0.094，输出恢复中文与英文：此前缺中文与自动语言/解码路径相关，并非权重完全不兼容。但不能全局强制中文来掩盖多语言任务。原始数据见 `llama-mixed-forced.json`，诊断命令如下。

```bash
"$HOME/.cache/r2t2-poc/llama-venv/bin/python" experiments/r2t2_cpu/llama_probe.py \
  --samples mixed --language Chinese --mode offline \
  --output experiments/r2t2_cpu/llama-mixed-forced.json
```
