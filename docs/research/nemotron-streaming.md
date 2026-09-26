# Nemotron 实时说话人标注可行性

日期：2026-09-26。结论：可以在现有 R2T2 实时转写旁增加“谁在说话”的时间轴，但仅接入 Nemotron 不会让 R2T2 获得重叠人声的分别识别能力。本次交付实现离线替换；本文是后续实时集成分析，不代表实时接口已提供说话人标签。

## 固定版本与证据

- 权重：`nvidia/Nemotron-3-Diarization`，revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`。
- Transformers：`27166ea03f12c940f23176a904ab1d2ff1a3dcbb`。已检查服务器实验保存的对应实现、模型配置及官方模型卡；下列链接固定版本，避免将后续开发分支能力混入当前结论。
- [模型卡](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md)、[处理器源码](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/src/transformers/models/nemotron3_diarization/processing_nemotron3_diarization.py)、[推理与缓存源码](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py)。

## 原生支持与延迟

模型用 AOSC 说话人缓存保存较早的身份信息，FIFO 保存最近上下文；每个流式调用接收上次返回的 `speaker_cache`。一个连接保留自己的缓存，断开后释放；不同连接可共享权重，不能共享缓存。离线调用不传缓存，模型在一次 forward 内自行分块，保持整个录音的身份连续。[推理源码](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L701)

| 模式 | 主块 / 右上下文，80 ms 编码帧 | 输入缓冲延迟 |
| --- | --- | --- |
| 离线默认 | 340 / 40 | 30.4 秒 |
| 流式默认 | 9 / 4 | 1.04 秒 |
| 很低延迟 | 6 / 2 | 0.64 秒 |
| 极低延迟 | 3 / 1 | 0.32 秒 |

这些是等待音频的缓冲量，不包含推理、排队、网络以及字词对齐耗时。0.32 秒不能表述为应用的端到端延迟。模型卡还提及更激进的配置，当前固定处理器公开支持的是表中三种流式模式，应先使用 1.04 秒模式评估身份稳定性，再比较更小缓冲。[模型卡流式配置](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md#streaming-inference)

输入为 16 kHz 单声道。输出 logits 的时间间隔是 **10 ms**，需要 sigmoid 才是活跃概率；不能把 80 ms 编码帧误当成输出步长。流式首块和后续块的窗函数中心不同，所需采样数也不同，应使用处理器的 `num_samples_first_audio_chunk`、`num_samples_per_audio_chunk` 和 `audio_chunk_start()`；最后一块使用结束标记，输出此前保留的右上下文。照搬 WebSocket 数据包边界或简单按固定毫秒切块会造成时轴错位。[处理器源码](https://github.com/huggingface/transformers/blob/27166ea03f12c940f23176a904ab1d2ff1a3dcbb/src/transformers/models/nemotron3_diarization/processing_nemotron3_diarization.py)

## 与本工程 R2T2 的接入边界

现有 `app/services/realtime/server.py` 发送 `delta`、`audio_ms`、`inference_ms`、`done`，结束时补充 `text`。`audio_ms` 表示累计已接收音频长度，不能当作当前文字的精确发音位置；当前增量事件没有字词时间戳。因此，按“此刻最活跃的说话人”给最新文本挂标签，会把 ASR 延迟误认为说话人切换，并在插话附近错配。

推荐分两步：

1. 在同一 PCM 输入旁维护 Nemotron 会话，提供独立的说话人活跃区间或当前说话人提示，不改变 R2T2 识别过程。这一步可以支持发言指示、轮次观察和重叠提醒；标签需明确属于录音内的匿名身份。
2. 对已经稳定的短文本窗口做时间对齐，再使用离线相同的归属规则输出可修订或已确认的说话人文本。需要设计窗口关闭和跨窗边界，不能把不断增长的全文反复送入 ForcedAligner。此步骤会引入额外延迟和 GPU 调度成本，应实测后确定刷新频率。

上述是基于当前协议的工程建议。Nemotron 不输出字词文本或词时间戳，不能直接替代 ForcedAligner。也不应按每个说话人的活跃区间重复向 R2T2 提交混合音频，这会重现重复文字和误归属问题。

## 重叠发言与能力上限

最多八个说话人；标签是会话内身份，不是实名，也不能保证跨连接保持编号。缓存结构有助于长时间不发言后身份保持，但不能替代真实长录音验证。更低延迟模式的准确率也不能由离线效果直接推断。[模型卡](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md)

官方另有与 Multitalker Parakeet、Nemotron 3.5 ASR 的多说话人耦合方案，ASR 接受说话人活动条件或遮罩。这些不等于 R2T2 的现有接口也支持相同条件输入。若要求同时完整识别重叠双方，属于额外的多说话人 ASR 或音源分离工作；本次离线关联在证据不足时使用未知归属，不复制同一句话给两个人。[官方 ASR 集成指南](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/ASR_INTEGRATION_GUIDE.md)

## 后续实时实验验收

复用同一 300 秒和完整约 32 分钟音频，按实时速度回放三种缓冲模式，记录首次标签和轮次确认延迟、长间隔返回者的标签、短插话、重叠区间、首次标签是否需要修订。为身份和边界准确率人工标注一小段样本，不能用标签数量代替准确率。再并发四路 R2T2，并叠加离线任务，测 p50/p95 延迟、队列长度、GPU 峰值内存；覆盖断连、尾块冲刷和缓存释放。现有离线单次耗时不能推算这组实时指标。

当前离线段落策略会吸收不足 2 秒、随后回到同一主讲者的短暂插话。实时若沿用该体验，需要先暂存候选身份，观察是否持续或是否回到主讲者，再确认切换；这会增加说话人标签的确认延迟，不能把离线使用的后续上下文当成实时已经可用的信息。R2T2 文字增量可以继续即时输出，身份标签单独延迟确认。
