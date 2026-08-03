# Ascend 910B 完整部署可行性研究

> 研究快照：2026-08-03
>
> 研究对象：当前仓库 `qwen3-asr` 服务在 Ascend 910B / Atlas A2 上的完整部署
>
> 证据范围：华为 Ascend/CANN/torch_npu/MindIE/ATC/AOE 官方资料与源码，以及 Qwen、FunASR、ModelScope、PyTorch、ONNX Runtime、vLLM Ascend 等上游官方资料与源码
>
> 验证边界：本文没有运行 NPU 测试；所有性能、精度和稳定性结论都必须通过目标硬件 PoC

> 后续部署决策：客户平台只允许启动一个镜像，并只向维护者提供容器 shell。当前实现因此改为单容器、双 Python 环境：基础环境运行 Ascend vLLM，隔离的 CPU venv 运行 API 与辅助模型，两者通过 loopback 通信。本文后续出现的双容器建议和旧仓库基线仅保留为研究过程记录，不代表当前交付方式。

## 1. 结论摘要

**结论：部署可行，但不是现有 CUDA 镜像的原位替换。**

最有力的一手证据是 vLLM Ascend 已提供
[Qwen3-ASR-1.7B 官方 Ascend 教程](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-ASR-1.7B.html)：
BF16 权重可部署在一张 64 GB Ascend 910B 上，教程给出了 A2 镜像、设备挂载、服务启动、准确率与性能验证方法。由此可以把 Qwen3-ASR-1.7B 核心识别能力判定为**官方支持**，而不是仅靠源码猜测。

但当前仓库仍存在四组确定的迁移工作：

1. 当前 Linux 依赖和镜像固定为 CUDA 12.8、NVIDIA PyTorch、`vllm[audio]==0.19.0`、标准 Triton/FlashInfer；这些组件不能直接用于 910B，必须使用一套完整匹配的 vLLM Ascend/CANN/torch_npu 版本组合。
2. 当前设备自动探测只识别 CUDA，Qwen 路由只接受 `cuda` 或 CPU Rust；即使配置 `DEVICE=npu:0`，Qwen 路由也会拒绝启动。
3. FunASR 当前锁定 `1.3.1`，该安装版本没有 NPU 可用性处理；FunASR `v1.3.2` 已有正式 NPU 入口，但 Paraformer、FSMN VAD 和标点仍需按本项目语料与流式语义做 PoC。
4. Qwen3 Forced Aligner、项目自定义实时流式语义没有 910B 官方端到端验证；CAM++ 更早被当前 ModelScope pipeline 的设备校验阻断，不能随核心 Qwen3-ASR 一并宣称支持。

**当前路线：**从官方 `vllm-ascend` A2 镜像派生唯一交付镜像。系统 Python 环境只运行 vLLM Ascend，独立 CPU venv 运行 FastAPI、音频前处理和辅助模型，两个进程通过 `127.0.0.1` 通信。该方案满足单镜像限制，同时避免 CPU PyTorch 覆盖基础镜像中的 `torch_npu`。原双容器方案在平台允许多工作负载时隔离性更强，但已不符合本客户的硬约束。

**生产发布状态：有条件可行。**Qwen3-ASR 的直接模型证据来自 `vllm-ascend 0.22.1rc1 + CANN 9.0.0`，但 CANN 官方下载页把 9.0.0 社区版定义为面向开发者的新特性 PoC 版本。华为同时公开了 CANN 9.0.0 商用版文档，但公开文档不等于目标 Atlas SKU 已获得相应软件包、HDK 配套与生产支持权益。生产 Go 的前置条件是客户取得与所选 vLLM 发布行精确匹配的商用 CANN 9.0.x/HDK/driver/firmware 支持包，并通过本文 PoC 门槛；社区 PoC 镜像不能直接批准为生产基线。

这里需要区分两个客户口径：**整套系统部署在 910B 服务器上**是可行的，CPU 仍承载音频处理、聚类和部分辅助模型；**所有模型推理都必须运行在 NPU 上**目前不可承诺，因为现有 CAM++ ModelScope pipeline 明确拒绝 `npu` 设备，其他辅助模型也缺少逐模型 910B 证明。

**当前判定：**

| 范围 | 判定 | 说明 |
|---|---|---|
| Qwen3-ASR-1.7B BF16，单张 64 GB 910B | 官方支持 | vLLM Ascend 有专门教程和镜像启动命令 |
| 当前服务原样运行 | 不可行，需改造 | 路由、依赖锁、Dockerfile、Compose、监控均绑定 CUDA |
| Qwen3-ASR-0.6B | 需要 PoC | Qwen 官方发布该模型，但 vLLM Ascend 910B 教程只明确验证 1.7B |
| Paraformer/FSMN/PUNC 使用 `torch_npu` | 源码推断可行，需 PoC | FunASR 新版本识别 `npu:0`，没有逐模型 910B 官方测试矩阵 |
| Forced Aligner/词级时间戳 | 需要 PoC | 上游 vLLM 支持该架构，但 Ascend Qwen3-ASR 教程未验证 aligner |
| 当前 CAM++/ModelScope pipeline 直接使用 NPU | 不可行，首期必须 CPU | ModelScope 1.34.0 的设备校验明确拒绝 `npu`；升级到 1.37.0 仍未解除 |
| ONNX Runtime CANN EP | 需要 PoC，不建议作首期主线 | 官方仍标为 preview，算子表明显不足以覆盖任意 ASR/说话人模型 |
| ATC 转 OM | 源码/工具链可行，逐图验证 | 官方支持 ONNX 转 OM 和动态 shape profile，但只支持 CANN 算子规格内的图 |
| 标准 Triton/xformers/flash-attn | 不可行，需替换或移除 | 官方硬件范围为 CUDA/ROCm；Ascend 使用 Triton-Ascend 或 CANN/ATB 算子 |

## 2. 判定口径

本文使用以下四级状态，避免把“框架能识别 NPU”扩大解释为“业务模型已被支持”。

| 状态 | 默认置信度 | 含义 |
|---|---|---|
| **官方支持** | 高 | 官方文档或官方仓库明确列出对应硬件、模型和版本组合，并提供可执行示例 |
| **源码推断可行** | 中 | 官方源码存在通用 NPU 路径，模型结构没有发现确定阻断，但没有对应模型/版本的官方验证记录 |
| **需要 PoC** | 低 | 版本、算子覆盖、动态 shape、精度、内存或并发行为至少有一项未由官方资料闭合 |
| **不可行/需替换** | 高 | 当前构件明确绑定 CUDA/ROCm/NVIDIA Runtime，或当前项目代码显式拒绝 NPU |

置信度表示“判定依据的确定性”，不是组件质量评分。例如“当前 CUDA 镜像不可直接运行”的置信度为高；“Forced Aligner 可在 910B 运行”的置信度为低。

“完整部署”按本仓库实际能力定义，包括：Qwen3-ASR 0.6B/1.7B、Qwen forced aligner、Paraformer 实时识别、FSMN VAD、离线/实时标点、CAM++ 说话人分离、音频解码/重采样、REST/WebSocket API、Docker/Kubernetes、批处理、并发、精度和性能。

## 3. 当前仓库基线与确定差距

### 3.1 依赖和镜像

- 项目要求 Python `>=3.10,<3.13`，该范围与 vLLM Ascend 当前发布矩阵相容；但 Linux 依赖固定到 CUDA 索引的 PyTorch 2.10.0，并固定 `vllm[audio]==0.19.0`，见 [`pyproject.toml`](../../pyproject.toml)。
- GPU 镜像从 `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` 构建，安装 NVIDIA apt 源、CUDA nvcc，并保留 nvcc 给 FlashInfer JIT，见 [`Dockerfile.gpu`](../../Dockerfile.gpu)。这属于**不可行/需替换**，不能通过换基础镜像名称解决。
- Compose 使用 `runtime: nvidia` 和 `CUDA_VISIBLE_DEVICES`，见 [`docker-compose.yml`](../../docker-compose.yml)。910B 必须改用 Ascend Docker Runtime 或显式挂载 Ascend 设备节点。

### 3.2 设备与后端路由

- `auto` 只调用 `torch.cuda.is_available()`，否则返回 CPU；虽然注释列出 `npu:0`，并没有 NPU 自动发现、容量查询或监控，见 [`app/core/device.py`](../../app/core/device.py)。
- Qwen 运行族只接受 `cuda` 进入 vLLM，或 CPU 进入 Rust，其他设备直接抛错，见 [`app/services/asr/runtime/router.py`](../../app/services/asr/runtime/router.py) 和 [`app/services/asr/qwen3_engine.py`](../../app/services/asr/qwen3_engine.py)。这是当前代码层面的确定阻断。
- 内存选择、健康信息和模型自动选择均读取 `torch.cuda`，910B 上会得到 CPU/0 GB 的错误结论。
- 当前离线 Qwen 请求被进程内 `asyncio.Lock` 串行化，外层并发信号量为 8；这会遮蔽 vLLM 的跨请求调度能力，见 [`app/services/asr/runtime/router.py`](../../app/services/asr/runtime/router.py)。

### 3.3 模型组件

仓库声明的核心模型与辅助资产见 [`app/services/asr/models.json`](../../app/services/asr/models.json) 和 [`app/services/asr/model_capabilities.py`](../../app/services/asr/model_capabilities.py)：

| 组件 | 当前实现 | Ascend 初判 |
|---|---|---|
| Qwen3-ASR-1.7B | 内嵌 upstream vLLM | 官方支持核心 ASR，但应用适配仍需实现 |
| Qwen3-ASR-0.6B | 内嵌 upstream vLLM | 需要 PoC |
| Qwen3-ForcedAligner-0.6B | 第二个 vLLM pooling runner | 需要 PoC，且与主模型争用 HBM |
| Paraformer online | FunASR `AutoModel` | 升级 FunASR 后源码推断可行，需 PoC |
| FSMN VAD | FunASR `AutoModel` | 需要 PoC；首期 CPU |
| CT-Transformer PUNC | FunASR `AutoModel` | 需要 PoC；首期 CPU |
| CAM++ SV/diarization | ModelScope pipeline + PyTorch/ONNX | 当前 NPU 路径被 ModelScope 设备校验阻断；首期 CPU，二期重写后再 PoC |
| 音频解码与重采样 | librosa/soundfile/FFmpeg subprocess | CPU 路径可保留，不需要 NPU 适配 |

音频规范化当前通过 librosa、soundfile 和 FFmpeg subprocess 完成，见 [`app/utils/audio.py`](../../app/utils/audio.py)。这些是宿主 CPU 工作，功能上不阻塞 Ascend，但在高并发长音频场景可能先于 NPU 成为吞吐瓶颈。

## 4. 推荐参考栈

### 4.1 以官方 Qwen3-ASR 教程为 PoC 冻结点

建议首轮 PoC 不手工拼装最新包，而是从 Qwen3-ASR 教程使用的
[`quay.io/ascend/vllm-ascend:v0.22.1rc1`](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-ASR-1.7B.html)
派生应用镜像，并按镜像 digest 冻结。

vLLM Ascend 的
[发布兼容矩阵](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)
把这组版本定义为一个整体：

| 层 | PoC 冻结版本 | 状态与理由 |
|---|---|---|
| 硬件 | Ascend 910B 64 GB / Atlas A2 | Qwen3-ASR 教程明确支持 |
| Python | `>=3.10,<3.13` | 官方矩阵支持；与项目现有范围一致 |
| vLLM Ascend | `0.22.1rc1` | Qwen3-ASR 教程给出的镜像版本；RC 是生产风险 |
| vLLM | `0.22.1` | 必须与 vLLM Ascend 同发布行 |
| CANN | `9.0.0` | `0.22.1rc1` 发布矩阵的稳定 CANN |
| PyTorch | `2.10.0` CPU wheel | 与项目主版本一致，但必须移除 CUDA wheel |
| torch_npu | `2.10.0` | `0.22.1rc1` 发布矩阵 |
| Triton-Ascend | `3.2.1` | `0.22.1rc1` 发布矩阵；不能与项目当前标准 Triton 混用 |
| driver/firmware | CANN 9.0.0 对应 HDK 映射 | 具体版本必须按实际 Atlas 产品和官方下载页确定 |

官方明确要求把 vLLM Ascend、vLLM、PyTorch、torch_npu、CANN 和 Triton-Ascend 当成一个兼容集合，不能任意交叉版本；当前
[安装页](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)
的 `main` 目标已是 CANN 9.0.1/torch_npu 2.10.0.post2，与 `0.22.1rc1` 发布行不同，不能混装。

### 4.2 官方版本链冲突与生产决策

官方资料给出了两条各自成立、但目前不能合并成单一生产栈的版本链：

| 用途 | CANN/HDK | PyTorch/torch_npu | Serving | 证据与判定 |
|---|---|---|---|---|
| Qwen3-ASR 直接模型 PoC | CANN `9.0.0`；HDK 按目标 SKU 查询 | `2.10.0/2.10.0` | vLLM/vLLM Ascend `0.22.1/0.22.1rc1` | Qwen3-ASR 教程直接支持 910B 64 GB；但 vLLM Ascend 为 RC，公开可下载的 CANN 9.0.0 社区版是 PoC 发布，**不能直接判定可生产** |
| 当前商用通用 PyTorch 基线 | CANN `8.5.0` + HDK `25.5.x` 代际 | 可选 `2.6.0/2.6.0.post5`、`2.7.1/2.7.1.post2`、`2.8.0/2.8.0.post2`、`2.9.0/2.9.0` | 通用 torch_npu | [CANN 8.5.0 配套表](https://www.hiascend.com/document/detail/zh/canncommercial/850/releasenote/releasenote_0000.html)与 [torch_npu 兼容表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)闭环基础设施，但**没有 Qwen3-ASR 模型级证明** |
| MindIE 产品化基线 | MindIE 2.3.0 对应 CANN `8.5.0`、PTA `7.3.0`、MindCluster `7.3.0`；MindIE 3.0.0 对应 CANN `8.5.1` | 服从各 MindIE 软件包表，不与通用兼容表交叉拼装 | MindIE | [MindIE 2.3.0 配套表](https://www.hiascend.com/document/detail/zh/mindie/230/releasenote/releasenote_0004.html)与 [MindIE 3.0.0 配套表](https://www.hiascend.com/document/detail/zh/mindie/300/releasenote/MindIE/26.0.0/release.md)属于商用产品链；没有找到 Qwen3-ASR 支持项 |

[CANN 社区版下载页](https://www.hiascend.com/en/software/cann/community/)将 9.0.0 描述为面向开发者的新特性 PoC 版本；华为也有独立的 [CANN 9.0.0 商用版文档](https://www.hiascend.com/document/detail/zh/canncommercial/900/releasenote/releasenote_0000.html)。两者并不矛盾：是否可生产取决于实际取得的软件包、许可、目标 SKU 的 HDK 配套和厂商支持，而不是 URL 中存在“商用版”文档。由此得到的决策是：

1. 技术 PoC 使用 Qwen3-ASR 教程的整行版本，证明业务功能、精度与容量；不把该环境直接晋级生产。
2. 生产 Go 必须新增一项合同/交付证据：厂商为实际 Atlas SKU 提供与所选 vLLM 行精确匹配的商用 CANN 9.0.x、HDK、driver/firmware 配套和支持范围。若采用当前 latest 的 CANN 9.0.1/torch_npu 2.10.0.post2 行，则必须整体切换并重跑 PoC，不能只替换 CANN。
3. 若必须在上述证据出现前投产，只能另做“Qwen3-ASR on CANN 8.5.x”对照 PoC；它当前属于无官方模型矩阵支撑的高风险偏离，不能作为默认方案。
4. 不在一个 Python 环境或 NPU worker 中混装 CANN 8.5.x 与 9.0.x。CPU API 容器不安装 CANN；以后迁移 FunASR 时，使用独立 NPU worker 和独立设备，或先验证统一到同一发布行。

### 4.3 备选更新栈

vLLM Ascend `0.23.0rc1` 的发布行使用 CANN 9.0.1、PyTorch 2.10.0、torch_npu 2.10.0.post2、Triton-Ascend 3.2.1；当前
[安装页](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)也把 latest A2 环境锁定到 CANN 9.0.1/torch_npu 2.10.0.post2。其
[发布说明](https://docs.vllm.ai/projects/ascend/en/main/user_guide/release_notes.html)
增加了 310P 的 Qwen3-ASR 支持。910B 首轮 PoC 没有必要为了该变化追新；只有当 `0.22.1rc1` 存在已知缺陷，才整体切换到 `0.23.0rc1`，不能只升级其中一包。

## 5. 宿主、驱动、容器与集群兼容矩阵

### 5.1 宿主 OS 与内核

“910B”不是足以确定 OS/内核的完整 BOM，必须先确认具体产品形态、CPU 架构和厂商交付版本。

| 产品形态 | 官方证据 | 建议 | 状态 |
|---|---|---|---|
| Atlas 800I A2 推理服务器 | MindIE 2.3 的[硬件与 OS 表](https://www.hiascend.com/document/detail/en/mindie/230/envpre/instg/mindie_instg_0001.html)列出 AArch64 Ubuntu 22.04/24.04、openEuler 22.03/24.03 等 | 优先使用厂商交付的 openEuler 22.03 LTS SP4 或 Ubuntu 22.04，并冻结内核 | 官方支持，具体小版本需 BOM 确认 |
| Atlas 800T A2 训练服务器 | MindCluster 7.3 的[产品与 OS 清单](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_002.html)列出 ARM Ubuntu 22.04.4、Linux 6.5.0-18-generic，以及多种 openEuler | 使用清单内的 OS/内核，不在 PoC 中升级通用内核 | 官方支持，具体小版本需 BOM 确认 |
| 未知 910B PCIe/整机组合 | 目前没有产品型号 | 不选择 OS/内核；先取得设备 PN、CPU 架构和官方兼容性查询结果 | 阻塞 |

CANN 8.2 的官方发布说明也明确列出 Atlas 800T A2 对 Ubuntu 22.04.4/6.5 内核、Atlas 800I A2 对 openEuler 22.03 LTS SP4 等新增适配，说明内核必须按产品级矩阵选择，而不是只看发行版大版本，见
[CANN OS 适配说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/releasenote/releasenote_0002.html)。

### 5.2 驱动、固件、CANN

| 层 | 要求 | 状态 |
|---|---|---|
| NPU driver | 安装在宿主机；版本按 CANN 9.0.0 与实际硬件的 HDK 映射获取 | 需要 BOM 确认 |
| firmware | 安装在宿主机；必须与 driver/CANN 映射一致 | 需要 BOM 确认 |
| CANN Toolkit + 910B ops | 容器中使用 vLLM Ascend 镜像内版本；宿主 driver 与其兼容 | 官方支持 |
| 验证 | 宿主 `npu-smi info`、容器设备可见、版本文件与镜像 tag/digest 记录入库 | PoC 必须执行 |

CANN 官方安装说明规定物理机/容器场景只在宿主安装 driver 和 firmware，并要求 driver、firmware 与 CANN 遵循版本映射，见
[NPU Driver and Firmware Installation](https://www.hiascend.com/document/detail/en/canncommercial/850/softwareinst/instg/instg_0005.html)。
本文不能在缺少具体硬件型号时给出一个安全的 driver/firmware 数字；这不是可由通用软件版本推断的字段。

### 5.3 Docker

- Ascend Docker Runtime 是基于 OCI 的 Docker 插件，不修改 Docker Engine，负责把 Ascend NPU 能力适配给容器，见
  [Installing Ascend-Docker](https://www.hiascend.com/document/detail/en/canncommercial/800/softwareinst/instg/instg_0020.html)。
- vLLM Ascend 的 Qwen3-ASR 教程明确挂载 `/dev/davinci*`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc` 以及宿主 driver/DCMI 文件。首轮裸 Docker PoC 应照官方命令执行，不应复用 `runtime: nvidia`。
- 生产镜像应从官方 vLLM Ascend A2 image 派生并记录 digest；不要从当前 CUDA Dockerfile 逐条删除 NVIDIA 包来构造，因为容易留下标准 Triton、CUDA PyTorch 或 FlashInfer 二进制冲突。
- `ffmpeg`、`sox`、`libsndfile`、Nginx 和应用依赖可以继续装在派生镜像中；它们在 CPU 上运行。

### 5.4 Kubernetes

| 组件 | 官方范围/作用 | 状态 |
|---|---|---|
| Kubernetes | MindCluster 7.3.0 支持 `1.17.x-1.34.x`，推荐 `>=1.19.x` | 官方支持；最终补丁与 CRI 组合需 PoC |
| Docker | MindCluster 7.3.0 支持 `18.09.x-28.5.1` | 官方支持 |
| containerd | MindCluster 7.3.0 支持 `1.4.x-2.1.4`，推荐 `1.6.x` | 官方支持；新集群优先此路径 |
| Ascend Docker Runtime | 容器运行时设备挂载 | 官方支持 |
| Ascend Device Plugin | 从 DCMI 发现、上报 `huawei.com/Ascend910`，向 kubelet 和运行时传递已分配设备 | 官方支持 |
| NPU Exporter | 设备指标与告警 | 建议纳入生产 BOM，版本与 MindCluster 行一致 |
| Volcano/Ascend Operator | 多卡/队列/故障恢复 | 单卡服务非首期必需；多卡或批量作业再引入 |

版本范围来自 [MindCluster 7.3.0 软件依赖表](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_004.html)。Kubernetes `>=1.24` 若使用 Docker 还需要 `cri-dockerd`；新集群优先使用 containerd，但仍要冻结 Kubernetes、containerd 与 MindCluster 的实际补丁组合。

Ascend Device Plugin 官方文档说明其从驱动读取型号、数量和健康状态，上报 kubelet，并把调度选择传给 Ascend Docker Runtime，见
[Ascend Device Plugin 组件介绍](https://www.hiascend.com/document/detail/zh/mindcluster/600/clustersched/introduction/schedulingsd/mxdlug_005.html)。
MindCluster 7.3 提供 A2/910 对应 DaemonSet，见
[Ascend Device Plugin 安装](https://www.hiascend.com/document/detail/en/mindcluster/730/clustersched/schedulingug/dlug_installation_019.html)。

应用编排建议为**一个 Pod/进程占用一张完整 NPU**。当前项目按 `CUDA_VISIBLE_DEVICES` 在单容器内拉多个实例的方式应删除或重写；Kubernetes 中由 Device Plugin 分配 NPU，水平扩容 Pod，再由 Service 负载均衡更清晰。多进程共享一张 910B、vNPU 切分和超卖都不应进入首轮 PoC。

## 6. Python、PyTorch、FunASR 与模型矩阵

### 6.1 PyTorch/torch_npu

Ascend Extension for PyTorch 的
[官方兼容表](https://github.com/Ascend/pytorch)
给出了 CANN、PyTorch、torch_npu 的精确对应关系，并明确 A2 训练/推理产品是支持硬件。对本项目最重要的规则不是“torch_npu 可以和任意 torch 配合”，而是必须使用 vLLM Ascend 发布矩阵中的成套版本。

PyTorch 2.5.1 及以后的 torch_npu 支持后端自动加载，但为降低隐式行为风险，PoC 仍应显式验证：

```python
import torch
import torch_npu

assert torch.npu.is_available()
assert torch.empty(1, device="npu:0").device.type == "npu"
```

这段仅是建议的环境验收代码，本文没有执行。

### 6.2 FunASR

当前项目锁定的 `funasr==1.3.1` 安装源码只检查 CUDA/CPU，不包含 NPU 可用性分支。FunASR 于 2026-02-28 合入
[Huawei NPU 支持提交](https://github.com/modelscope/FunASR/commit/34360664e3aa7638700bad5636af32e64c0fbfbd)，
并从正式 tag `v1.3.2` 开始在 `AutoModel` 中加入 `is_npu_available()`、`npu:0` 文档和 fallback 检查；例如
[FunASR v1.3.2 `auto_model.py`](https://github.com/modelscope/FunASR/blob/v1.3.2/funasr/auto/auto_model.py)
以及当前
[FunASR `main`](https://github.com/modelscope/FunASR/blob/main/funasr/auto/auto_model.py)。

这只证明框架入口能把模型移动到 NPU，不能证明每个 FunASR 模型的全部算子、缓存和流式状态都已在 910B 验证。因此建议：

1. PoC 单独创建 Ascend 依赖组，把 FunASR 升级到一个包含 NPU 分支的正式 tag；不要直接依赖 `main`。
2. 按 Paraformer online、FSMN VAD、离线 PUNC、实时 PUNC、CAM++ SV 的顺序逐个验证。
3. 每个模型都必须记录 CPU/CUDA 基线输出、NPU 输出、峰值 HBM、首次编译时间、稳态 RTF 和并发行为。

### 6.3 业务模型兼容矩阵

| 能力 | 官方/源码证据 | 判定 | 置信度 | 迁移方式 |
|---|---|---|---|---|
| Qwen3-ASR-1.7B BF16 离线识别 | vLLM Ascend 专门教程明确一张 64 GB 910B | **官方支持** | 高 | 使用 `vllm-ascend` A2 镜像与 1.7B BF16 |
| Qwen3-ASR-1.7B 在线服务 | 教程给出 `vllm serve` 与 Chat Completions audio 请求 | **官方支持** | 高 | 让 API 容器调用独立 vLLM 服务 |
| Qwen3-ASR 实时增量流 | 模型介绍称支持 streaming，但教程验证的是普通服务请求；项目有自定义累计缓冲语义 | **需要 PoC** | 低 | 用真实连续语音验证 partial/final、去重、上下文与延迟 |
| Qwen3-ASR-0.6B | Qwen 官方仓库发布 0.6B/1.7B；Ascend 教程只列 1.7B | **源码推断可行/需要 PoC** | 中低 | 同架构试跑，不作为首轮生产默认 |
| Qwen3 Forced Aligner | Qwen 官方 CUDA 示例和 upstream vLLM 架构支持；Ascend 教程未覆盖 | **需要 PoC** | 低 | 单独加载 pooling runner，验证 `encode`、时间戳 token 与 HBM |
| Paraformer online | FunASR 新版接受 `npu:0` | **源码推断可行/需要 PoC** | 中低 | 升级 FunASR，逐算子与流式 cache 验证 |
| FSMN VAD | 通用 torch_npu 路径，无模型级 910B 官方矩阵 | **需要 PoC** | 低 | 首期 CPU，二期迁 NPU |
| CT-Transformer PUNC | 通用 torch_npu 路径，无模型级 910B 官方矩阵 | **需要 PoC** | 低 | 首期 CPU，二期迁 NPU |
| 当前 CAM++ speaker verification | ModelScope pipeline 会先校验 device，随后项目才调用 `.to(device)` | **当前 NPU 实现不可行/需重写** | 高 | 首期 CPU；二期改用 FunASR/direct torch_npu 后再测 SV |
| CAM++ diarization ONNX 子模型 | ORT CANN EP 是 preview，算子表有限 | **需要 PoC/首期 CPU** | 低 | 保持 CPU EP，或逐个 ONNX 用 ATC 预检查 |
| HDBSCAN 聚类 | CPU Python/native library | **可保留 CPU** | 高 | 不迁 NPU，测 CPU 并发与内存 |
| ITN/weText | CPU 文本后处理 | **可保留 CPU** | 高 | 无需 NPU |

Qwen 官方仓库的
[本地部署说明](https://github.com/QwenLM/Qwen3-ASR)
主要展示 CUDA Transformers/vLLM；Ascend 的正式依据应取 vLLM Ascend 的专门教程，而不是把 Qwen 的 CUDA 命令直接替换成 `npu:0`。

### 6.4 ModelScope/CAM++ 的确定阻断

项目固定 `modelscope[framework]==1.34.0`，说话人分离通过 `modelscope.pipelines.pipeline(..., device=...)` 创建主 pipeline 及 SV/VAD/change-locator 子 pipeline。ModelScope v1.34.0 的
[`verify_device`](https://github.com/modelscope/modelscope/blob/v1.34.0/modelscope/utils/device.py)
仅接受 `cpu`、`cuda` 和 `gpu`，其
[`Pipeline` 基类](https://github.com/modelscope/modelscope/blob/v1.34.0/modelscope/pipelines/base.py)
也只构造 CUDA 或 CPU 设备；v1.37.0 源码仍保留这一限制。因此配置 `DEVICE=npu:0` 时，CAM++ 会在模型迁移前初始化失败，这不是性能未知，而是当前代码路径的确定兼容性错误。

首期必须把 CAM++ 固定到 CPU。若客户要求 CAM++ embedding 也在 NPU 上运行，需要绕开 ModelScope pipeline，改为 FunASR `AutoModel` 的 speaker model 路径或直接加载 CAM++ PyTorch 模型，再对预处理、SV、change-locator、聚类和精度逐层验证；HDBSCAN 聚类仍合理地保留在 CPU。

## 7. ONNX Runtime、ATC 与 AOE

### 7.1 ONNX Runtime CANN EP

ONNX Runtime 将 Huawei CANN EP 标为 **preview**，见
[Execution Provider 总表](https://onnxruntime.ai/docs/execution-providers/)。
官方
[CANN EP 算子表](https://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html)
只列出有限单算子集合，且 Conv/Pool 有二维与常量权重限制。官方
[构建说明](https://onnxruntime.ai/docs/build/eps.html#cann)
要求 Linux、CANN Toolkit，并从源码用 `--use_cann` 构建，支持 x64 与 AArch64。

因此：

- 不能假设 PyPI 的通用 `onnxruntime` wheel 自动获得 CANN EP。
- 不能假设 CAM++ 目录中的所有 ONNX 模型会完整落到 NPU；即使 Session 可创建，也可能把未支持节点回退到 CPU。
- 若业务要求严格 NPU-only，应关闭 CPU fallback 并检查实际 provider/node placement；否则“请求成功”不能证明 NPU 加速。
- 首期保持 `CPUExecutionProvider` 是更低风险方案，CPU 资源需进入 BOM。

### 7.2 ATC/OM

CANN 9.0
[ATC 参数表](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/devaids/atctool/atlasatc_16_0039.html)
支持 ONNX、动态 batch、动态维度和目标 `soc_version`，但前提是图中算子符合 CANN 规格。`soc_version` 必须与实际运行芯片一致，CANN 运行版本不得低于 OM 转换版本，见
[ATC `soc_version`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/devaids/atctool/atlasatcparam_16_0036.html)。

ATC 适合二期优化以下固定边界模型：FSMN VAD、PUNC、CAM++ 子模型。它不适合作为首期 Qwen3-ASR 迁移方式，因为 Qwen3-ASR 已有官方 vLLM Ascend 路线，而自行导出整个音频编码器+LLM+生成循环会显著扩大动态 shape、KV cache、tokenizer 和服务调度工作量。

动态 batch 的 OM 会按预设最大 batch 分配内存，见
[CANN Dynamic Batch Size](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/graphdevg/atlasag_25_0049.html)。
因此 profile 应从生产音频长度与 batch 分布反推，不能无成本地把上限设大。

### 7.3 AOE

AOE 不是首轮阻塞项。CANN 8.5 的官方 AOE 文档明确写明 Atlas A2 训练/推理产品不支持 offline inference tuning，见
[AOE Tuning Workflow](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/aoe/aoeep_16_020.html)。
在所选 CANN 9.0.0 环境中应重新查询对应 A2 功能矩阵；在官方支持得到确认前，不把 AOE 收益写入容量或工期承诺。优先使用 vLLM Ascend 自带优化和 CANN profiling，ATC 子模型在无 AOE 的情况下也必须先达到验收线。

## 8. Triton、xformers、flash-attn 与 FlashInfer

| 当前/潜在组件 | 官方硬件范围 | 910B 结论 |
|---|---|---|
| 标准 Triton | NVIDIA CUDA、AMD ROCm，CPU 开发中，见 [Triton README](https://github.com/triton-lang/triton/blob/main/README.md) | **不可直接使用** |
| Triton-Ascend | 官方项目给出 910B image、CANN/torch_npu 矩阵，见 [安装指南](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/installation_guide.md) | **替代项**，使用 vLLM Ascend 发布行指定版本 |
| xformers | 官方 wheel 为 CUDA 12.6/12.8/13.0 与实验 ROCm，见 [xformers README](https://github.com/facebookresearch/xformers) | **移除**，不要进入 Ascend 锁文件 |
| flash-attn | 官方要求 CUDA 或 ROCm，见 [FlashAttention README](https://github.com/Dao-AILab/flash-attention) | **移除/替换**，使用 vLLM Ascend/CANN attention 算子 |
| FlashInfer CUDA cubin/JIT | 当前镜像保留 nvcc 专用于它 | **移除**；不能在 Ascend 构建 |

不要手动把 GPU Triton kernel 翻译为 NPU 作为首轮目标。Triton-Ascend 官方编程指南也提醒 GPU kernel 直接映射到 Ascend 会受到 AI Core 启动与任务粒度差异影响，见
[Triton-Ascend Programming Guide](https://github.com/triton-lang/triton-ascend/blob/main/docs/en/programming_guide.md)。

## 9. 动态 batch、并发、精度与性能

### 9.1 调度边界

本项目目前存在三种“batch”，需要分开处理：

1. 应用把单个长音频分段后按 `max_inference_batch_size` 调用 `LLM.generate`。
2. vLLM 在多个请求之间做运行时调度。
3. FunASR 的 `batch_size_s`/模型 batch 按音频时长聚合。

当前离线 Qwen 的全局 lock 会把第 2 类并发串行化。推荐路线若改为本地 `vllm serve`，应让 vLLM 接受并发请求，并只在应用层保留有业务语义的限流；不能同时用“单请求分段 batch”和“HTTP 请求数”推导有效 batch。

Paraformer streaming 有会话 cache，不能把不同 WebSocket 会话的状态合并。可以按会话并发、按固定 chunk 长度做批量，但必须验证 cache 隔离和 final flush。

### 9.2 精度

- Qwen3-ASR 1.7B 的官方 910B 路线是 BF16；教程要求用 WER/CER 评估，见
  [Qwen3-ASR Accuracy Evaluation](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-ASR-1.7B.html#accuracy-evaluation)。
- 不在第一阶段引入 INT8/W8A8。量化不是“无损开关”，并且 ASR 音频编码器、forced aligner 和小模型的量化支持不能从通用 Qwen3 文本模型矩阵外推。
- FunASR 默认 dtype、NPU 算子融合和 ATC mixed precision 都必须以 CUDA/CPU 基线做逐语料比较。
- ATC 官方说明默认精度策略偏向性能，可能发生精度溢出；如果出现精度问题，应使用 origin/keep-dtype 路径定位，见
  [ATC Getting Started](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/atctool/atlasatc_16_0003.html)。

### 9.3 性能

官方没有为本项目的音频长度、并发和前后处理给出可直接套用的吞吐数字。vLLM Ascend Qwen3-ASR 教程要求至少记录音频时长、请求并发、端到端延迟、RTF 和吞吐，并分别测试短音频、长音频和并发请求，见
[Performance Evaluation](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-ASR-1.7B.html#performance-evaluation)。

性能测试必须包含：

- FFmpeg/librosa 解码与 16 kHz 重采样；
- 请求解析、文件 IO 和模型缓存命中；
- Qwen audio encoder prefill 与 token decode；
- 可选 forced aligner；
- 可选 VAD、PUNC、CAM++ 和 HDBSCAN；
- API 序列化和 WebSocket partial/final 发送。

只测一个已解码 NumPy 数组的模型调用不能作为服务容量结论。

## 10. 技术路线

### 路线 A：Qwen NPU + 辅助链 CPU，随后渐进迁移（推荐）

**思路**

- 从官方 `vllm-ascend:v0.22.1rc1` A2 镜像派生应用镜像。
- Qwen3-ASR-1.7B 通过 vLLM Ascend 使用一张 64 GB 910B。
- FFmpeg/librosa、FSMN VAD、PUNC、CAM++、ONNX、HDBSCAN 首期留在 CPU。
- 应用调用同 Pod/节点的 vLLM HTTP 服务，或在确认 API/thread safety 后使用 NPU plugin 的 Python API；优先 HTTP 服务隔离升级边界。
- 第二阶段按 Paraformer、VAD/PUNC、CAM++ 的顺序迁移到 torch_npu。

**影响范围**

- 新增独立 Ascend 依赖锁和 Dockerfile/Compose/K8s 清单。
- 抽象设备后端与内存指标，增加 NPU 路由。
- 调整 Qwen adapter，使其使用 vLLM Ascend 支持的服务/API。
- 重新设计 offline lock、限流和模型生命周期。

**优点**

- 核心模型沿官方验证路径，最小化未知算子风险。
- 保持完整业务 API，辅助链有可靠 CPU 退路。
- 能逐项衡量 NPU 迁移是否真正带来收益。

**缺点/风险**

- CPU 仍需较高核数，长音频/说话人分离可能成为瓶颈。
- Qwen 与 forced aligner 若同卡常驻，HBM 和调度需要单独验证。
- vLLM Ascend 参考版本是 RC，需固定镜像 digest 和回归集。

**建议验证**

按第 13 节 PoC 清单验收；第一道关只启用 Qwen 1.7B，第二道关恢复完整 API，第三道关再迁 FunASR。

### 路线 B：单一 torch_npu 进程承载 Qwen Transformers + FunASR

**思路**

- 使用 CANN/torch_npu 统一 PyTorch 环境。
- Qwen3-ASR 改用 Transformers/qwen-asr eager NPU 推理，FunASR/辅助模型直接 `.to("npu:0")`。
- 应用自己实现 batch、队列、HBM 管理和并发限制。

**影响范围**

- 重写当前 vLLM adapter 和 Qwen 流式/aligner 路径。
- 需要全面审计 Qwen audio encoder、attention、generation、aligner 的 NPU 算子与 fallback。
- 需要自建连续批处理或接受较低吞吐。

**优点**

- 单一 Python 设备抽象，FunASR 与 Qwen 更易共享 tensor/device。
- 避开 vLLM server 进程与应用协议适配。

**缺点/风险**

- Qwen 官方仓库本地示例以 CUDA 为主；没有找到 Qwen3-ASR Transformers-on-910B 的官方端到端教程。
- 放弃 vLLM Ascend 已验证的模型实现、attention kernel 和请求调度。
- 工期与性能不确定性最高。

**判定**

仅适合做对照 PoC，不推荐作为第一生产路线。

### 路线 C：经典模型 ONNX -> ATC/OM，Qwen 保持 vLLM Ascend

**思路**

- Qwen3-ASR 沿路线 A。
- 把 Paraformer/VAD/PUNC/CAM++ 中边界清晰的模型导出 ONNX，逐图用 ATC 转 OM。
- 使用 AscendCL/服务层执行 OM，按生产 shape 分布建立动态 profile。

**优点**

- 经典模型可获得稳定的编译图和显式 shape/precision 控制。
- 运行时不依赖 ONNX Runtime CANN EP 的 preview 状态。

**缺点/风险**

- 每个模型需要导出、算子预检、精度校准、动态 shape 和后处理接线。
- 流式 Paraformer cache、变长语音和 CAM++ 组合 pipeline 会增加模型边界数量。
- AOE 在 A2 offline 场景的支持有限，不能预设自动调优收益。

**判定**

作为路线 A 的二期性能工程，不作为首期完整部署主线。

## 11. 推荐架构

```text
Client
  -> FastAPI / WebSocket API
       -> CPU audio decode, resample, VAD, punctuation, diarization
       -> Qwen adapter
            -> local vLLM Ascend service
                 -> Qwen3-ASR-1.7B BF16 on one Ascend 910B
            -> optional forced aligner after separate PoC
       -> response normalization and timestamps
```

生产 Kubernetes 形态：一个 API Pod 绑定一张完整 910B，Pod 内包含应用进程和本地 vLLM Ascend 服务进程，或把两者放在同一容器由轻量 supervisor 管理。若拆成两个 Pod，需要解决 NPU 不能被两个 Pod 同时独占的问题，因此不建议在首期做跨 Pod 拆分。

为了减少进程耦合，应用与 vLLM 间优先使用 localhost HTTP。其代价是要把当前自定义 forced aligner 和 streaming state 语义重新映射到 vLLM Ascend 实际支持的 API；若这些能力无法映射，再对特定路径使用 Python API，而不是整个服务都回退到内嵌模式。

## 12. 阻塞项

以下项目在进入开发前或 PoC 第一周必须闭合：

1. **硬件 BOM 不完整**：确认 Atlas 产品型号、910B revision、每卡 HBM、卡数、CPU 架构、宿主 OS/内核、NUMA 拓扑。
2. **HDK 映射缺失**：从 CANN 9.0.0 官方下载/版本映射获取精确 driver 与 firmware 版本，不能凭经验填写。
3. **vLLM Ascend 为 RC**：确认可接受 `0.22.1rc1`，或选择已有 Qwen3-ASR 教程覆盖的后续 final/post 版本。
4. **当前应用拒绝 NPU**：必须修改设备发现、Qwen runtime family、HBM 指标、健康检查与模型选择。
5. **Forced Aligner 无 Ascend 结论**：决定首期可否关闭 `word_timestamps`；若不能，aligner 是发布阻塞项。
6. **Qwen 实时语义未闭合**：官方服务支持音频请求不等于项目自定义增量 partial/final 行为一致。
7. **FunASR 版本升级**：从 1.3.1 升级到含 NPU 分支的正式 tag，并验证 API/输出兼容。
8. **CAM++ 当前实现明确拒绝 NPU**：首期固定 CPU；若业务要求全模型 NPU，必须重写 ModelScope pipeline 边界，并确认哪些子模型走 PyTorch、哪些走 ONNX 以及实际 provider placement。
9. **依赖锁拆分**：Ascend 环境不能解析到 CUDA torch、标准 Triton、xformers、flash-attn、FlashInfer cubin。
10. **容量 SLO 未给出**：需要目标音频分布、并发、p95/p99 延迟、实时率和错误率，才能判断单卡容量。

## 13. PoC 验收清单

以下为工程建议门槛，不是已测结果。业务 SLO 更严格时以业务值为准。

### 13.1 环境与可重复性

- [ ] 记录服务器 PN、CPU 架构、NUMA、910B 型号/revision、HBM、OS、内核、driver、firmware、CANN、镜像 digest。
- [ ] 宿主 `npu-smi info` 正常，容器内只看见分配的设备。
- [ ] `pip show` 的 vLLM/vLLM Ascend/PyTorch/torch_npu 与发布矩阵完全一致。
- [ ] `pip check` 无冲突；环境中没有 CUDA torch、xformers、flash-attn、FlashInfer CUDA 或错误的标准 Triton 覆盖。
- [ ] 离线启动只使用预置模型缓存，不产生外网下载。
- [ ] 镜像在同型号第二台节点可重复启动。

### 13.2 功能

- [ ] 官方 Qwen3-ASR 1.7B 示例音频返回 HTTP 200 与非空转写。
- [ ] 项目 OpenAI transcription API、阿里云兼容 API、健康检查和模型列表通过。
- [ ] 中文、英文、粤语、混语、静音、噪声、音乐/歌声、短音频、60 秒分段和长音频用例通过。
- [ ] Paraformer WebSocket 的 partial/final、断开重连、空 chunk、final flush 与 CUDA/CPU 基线一致。
- [ ] 若首期承诺词级时间戳，forced aligner 在 NPU 上加载、输出数量正确、无反向时间戳；否则接口明确禁用该能力。
- [ ] 若启用说话人分离，CAM++ 输出段数、speaker id、时序合并与 CPU 基线一致。
- [ ] 模型同时加载时无 OOM；分别记录 Qwen、aligner、FunASR/CAM++ 的增量 HBM。

### 13.3 精度

- [ ] 固定一套带转写、语言、时间戳和说话人的金标语料；版本化音频 hash 与评测脚本。
- [ ] Qwen/Paraformer 的中文 CER、英文 WER 相对当前 CUDA 基线绝对劣化不超过 0.5 个百分点。
- [ ] 语言识别准确率相对基线绝对劣化不超过 0.5 个百分点。
- [ ] Forced Aligner 有效 token 覆盖率 100%，无 NaN/Inf/反向区间；时间戳 MAE 相对 CUDA 基线增加不超过 50 ms。
- [ ] Diarization DER 相对 CPU 基线绝对劣化不超过 1 个百分点。
- [ ] 对所有精度异常保留 CPU/CUDA/NPU 中间输出，禁止用开启 CPU fallback 的“成功请求”掩盖 NPU 算子问题。

### 13.4 性能与并发

- [ ] 对 5 s、30 s、5 min、30 min 音频分别测并发 1/2/4/8。
- [ ] 记录端到端 p50/p95/p99、RTF、audio-hours/hour、首 token/首 partial 延迟、CPU、RSS、HBM、NPU 利用率。
- [ ] 首期最低门槛：单请求端到端 RTF < 1.0；生产门槛由业务 SLO 替换。
- [ ] 并发压测错误率 < 0.1%，无 OOM、worker crash、hang 或跨会话 cache 污染。
- [ ] 2 小时稳定性压测后 RSS/HBM 稳态增长不超过 5%，模型不重复加载。
- [ ] 分别给出“仅 Qwen”“Qwen+aligner”“完整 VAD/PUNC/CAM++”三组容量数据。
- [ ] FFmpeg/librosa 前处理 CPU 时间单独计量；若占端到端 p95 超过 20%，增加 CPU 并行或预解码队列后复测。

### 13.5 容器与 Kubernetes

- [ ] Docker Runtime 只挂载分配的 NPU 与必要 driver 文件，不使用 `--privileged` 作为最终生产方案。
- [ ] Device Plugin 上报的 `huawei.com/Ascend910` 资源数量与物理设备一致。
- [ ] Pod 删除/重建后设备释放，模型可重新加载，Service 无僵尸 endpoint。
- [ ] readiness 在模型未就绪时失败，liveness 不因长推理误杀进程。
- [ ] 节点 drain、Pod 滚动升级和单卡故障告警有明确行为。

## 14. BOM

### 14.1 硬件与基础设施

| 项目 | 最低/推荐 | 依据或备注 |
|---|---|---|
| NPU | 1 x Ascend 910B 64 GB | Qwen3-ASR 1.7B BF16 官方教程明确可部署 |
| 服务器 | 明确 Atlas 800I A2/800T A2 或实际卡型 | 产品型号决定 OS、内核和 HDK |
| CPU | 工程建议至少 16 物理核；完整 CPU 辅助链建议 32 核起测 | FFmpeg/librosa/VAD/PUNC/CAM++/HDBSCAN 均可能使用 CPU；需压测定容 |
| RAM | 工程建议 128 GB 起测，长音频/多 worker 建议 256 GB | 非官方最低值；2 GB 请求上限和模型缓存会放大 RSS |
| 本地盘 | 工程建议预留 200 GB+ SSD | 镜像、模型缓存、日志、临时音频和评测集；按实际模型快照复核 |
| 网络 | 单节点 PoC 无特殊要求；模型离线预置 | 多节点/共享缓存不在首期范围 |

### 14.2 软件

| 项目 | PoC 版本/选择 |
|---|---|
| Host OS/kernel | 实际 Atlas 产品官方清单内版本；优先厂商交付镜像 |
| Driver/firmware | CANN 9.0.0 对应 HDK 映射，版本待硬件 BOM 确认 |
| Container engine | Docker/containerd + 对应 Ascend Docker Runtime |
| Base image | `quay.io/ascend/vllm-ascend:v0.22.1rc1`，冻结 digest |
| CANN | 9.0.0 |
| Python | 3.10 或 3.11；建议 3.11，避免同时试多个版本 |
| PyTorch/torch_npu | 2.10.0 / 2.10.0 |
| vLLM/vLLM Ascend | 0.22.1 / 0.22.1rc1 |
| Triton | Triton-Ascend 3.2.1，由兼容行提供 |
| FunASR | 一个含 NPU 分支的正式 tag；PoC 后再冻结，不能继续 1.3.1 |
| ONNX Runtime | 首期 CPU EP；CANN EP 单独构建与验证，不进入基础镜像默认路径 |
| System packages | FFmpeg、SoX、libsndfile、Nginx、必要编译/诊断工具 |
| Kubernetes | Ascend Docker Runtime + 匹配版本 Device Plugin；Exporter 建议同一 MindCluster 发布行 |

## 15. 风险清单

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| 具体 910B 产品/OS/HDK 组合不明 | 高 | 高 | 采购/运维先提供完整 BOM 与官方映射 |
| vLLM Ascend 参考版本为 RC | 高 | 高 | 固定 digest；建立回归集；评估后续 final/post 整体升级 |
| 当前 CUDA 专用依赖污染 Ascend 环境 | 高 | 高 | 独立锁文件/依赖组；从官方 Ascend image 派生；`pip check` |
| Qwen adapter 与 vLLM Ascend API 差异 | 高 | 高 | 先做官方 server smoke test，再实现最薄 adapter |
| Forced Aligner 不支持或 HBM 冲突 | 中高 | 高 | 单独 PoC；首期允许禁用词级时间戳；必要时 CPU/独立卡 |
| 自定义实时流语义与官方 serving 不一致 | 高 | 高 | 真实连续音频回归 partial/final/cache，而非只测离线音频 |
| FunASR 模型存在 NPU 不支持算子 | 中 | 中高 | 逐模型 gate；首期 CPU；必要时 ATC/自定义算子 |
| CAM++ ModelScope pipeline 拒绝 `npu:0` | 高（当前必现） | 中高 | 首期固定 CPU；全模型 NPU 需求需重写加载与执行路径 |
| 单卡多模型 HBM 争用 | 中高 | 高 | 分阶段加载；下调 vLLM memory utilization；记录峰值；必要时双卡 |
| 应用 lock 抵消 vLLM 连续调度 | 高 | 中高 | 移除全局串行边界，保留有界队列和业务限流 |
| FFmpeg/librosa 成为 CPU 瓶颈 | 中 | 中 | 独立计时；CPU 池/预解码；限制超长请求并分段 |
| NPU 图编译/首次请求抖动 | 中 | 中 | readiness 前 warmup；分离 cold/warm 指标；持久化编译缓存（若官方支持） |
| ATC 动态 profile 选错导致浪费/OOM | 中 | 中 | 从真实 shape 分布设计 profile；最大 profile 单独压测 |
| K8s 运行时/Device Plugin 版本错配 | 中 | 高 | 使用同一 MindCluster 发布行；先裸 Docker 后 K8s |
| 精度回归被 CPU fallback 隐藏 | 中 | 高 | 禁 fallback 的 NPU-only 评测；记录节点/provider placement |

## 16. 工期估算

以下是**工程估算**，不是厂商承诺；假设一名熟悉 Python/FastAPI/vLLM 的高级工程师，另有 0.25-0.5 名运维/Ascend 工程师支持，目标硬件、模型缓存和 driver 安装权限可用。

| 阶段 | 路线 A | 路线 B | 路线 C 增量 |
|---|---:|---:|---:|
| 环境/BOM/官方 Qwen smoke test | 3-5 人日 | 3-5 人日 | 不适用 |
| 应用设备与依赖改造 | 5-8 人日 | 8-12 人日 | 2-4 人日 |
| Qwen API、离线/实时、aligner PoC | 5-10 人日 | 10-18 人日 | 不适用 |
| FunASR/辅助链恢复 | 4-8 人日（CPU 首期） | 8-15 人日（NPU） | 每模型 3-7 人日 |
| 精度、并发、稳定性与 K8s | 8-12 人日 | 10-15 人日 | 5-10 人日 |
| 合计 | **25-43 人日，约 5-9 周** | **39-65 人日，约 8-13 周** | **在路线 A 上增加 15-35 人日** |

若首期允许关闭 forced aligner、CAM++ 和 Qwen 实时增量，只交付 Qwen3-ASR-1.7B 离线 API，路线 A 可压缩到约 **8-15 人日**。若必须一次交付所有现有能力且全部在 NPU 上运行，应采用路线 A + C 的上界，不应承诺两周内完成。

## 17. 建议决策

1. 采购/运维先闭合具体 Atlas 产品、OS/内核、driver/firmware/CANN 映射。
2. 用官方 `vllm-ascend:v0.22.1rc1` 镜像完成 Qwen3-ASR-1.7B 单卡 smoke test，作为 Go/No-Go 的第一道门。
3. 选择路线 A，首期完整服务保留 CPU 辅助链；不要同时引入 ATC/OM 和 AOE。
4. 把 forced aligner 与实时增量单独设为发布 gate，不从普通 `/v1/chat/completions` 成功推断二者可用。
5. 完成精度、RTF、并发和 2 小时稳定性测试后再决定是否把 Paraformer/VAD/PUNC/CAM++ 迁到 NPU。
6. 只有 CPU 辅助链被实测为瓶颈时，才启动路线 C 的逐模型 ATC 工程。

最终判断是：**910B 能承载本项目的核心 Qwen3-ASR-1.7B 推理；整套服务部署在 910B 服务器上有条件可行，但“所有模型都在 NPU 上运行”当前不可直接承诺。完整交付需要明确的软件栈分叉和业务适配，最大技术风险在 forced aligner、实时语义与辅助模型，而不再是 Qwen3-ASR 主模型本身。**
