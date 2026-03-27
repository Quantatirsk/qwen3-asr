#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预下载脚本
用于构建 Docker 镜像时预下载所有模型

- Paraformer 模型从 ModelScope 下载
- Qwen3-ASR 模型从 HuggingFace 下载 (vLLM 要求)
"""

import os
import json
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

# === Qwen3-ASR 模型选择 ===
_ENABLED_MODELS = os.getenv("ENABLED_MODELS", "auto")


def _get_qwen_models() -> list[tuple[str, str]]:
    """返回要从 HuggingFace 下载的 Qwen3-ASR 模型列表"""
    config = _ENABLED_MODELS.strip()
    config_lower = config.lower()

    # 从逗号分隔的列表中提取 Qwen 模型
    if config_lower not in ("auto", "all"):
        models = []
        for model in config.split(","):
            model = model.strip()
            if model == "qwen3-asr-1.7b":
                models.append(("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B"))
            elif model == "qwen3-asr-0.6b":
                models.append(("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B"))
        if models:
            # 添加强制对齐器（所有 Qwen 模型都需要）
            models.append(("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner"))
            print(f"ENABLED_MODELS={config}，加载指定 Qwen 模型")
            return models

    # all 模式：下载所有 Qwen 模型
    if config_lower == "all":
        print("ENABLED_MODELS=all，下载所有 Qwen 模型")
        return [
            ("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B"),
            ("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B"),
            ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner"),
        ]

    # auto 模式（或空列表时）根据显存/内存选择
    try:
        from app.core.device import has_gpu, get_vram_gb

        if not has_gpu():
            print("无 GPU，跳过 Qwen3-ASR")
            return []

        vram = get_vram_gb()
        if vram >= 32:
            print(f"ENABLED_MODELS=auto，显存/内存 {vram:.1f}GB >= 32GB，加载 Qwen3-ASR-1.7B")
            return [("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B"), ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner")]
        else:
            print(f"ENABLED_MODELS=auto，显存/内存 {vram:.1f}GB < 32GB，加载 Qwen3-ASR-0.6B")
            return [("Qwen/Qwen3-ASR-0.6B", "Qwen3-ASR 0.6B"), ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner")]
    except ImportError:
        print("ENABLED_MODELS=auto，默认加载 Qwen3-ASR-1.7B")
        return [("Qwen/Qwen3-ASR-1.7B", "Qwen3-ASR 1.7B"), ("Qwen/Qwen3-ForcedAligner-0.6B", "Qwen3-ForcedAligner")]


# === ModelScope 模型 (Paraformer) ===
# 注意：仅列出代码中实际使用的模型，避免重复加载
# 同一模型的不同命名（如 iic/ vs damo/）只保留实际使用的版本
# 格式: (model_id, description, revision)  revision 为 None 表示使用最新版本
MODELSCOPE_MODELS = [
    ("iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "Paraformer Large", None),
    ("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", "Paraformer Online", None),
    # VAD 模型: config.py 中使用的是 damo/ 版本
    # CAM++ 依赖 v2.0.2 版本，不是最新版
    ("damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", "VAD", "v2.0.2"),
    # CAM++ 说话人分离: speaker_diarizer.py 中使用的是 iic/speech_campplus_speaker-diarization_common
    # 注意：以下两个 damo/ 模型是 CAM++ 的隐式依赖，FunASR 会自动下载，需预置避免运行时下载
    ("iic/speech_campplus_speaker-diarization_common", "CAM++", None),
    ("damo/speech_campplus_sv_zh-cn_16k-common", "CAM++ SV (隐式依赖)", None),
    ("damo/speech_campplus-transformer_scl_zh-cn_16k-common", "CAM++ Transformer (隐式依赖)", None),
    ("iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", "标点模型", None),
    ("iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727", "标点模型(实时)", None),
    ("iic/speech_ngram_lm_zh-cn-ai-wesp-fst", "N-gram LM", None),
]

# === HuggingFace 模型 (Qwen3-ASR) ===
HF_MODELS = _get_qwen_models()

# === 合并所有模型（用于检查）===
ALL_MODELS = MODELSCOPE_MODELS + HF_MODELS


def _get_cache_path(model_id: str, source: str = "modelscope") -> Path:
    """获取模型缓存路径"""
    if source == "huggingface":
        # HF 缓存格式: ~/.cache/huggingface/hub/models--{org}--{model}/
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        org, model = model_id.split("/", 1)
        model_path = cache_dir / f"models--{org}--{model}"
    else:
        cache_dir = Path.home() / ".cache" / "modelscope"
        model_path = cache_dir / "hub" / "models" / model_id
    return model_path


def check_model_exists(model_id: str, source: str = "modelscope") -> tuple[bool, str]:
    """检查模型是否已存在于本地缓存"""
    try:
        model_path = _get_cache_path(model_id, source)

        if model_path.exists() and model_path.is_dir():
            if any(model_path.iterdir()):
                return True, str(model_path)
    except Exception:
        pass

    return False, ""


def check_all_models() -> list[tuple[str, str, str, Optional[str]]]:
    """检查所有模型是否存在

    Returns:
        缺失的模型列表，每个元素为 (model_id, description, source, revision)
    """
    missing = []

    # 检查 ModelScope 模型
    for item in MODELSCOPE_MODELS:
        model_id, desc, revision = item
        exists, _ = check_model_exists(model_id, source="modelscope")
        if not exists:
            missing.append((model_id, desc, "modelscope", revision))

    # 检查 HuggingFace 模型 (HF 模型暂不支持指定版本)
    for model_id, desc in HF_MODELS:
        exists, _ = check_model_exists(model_id, source="huggingface")
        if not exists:
            missing.append((model_id, desc, "huggingface", None))

    return missing


def fix_camplusplus_config() -> bool:
    """修复 CAM++ 配置文件，将模型ID替换为本地路径（用于离线环境）

    修复 issue #15: 离线环境下 CAM++ 模型会尝试从 modelscope.cn 获取依赖模型配置

    Returns:
        是否修复成功
    """
    try:
        cache_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models"
        config_file = cache_dir / "iic/speech_campplus_speaker-diarization_common/configuration.json"

        if not config_file.exists():
            return False

        # 读取配置文件
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 需要替换的模型ID -> 本地路径映射
        replacements = {
            "damo/speech_campplus_sv_zh-cn_16k-common": str(cache_dir / "damo/speech_campplus_sv_zh-cn_16k-common"),
            "damo/speech_campplus-transformer_scl_zh-cn_16k-common": str(cache_dir / "damo/speech_campplus-transformer_scl_zh-cn_16k-common"),
            "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch": str(cache_dir / "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        }

        # 检查是否需要修改
        modified = False
        if "model" in config:
            for key in ["speaker_model", "change_locator", "vad_model"]:
                if key in config["model"]:
                    old_value = config["model"][key]
                    if old_value in replacements:
                        new_value = replacements[old_value]
                        # 检查本地路径是否存在
                        if Path(new_value).exists():
                            config["model"][key] = new_value
                            modified = True

        # 写回配置文件
        if modified:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            return True

        return False

    except Exception as e:
        print(f"⚠️  修复 CAM++ 配置文件失败: {e}")
        return False


def download_models(auto_mode: bool = False, export_dir: Optional[str] = None) -> bool:
    """下载所有需要的模型

    Args:
        auto_mode: 如果为True，表示自动模式（从start.py调用），会简化输出
        export_dir: 如果指定，将下载的模型导出到该目录（用于离线部署）

    Returns:
        是否全部下载成功
    """
    import shutil

    # 检查缺失的模型
    missing = check_all_models()

    # 导出模式下，需要包含已存在的模型
    export_path = Path(export_dir) if export_dir else None
    models_to_export = set()

    if not missing:
        if export_path:
            # 导出模式下，收集所有模型路径
            for item in MODELSCOPE_MODELS:
                model_id, desc, _ = item
                models_to_export.add((model_id, "modelscope"))
            for model_id, desc in HF_MODELS:
                models_to_export.add((model_id, "huggingface"))
        if not auto_mode:
            print("✅ 所有模型已存在，无需下载")
        if not export_path:
            return True

    ms_cache_dir = Path.home() / ".cache" / "modelscope"
    hf_cache_dir = Path.home() / ".cache" / "huggingface"

    if auto_mode:
        print(f"📦 检测到 {len(missing)} 个模型需要下载...")
    else:
        print("=" * 60)
        print("FunASR-API 模型预下载")
        print("=" * 60)
        print(f"ModelScope 缓存: {ms_cache_dir}")
        print(f"HuggingFace 缓存: {hf_cache_dir}")
        print(f"待下载模型: {len(missing)} 个")
        print("=" * 60)

    failed = []
    downloaded = []

    # 下载 ModelScope 模型 (Paraformer)
    ms_missing = [(mid, desc, rev) for mid, desc, src, rev in missing if src == "modelscope"]
    if ms_missing:
        if not auto_mode:
            print("\n📦 开始下载 ModelScope 模型 (Paraformer)...")
            print("-" * 60)

        for i, (model_id, desc, revision) in enumerate(ms_missing, 1):
            if not auto_mode:
                print(f"\n[{i}/{len(ms_missing)}] {desc}")
                print(f"    模型ID: {model_id}")
                if revision:
                    print(f"    版本: {revision}")
                print(f"    📥 开始下载...", end="")

            try:
                # 传递版本参数，如果指定了版本
                if revision:
                    path = ms_snapshot_download(model_id, revision=revision)
                else:
                    path = ms_snapshot_download(model_id)
                if not auto_mode:
                    print(f" ✅ 完成: {path}")
                downloaded.append((model_id, "modelscope", path))
            except Exception as e:
                if not auto_mode:
                    print(f" ❌ 失败: {e}")
                failed.append((model_id, str(e)))

    # 下载 HuggingFace 模型 (Qwen3-ASR)
    hf_missing = [(mid, desc, rev) for mid, desc, src, rev in missing if src == "huggingface"]
    if hf_missing:
        if not auto_mode:
            print("\n📦 开始下载 HuggingFace 模型 (Qwen3-ASR)...")
            print("-" * 60)

        for i, (model_id, desc, _) in enumerate(hf_missing, 1):
            if not auto_mode:
                print(f"\n[{i}/{len(hf_missing)}] {desc}")
                print(f"    模型ID: {model_id}")
                print(f"    📥 开始下载...", end="")

            try:
                path = hf_snapshot_download(model_id)
                if not auto_mode:
                    print(f" ✅ 完成: {path}")
                downloaded.append((model_id, "huggingface", path))
            except Exception as e:
                if not auto_mode:
                    print(f" ❌ 失败: {e}")
                failed.append((model_id, str(e)))

    # 修复 CAM++ 配置文件（用于离线环境）
    if not auto_mode:
        print("\n🔧 修复 CAM++ 配置文件...")
    if fix_camplusplus_config():
        if not auto_mode:
            print("  ✅ CAM++ 配置已修复（离线环境可用）")
    else:
        if not auto_mode:
            print("  ℹ️  无需修复或配置文件不存在")

    # 导出模式：复制模型到项目 models/ 目录（与 docker-compose 挂载路径一致）
    if export_path and not failed:
        if not auto_mode:
            print(f"\n📦 导出模型到: {export_path}")

        # models/modelscope/ 和 models/huggingface/ 结构
        ms_target = export_path / "modelscope"
        hf_target = export_path / "huggingface"

        # 收集所有需要导出的模型
        all_models = []
        for item in MODELSCOPE_MODELS:
            model_id, desc, _ = item
            all_models.append((model_id, "modelscope"))
        for model_id, desc in HF_MODELS:
            all_models.append((model_id, "huggingface"))

        exported = 0
        for model_id, source in all_models:
            cache_path = _get_cache_path(model_id, source)
            if cache_path.exists():
                # 计算相对路径，保持原结构
                if source == "modelscope":
                    rel_path = cache_path.relative_to(Path.home() / ".cache" / "modelscope")
                    target_dir = ms_target / rel_path
                else:
                    rel_path = cache_path.relative_to(Path.home() / ".cache" / "huggingface" / "hub")
                    target_dir = hf_target / "hub" / rel_path

                target_dir.parent.mkdir(parents=True, exist_ok=True)
                if not auto_mode:
                    print(f"  📂 {model_id}", end="")
                try:
                    shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)
                    exported += 1
                    if not auto_mode:
                        print(" ✅")
                except Exception as e:
                    if not auto_mode:
                        print(f" ❌ {e}")

        if not auto_mode:
            print(f"\n✅ 已导出 {exported} 个模型到 models/")

    if not auto_mode:
        print("\n" + "=" * 60)
        print("📊 下载统计:")
        print(f"  ✅ 已下载: {len(downloaded)} 个")
        print(f"  ❌ 失败: {len(failed)} 个")
        print("=" * 60)

        if failed:
            print(f"\n失败的模型:")
            for model_id, err in failed:
                print(f"  - {model_id}: {err}")
            return False
        else:
            print("\n✅ 所有模型准备就绪!")
            print("=" * 60)

    return len(failed) == 0


if __name__ == "__main__":
    download_models()
