#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预下载脚本
用于构建 Docker 镜像时预下载所有模型

- Paraformer 模型从 ModelScope 下载
- Qwen3-ASR 模型从 HuggingFace 下载 (CUDA vLLM / CPU Rust)
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
from app.services.asr.model_capabilities import (
    get_camplusplus_replacement_paths,
    get_download_modelscope_assets,
    get_enabled_qwen_huggingface_assets,
)


def _get_huggingface_assets():
    """Return HuggingFace assets for the current Qwen runtime plan."""
    from app.core.device import has_gpu, get_vram_gb

    hf_assets = get_enabled_qwen_huggingface_assets()
    if not hf_assets:
        print("当前部署计划未启用 Qwen3-ASR，跳过 HuggingFace 模型下载")
        return []

    if has_gpu():
        print(f"显存/内存 {get_vram_gb():.1f}GB，加载当前运行计划所需的 Qwen 模型")
    else:
        print("CPU 环境加载当前运行计划所需的 Qwen 模型（含 forced aligner）")
    return hf_assets

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
    ms_assets = get_download_modelscope_assets()
    hf_assets = _get_huggingface_assets()

    # 检查 ModelScope 模型
    for asset in ms_assets:
        exists, _ = check_model_exists(asset.model_id, source="modelscope")
        if not exists:
            missing.append((asset.model_id, asset.description, "modelscope", asset.revision))

    # 检查 HuggingFace 模型 (HF 模型暂不支持指定版本)
    for asset in hf_assets:
        exists, _ = check_model_exists(asset.model_id, source="huggingface")
        if not exists:
            missing.append((asset.model_id, asset.description, "huggingface", None))

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
        replacements = get_camplusplus_replacement_paths(str(cache_dir))

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


def download_models(
    auto_mode: bool = False,
    export_dir: Optional[str] = None,
) -> bool:
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
    ms_assets = get_download_modelscope_assets()
    hf_assets = _get_huggingface_assets()

    export_path = Path(export_dir) if export_dir else None

    if not missing:
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
        print("Qwen3-ASR 模型预下载")
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
        for asset in ms_assets:
            all_models.append((asset.model_id, "modelscope"))
        for asset in hf_assets:
            all_models.append((asset.model_id, "huggingface"))

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


def main() -> int:
    """CLI entrypoint for model download and export."""
    parser = argparse.ArgumentParser(description="Download or export Qwen3-ASR models")
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional export directory for offline deployment packaging",
    )
    parser.add_argument(
        "--auto-mode",
        action="store_true",
        help="Reduce output for startup/bootstrap usage",
    )
    args = parser.parse_args()

    success = download_models(
        auto_mode=args.auto_mode,
        export_dir=args.export_dir,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
