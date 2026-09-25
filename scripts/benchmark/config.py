# -*- coding: utf-8 -*-
"""
测试配置模块
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class TestConfig:
    """测试配置"""

    # 服务器配置
    host: str = "localhost"
    port: int = 8000
    timeout_seconds: float = 300.0  # 默认 5 分钟，并发 TTS 可能需要更长时间
    warmup_requests: int = 3

    # 并发配置
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4])

    # ASR 配置
    asr_audio_file: Optional[Path] = None
    asr_sample_rate: int = 16000
    asr_chunk_size: int = 2560  # 160ms @ 16kHz
    asr_format: str = "int16_le"

    # TTS 配置
    tts_text_count: int = 50  # 预生成的测试文本数量
    tts_text_length_range: tuple = (50, 100)  # 文本字符数范围
    tts_voice: str = "中文女"
    tts_format: str = "PCM"
    tts_sample_rate: int = 22050
    tts_chunk_interval: float = 0.05  # 发送间隔秒数 (模拟 LLM 生成速度)

    # 输出配置
    output_dir: Path = field(default_factory=lambda: Path("./benchmark_results"))
    report_name: str = "benchmark_report"

    @property
    def ws_base_url(self) -> str:
        """WebSocket 基础 URL"""
        return f"ws://{self.host}:{self.port}"

    @property
    def asr_ws_url(self) -> str:
        """ASR WebSocket URL"""
        return f"{self.ws_base_url}/v1/stream"

    @property
    def tts_ws_url(self) -> str:
        """TTS WebSocket URL"""
        return f"{self.ws_base_url}/ws/v1/tts"

    def validate(self, test_type: str = "both") -> None:
        """
        验证配置

        Args:
            test_type: 测试类型 (asr/tts/both)

        Raises:
            ValueError: 配置无效
        """
        if test_type in ("asr", "both"):
            if self.asr_audio_file is None:
                raise ValueError("ASR 测试需要提供音频文件路径 (--audio-file)")
            if not self.asr_audio_file.exists():
                raise ValueError(f"音频文件不存在: {self.asr_audio_file}")

        if not self.concurrency_levels:
            raise ValueError("至少需要一个并发级别")

        for level in self.concurrency_levels:
            if level < 1:
                raise ValueError(f"并发级别必须大于 0: {level}")

        if self.timeout_seconds <= 0:
            raise ValueError("超时时间必须大于 0")
