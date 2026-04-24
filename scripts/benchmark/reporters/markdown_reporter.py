# -*- coding: utf-8 -*-
"""
Markdown 报告生成器
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..metrics.models import AggregatedMetrics


class MarkdownReporter:
    """Markdown 报告生成器"""

    def generate(
        self,
        asr_results: List[AggregatedMetrics],
        tts_results: List[AggregatedMetrics],
        output_path: Path,
        config_info: Optional[dict] = None,
    ) -> None:
        """
        生成 Markdown 报告

        Args:
            asr_results: ASR 测试结果
            tts_results: TTS 测试结果
            output_path: 输出文件路径
            config_info: 配置信息
        """
        lines = []

        # 标题
        lines.append("# Qwen3-ASR 并发性能测试报告")
        lines.append("")
        lines.append(f"**测试时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if config_info:
            lines.append(f"**服务器:** {config_info.get('host', 'localhost')}:{config_info.get('port', 8000)}")
            lines.append(f"**并发级别:** {', '.join(map(str, config_info.get('concurrency_levels', [])))}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # ASR 结果
        if asr_results:
            lines.extend(self._generate_asr_section(asr_results))

        # TTS 结果
        if tts_results:
            lines.extend(self._generate_tts_section(tts_results))

        # 结论
        lines.extend(self._generate_conclusions(asr_results, tts_results))

        # 写入文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _generate_asr_section(self, results: List[AggregatedMetrics]) -> List[str]:
        """生成 ASR 结果部分"""
        lines = []
        lines.append("## ASR 性能测试结果")
        lines.append("")

        # 延迟指标表格
        lines.append("### 延迟指标 (毫秒)")
        lines.append("")
        lines.append("| 并发数 | 首次响应 (Avg) | 首次响应 (P95) | 总时间 (Avg) | 总时间 (P95) | 总时间 (Max) |")
        lines.append("|--------|---------------|---------------|-------------|-------------|-------------|")

        for r in results:
            lines.append(
                f"| {r.concurrency_level} | "
                f"{r.first_latency_avg:.1f} | "
                f"{r.first_latency_p95:.1f} | "
                f"{r.total_time_avg:.1f} | "
                f"{r.total_time_p95:.1f} | "
                f"{r.total_time_max:.1f} |"
            )

        lines.append("")

        # RTF 和吞吐量表格
        lines.append("### RTF 和吞吐量")
        lines.append("")
        lines.append("| 并发数 | RTF (Avg) | RTF (P95) | 吞吐量 (req/s) | 成功率 |")
        lines.append("|--------|----------|----------|---------------|--------|")

        for r in results:
            lines.append(
                f"| {r.concurrency_level} | "
                f"{r.rtf_avg:.3f} | "
                f"{r.rtf_p95:.3f} | "
                f"{r.throughput:.3f} | "
                f"{r.success_rate:.1f}% |"
            )

        lines.append("")
        return lines

    def _generate_tts_section(self, results: List[AggregatedMetrics]) -> List[str]:
        """生成 TTS 结果部分"""
        lines = []
        lines.append("## TTS 性能测试结果")
        lines.append("")

        # 延迟指标表格
        lines.append("### 延迟指标 (毫秒)")
        lines.append("")
        lines.append("| 并发数 | 首包延迟 (Avg) | 首包延迟 (P95) | 总时间 (Avg) | 总时间 (P95) | 总时间 (Max) |")
        lines.append("|--------|---------------|---------------|-------------|-------------|-------------|")

        for r in results:
            lines.append(
                f"| {r.concurrency_level} | "
                f"{r.first_latency_avg:.1f} | "
                f"{r.first_latency_p95:.1f} | "
                f"{r.total_time_avg:.1f} | "
                f"{r.total_time_p95:.1f} | "
                f"{r.total_time_max:.1f} |"
            )

        lines.append("")

        # RTF 和吞吐量表格
        lines.append("### RTF 和吞吐量")
        lines.append("")
        lines.append("| 并发数 | RTF (Avg) | RTF (P95) | 吞吐量 (req/s) | 成功率 |")
        lines.append("|--------|----------|----------|---------------|--------|")

        for r in results:
            lines.append(
                f"| {r.concurrency_level} | "
                f"{r.rtf_avg:.3f} | "
                f"{r.rtf_p95:.3f} | "
                f"{r.throughput:.3f} | "
                f"{r.success_rate:.1f}% |"
            )

        lines.append("")
        return lines

    def _generate_conclusions(
        self,
        asr_results: List[AggregatedMetrics],
        tts_results: List[AggregatedMetrics],
    ) -> List[str]:
        """生成结论部分"""
        lines = []
        lines.append("## 结论")
        lines.append("")

        if asr_results:
            max_level = max(asr_results, key=lambda x: x.concurrency_level)
            lines.append(f"- **ASR 最大并发 ({max_level.concurrency_level}) RTF:** {max_level.rtf_avg:.3f}")
            lines.append(f"- **ASR 最大并发吞吐量:** {max_level.throughput:.3f} req/s")

            # 找到 RTF 超过 1.0 的并发级别
            stable_levels = [r for r in asr_results if r.rtf_avg <= 1.0]
            if stable_levels:
                max_stable = max(stable_levels, key=lambda x: x.concurrency_level)
                lines.append(f"- **ASR 稳定并发上限 (RTF < 1.0):** {max_stable.concurrency_level}")

        if tts_results:
            max_level = max(tts_results, key=lambda x: x.concurrency_level)
            lines.append(f"- **TTS 最大并发 ({max_level.concurrency_level}) RTF:** {max_level.rtf_avg:.3f}")
            lines.append(f"- **TTS 最大并发吞吐量:** {max_level.throughput:.3f} req/s")

            stable_levels = [r for r in tts_results if r.rtf_avg <= 1.0]
            if stable_levels:
                max_stable = max(stable_levels, key=lambda x: x.concurrency_level)
                lines.append(f"- **TTS 稳定并发上限 (RTF < 1.0):** {max_stable.concurrency_level}")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*RTF (Real-Time Factor): 处理时间与音频时长的比值，小于 1.0 表示处理速度快于实时*")
        lines.append("")

        return lines
