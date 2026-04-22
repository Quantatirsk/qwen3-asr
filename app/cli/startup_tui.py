# -*- coding: utf-8 -*-
"""Textual startup dashboard that owns the terminal and streams child logs."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, ProgressBar, RichLog, Static

from app.bootstrap import run_cli_preflight
from app.core.config import settings
from app.utils.boot_events import get_boot_event_prefix

_BOOT_PREFIX = get_boot_event_prefix()


class StartupDashboard(App[int]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #summary {
        height: 2;
        padding: 0 1;
        content-align: left middle;
    }

    #status {
        height: 2;
        padding: 0 1;
        content-align: left middle;
    }

    #progress {
        height: 3;
        padding: 0 1;
    }

    #log {
        height: 1fr;
        border: round $panel;
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, server_args: list[str]):
        super().__init__()
        self._server_args = server_args
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._runner: Optional[asyncio.Task[None]] = None
        self._stream_tasks: list[asyncio.Task[None]] = []
        self._exit_code = 0
        self._service_ready = False
        self._current_phase = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            f"FunASR-API | http://{settings.HOST}:{settings.PORT} | device={settings.DEVICE} | workers=1",
            id="summary",
        )
        yield Static("准备启动...", id="status")
        yield ProgressBar(total=1, show_eta=False, id="progress")
        yield RichLog(id="log", wrap=False, markup=False, highlight=False)
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "FunASR Startup"
        self.sub_title = "Textual dashboard"
        self._runner = asyncio.create_task(self._run())

    async def action_quit(self) -> None:
        await self._terminate_child()
        self.exit(self._exit_code)

    async def _run(self) -> None:
        preflight_output = io.StringIO()
        with contextlib.redirect_stdout(preflight_output), contextlib.redirect_stderr(preflight_output):
            preflight_ok = run_cli_preflight()
        for line in preflight_output.getvalue().splitlines():
            if line.strip():
                self._log_message(line, stream="stdout")

        if not preflight_ok:
            self._log_message("模型 preflight 失败，启动终止", stream="stderr")
            self._set_status("preflight 失败")
            self._exit_code = 1
            self.exit(self._exit_code)
            return

        self._set_status("启动服务子进程...")
        env = os.environ.copy()
        env["FUNASR_TUI_CHILD"] = "1"
        env["FUNASR_BOOT_EVENTS"] = "1"
        env["FUNASR_STARTUP_UI"] = "plain"
        env["PYTHONUNBUFFERED"] = "1"

        start_path = Path(__file__).resolve().parents[2] / "start.py"
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(start_path),
            *self._server_args,
            cwd=str(Path(__file__).resolve().parents[2]),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        self._stream_tasks = [
            asyncio.create_task(self._read_stream(self._proc.stdout, "stdout")),
            asyncio.create_task(self._read_stream(self._proc.stderr, "stderr")),
        ]

        return_code = await self._proc.wait()
        await asyncio.gather(*self._stream_tasks, return_exceptions=True)
        self._stream_tasks.clear()

        self._exit_code = return_code
        if return_code == 0:
            self._set_status("服务已停止")
        else:
            self._set_status(f"服务异常退出，exit={return_code}")
            self._log_message(f"服务异常退出，exit={return_code}", stream="stderr")

        self.exit(self._exit_code)

    async def _read_stream(self, stream: Optional[asyncio.StreamReader], stream_name: str) -> None:
        if stream is None:
            return

        while True:
            line = await stream.readline()
            if not line:
                break

            decoded = line.decode("utf-8", errors="replace")
            chunks = [part for part in decoded.replace("\r", "\n").splitlines() if part.strip()]
            for chunk in chunks:
                if chunk.startswith(_BOOT_PREFIX):
                    self._handle_boot_event(chunk[len(_BOOT_PREFIX):])
                else:
                    self._log_message(chunk, stream=stream_name)

    def _handle_boot_event(self, raw_payload: str) -> None:
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            self._log_message(raw_payload, stream="stderr")
            return

        progress = self.query_one("#progress", ProgressBar)
        event = payload.get("event")

        if event == "phase_start":
            total = max(int(payload.get("total", 1)), 1)
            progress.update(total=total, progress=0)
            self._current_phase = str(payload.get("phase") or "")
            self._set_status(
                self._format_status(
                    phase=self._current_phase,
                    step=0,
                    total=total,
                    message=str(payload.get("message") or payload.get("phase") or "启动中"),
                )
            )
            return

        if event == "step_start":
            total = max(int(payload.get("total", progress.total or 1)), 1)
            step = max(int(payload.get("step", 1)), 1)
            progress.update(total=total, progress=min(step - 1, total))
            self._current_phase = str(payload.get("phase") or self._current_phase)
            self._set_status(
                self._format_status(
                    phase=self._current_phase,
                    step=step,
                    total=total,
                    message=str(payload.get("message") or "处理中"),
                )
            )
            return

        if event == "step_done":
            total = max(int(payload.get("total", progress.total or 1)), 1)
            step = max(int(payload.get("step", 1)), 1)
            progress.update(total=total, progress=min(step, total))
            self._current_phase = str(payload.get("phase") or self._current_phase)
            self._set_status(
                self._format_status(
                    phase=self._current_phase,
                    step=step,
                    total=total,
                    message=str(payload.get("message") or "完成"),
                )
            )
            return

        if event == "ready":
            progress.update(total=1, progress=1)
            self._service_ready = True
            self._set_status(str(payload.get("message") or "服务已就绪"))
            self._log_message(str(payload.get("message") or "服务已就绪"), stream="stdout", style="bold green")
            return

        if event == "error":
            self._set_status(str(payload.get("message") or "启动失败"))
            self._log_message(str(payload.get("message") or "启动失败"), stream="stderr")

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _log_message(self, message: str, stream: str, style: str | None = None) -> None:
        log = self.query_one("#log", RichLog)
        line = message.rstrip()
        if not line:
            return
        if style is None:
            style = self._style_for_message(line, stream)
        if style:
            log.write(Text(line, style=style))
        else:
            log.write(line)

    def _format_status(self, phase: str, step: int, total: int, message: str) -> str:
        phase_text = phase or "startup"
        if step <= 0:
            return f"[{phase_text}] 0/{total} {message}"
        return f"[{phase_text}] {step}/{total} {message}"

    def _style_for_message(self, message: str, stream: str) -> str | None:
        lower = message.lower()
        if "traceback" in lower or re.search(r"\b(error|critical|fatal)\b", lower):
            return "bold red"
        if re.search(r"\bwarning\b", lower) or "futurewarning" in lower:
            return "yellow"
        if "loading safetensors checkpoint shards" in lower or "completed |" in lower:
            return "cyan"
        if stream == "stderr":
            return "bright_black"
        return None

    async def _terminate_child(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return

        self._set_status("正在停止服务...")
        self._proc.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()


def run_tui(server_args: list[str]) -> int:
    app = StartupDashboard(server_args=server_args)
    result = app.run()
    return int(result or 0)
