"""R2T2 native WebSocket benchmark client."""

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from .base_client import BaseWebSocketClient
from ..metrics.models import ASRMetrics
from app.services.realtime.protocol import MODEL_ID, supervise


class ASRWebSocketClient(BaseWebSocketClient):
    def __init__(
        self,
        ws_url: str,
        audio_data: bytes,
        audio_duration_ms: float,
        sample_rate: int = 16000,
        chunk_size: int = 2560,
        timeout: float = 120.0,
        save_result_dir: Optional[Path] = None,
    ):
        super().__init__(ws_url, timeout)
        if sample_rate != 16000 or not 1 <= chunk_size <= sample_rate:
            raise ValueError(
                "R2T2 requires 16 kHz int16 PCM and frames <= 1 second"
            )
        self.audio_data = audio_data
        self.audio_duration_ms = audio_duration_ms
        self.sample_rate = sample_rate
        self.chunk_bytes = chunk_size * 2
        self.save_result_dir = save_result_dir

    async def run_test(self) -> ASRMetrics:
        metrics = ASRMetrics(
            request_id=self.task_id,
            concurrency_level=0,
            start_time=time.perf_counter(),
            audio_duration_ms=self.audio_duration_ms,
        )
        try:
            await asyncio.wait_for(self._session(metrics), self.timeout)
            metrics.success = True
        except Exception as error:
            metrics.error_message = str(error) or type(error).__name__
        finally:
            await self.close()
        return metrics

    async def _session(self, metrics):
        await self.connect()
        await self.send_json({})
        ready = await self.receive_json()
        if not ready or not ready.get("ready") or ready.get("model") != MODEL_ID:
            raise RuntimeError(str(ready))
        started = time.perf_counter()

        async def send():
            for offset in range(0, len(self.audio_data), self.chunk_bytes):
                end = min(offset + self.chunk_bytes, len(self.audio_data))
                await asyncio.sleep(
                    max(0, started + end / (16000 * 2) - time.perf_counter())
                )
                await self.send_bytes(self.audio_data[offset:end])
            await self.websocket.send("end")
            await asyncio.Future()  # Receiver owns successful completion.

        async def receive():
            while True:
                event = json.loads(await self.receive())
                if event.get("error"):
                    raise RuntimeError(event["error"])
                if event.get("delta") and metrics.first_result_time is None:
                    metrics.first_result_time = time.perf_counter()
                if event.get("done"):
                    metrics.complete_time = time.perf_counter()
                    metrics.sentence_end_time = metrics.complete_time
                    metrics.result_text = event["text"]
                    if self.save_result_dir:
                        self.save_result_dir.mkdir(parents=True, exist_ok=True)
                        (self.save_result_dir / (self.task_id + ".json")).write_text(
                            json.dumps(event, ensure_ascii=False), encoding="utf-8"
                        )
                    return

        await supervise(send(), receive())
