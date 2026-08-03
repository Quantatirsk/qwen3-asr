"""Remote vLLM transcription client used by the Ascend deployment."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

import requests

from app.services.asr.engines import ASRRawResult, ASRSegmentResult
from app.utils.text_processing import normalize_asr_text


class Qwen3RemoteVLLMBackend:
    """Offline Qwen3-ASR adapter for vLLM's OpenAI transcription API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str],
        timeout_sec: float,
        max_inference_batch_size: int,
    ) -> None:
        normalized_base_url = base_url.rstrip("/")
        if normalized_base_url.endswith("/v1"):
            self._transcriptions_url = f"{normalized_base_url}/audio/transcriptions"
        else:
            self._transcriptions_url = f"{normalized_base_url}/v1/audio/transcriptions"
        self._model = model
        self._api_key = api_key
        self._timeout_sec = timeout_sec
        self._max_inference_batch_size = max(1, max_inference_batch_size)

    def _transcribe(
        self,
        audio_path: str,
        *,
        context: str = "",
        language: Optional[str] = None,
    ) -> str:
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        data = {
            "model": self._model,
            "response_format": "json",
        }
        if context.strip():
            data["prompt"] = context.strip()
        if language:
            data["language"] = language

        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"

        try:
            with path.open("rb") as audio_file:
                response = requests.post(
                    self._transcriptions_url,
                    data=data,
                    files={"file": (path.name, audio_file, content_type)},
                    headers=headers,
                    timeout=self._timeout_sec,
                )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Remote vLLM transcription failed at {self._transcriptions_url}: {exc}"
            ) from exc

        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise RuntimeError(
                "Remote vLLM transcription returned invalid JSON from "
                f"{self._transcriptions_url}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise RuntimeError(
                "Remote vLLM transcription response from "
                f"{self._transcriptions_url} does not contain a text field"
            )
        return payload["text"]

    def transcribe_text(
        self,
        audio_path: str,
        *,
        context: str = "",
        language: Optional[str] = None,
        enable_itn: bool = False,
    ) -> str:
        text = self._transcribe(
            audio_path,
            context=context,
            language=language,
        )
        return normalize_asr_text(text, enable_itn=enable_itn)

    def transcribe_raw(
        self,
        audio_path: str,
        *,
        context: str = "",
        language: Optional[str] = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
    ) -> ASRRawResult:
        if word_timestamps:
            raise RuntimeError(
                "Remote Ascend vLLM word timestamps are not supported until the "
                "Qwen3 forced aligner passes the NPU compatibility gate"
            )
        text = self.transcribe_text(
            audio_path,
            context=context,
            language=language,
            enable_itn=enable_itn,
        )
        segments = (
            [ASRSegmentResult(text=text, start_time=0.0, end_time=0.0)] if text else []
        )
        return ASRRawResult(text=text, segments=segments)

    def transcribe_batch(
        self,
        audio_paths: list[str],
        *,
        context: str = "",
        language: Optional[str] = None,
        word_timestamps: bool = False,
        enable_itn: bool = False,
    ) -> list[ASRSegmentResult]:
        if word_timestamps:
            raise RuntimeError(
                "Remote Ascend vLLM word timestamps are not supported until the "
                "Qwen3 forced aligner passes the NPU compatibility gate"
            )

        results: list[ASRSegmentResult] = []
        for start in range(0, len(audio_paths), self._max_inference_batch_size):
            chunk = audio_paths[start : start + self._max_inference_batch_size]
            results.extend(
                ASRSegmentResult(
                    text=self.transcribe_text(
                        audio_path,
                        context=context,
                        language=language,
                        enable_itn=enable_itn,
                    ),
                    start_time=0.0,
                    end_time=0.0,
                )
                for audio_path in chunk
            )
        return results
