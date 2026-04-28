# -*- coding: utf-8 -*-
"""Aliyun-compatible FunASR websocket service."""

import json
import logging
import numpy as np
import soundfile as sf
import io
from contextlib import suppress
from typing import Any, Optional, Dict, TYPE_CHECKING, cast
from enum import IntEnum

from fastapi import WebSocketDisconnect

if TYPE_CHECKING:
    from .asr.engines import FunASREngine, BaseASREngine

from ..core.config import settings
from ..core.executor import run_sync
from ..core.security import validate_websocket_token
from ..utils.text_processing import apply_itn_to_text
from ..utils.audio_filter import is_nearfield_voice
from ..models.websocket_asr import (
    AliyunASRWSHeader,
    AliyunASRNamespace,
    AliyunASRMessageName,
    AliyunASRStatus,
)
from .asr.runtime import RuntimeEngineLease, get_runtime_router
from .asr.engines import (
    get_global_punc_model,
    get_global_punc_realtime_model,
    get_punc_inference_lock,
    get_punc_realtime_inference_lock,
)

logger = logging.getLogger(__name__)


class ConnectionState(IntEnum):
    """连接状态"""

    READY = 1
    STARTED = 2


class AliyunWebSocketASRService:
    """阿里云WebSocket实时ASR服务"""

    def __init__(self):
        self.asr_engine = None
        self.engine_lease: Optional[RuntimeEngineLease] = None

    async def cleanup(self):
        """清理资源"""
        try:
            if self.engine_lease is not None:
                await self.engine_lease.close()
                self.engine_lease = None
            if self.asr_engine is not None:
                self.asr_engine = None
                logger.info("WebSocket ASR引擎资源已清理")
        except Exception as e:
            logger.warning(f"清理WebSocket ASR资源异常: {e}")

    def _ensure_asr_engine(self) -> "BaseASREngine":
        """确保ASR引擎已加载"""
        assert self.asr_engine is not None, "ASR引擎初始化失败"
        return self.asr_engine

    async def handle_connection(self, websocket, task_id: str):
        """处理WebSocket连接"""
        state = ConnectionState.READY
        session_id = f"session_{task_id}"
        transcription_params = None
        audio_cache = {}
        sentence_index = 0
        audio_time = 0
        sentence_active = False
        sentence_start_time = 0
        last_sentence_text = ""
        sentence_texts = []
        sentence_texts_raw = []
        empty_result_count = 0
        audio_buffer = np.array([], dtype=np.float32)  # 音频缓冲区，用于累积到完整chunk
        max_buffer_size = settings.WS_MAX_BUFFER_SIZE  # 最大缓冲区大小（样本数）

        logger.info(f"[{task_id}] WebSocket ASR连接开始")

        try:
            result, message = validate_websocket_token(websocket, task_id)
            if not result:
                await self._send_task_failed(websocket, task_id, message)
                return

            self.engine_lease = await get_runtime_router().acquire_engine("paraformer-large")
            self.asr_engine = self.engine_lease.engine
            if not self.asr_engine.supports_realtime:
                raise Exception("当前ASR引擎不支持实时识别")
            logger.info("WebSocket ASR引擎: 使用 paraformer-large")

            while True:
                message = await websocket.receive()

                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        logger.debug(
                            f"[{task_id}] 收到消息: {data.get('header', {}).get('name', '')}"
                        )

                        header = data.get("header", {})
                        message_name = header.get("name", "")
                        message_task_id = header.get("task_id", "")
                        namespace = header.get("namespace", "")

                        if namespace != AliyunASRNamespace.SPEECH_TRANSCRIBER:
                            await self._send_task_failed(
                                websocket, task_id, "Invalid namespace"
                            )
                            continue

                        if message_name == AliyunASRMessageName.START_TRANSCRIPTION:
                            if state == ConnectionState.READY:
                                transcription_params = self._parse_start_transcription(
                                    data, task_id
                                )
                                task_id = message_task_id or task_id
                                await self._send_transcription_started(
                                    websocket, task_id, session_id
                                )
                                state = ConnectionState.STARTED
                                sentence_index = 0
                                audio_time = 0
                                sentence_active = False
                                sentence_start_time = 0
                                last_sentence_text = ""
                                sentence_texts = []
                                sentence_texts_raw = []
                                empty_result_count = 0
                            else:
                                await self._send_task_failed(
                                    websocket, task_id, "Connection already started"
                                )

                        elif message_name == AliyunASRMessageName.STOP_TRANSCRIPTION:
                            if state == ConnectionState.STARTED:
                                if message_task_id != task_id:
                                    await self._send_task_failed(
                                        websocket, task_id, "Task ID not match"
                                    )
                                    continue

                                # 确保 transcription_params 已设置
                                if not transcription_params:
                                    await self._send_task_failed(
                                        websocket, task_id, "StartTranscription not received"
                                    )
                                    continue

                                # 如果有未完成的句子，直接结束
                                if sentence_active and sentence_texts_raw:
                                    sentence_index += 1
                                    full_sentence_text = "".join(sentence_texts_raw)

                                    if transcription_params.get(
                                        "enable_punctuation_prediction", True
                                    ):
                                        full_sentence_text = await self._apply_final_punctuation_to_sentence(
                                            full_sentence_text, task_id
                                        )

                                    await self._send_sentence_end(
                                        websocket,
                                        task_id,
                                        sentence_index,
                                        audio_time,
                                        full_sentence_text,
                                        sentence_start_time,
                                        enable_itn=transcription_params.get(
                                            "enable_inverse_text_normalization", True
                                        ),
                                    )

                                await self._send_transcription_completed(
                                    websocket, task_id
                                )
                                logger.info(f"[{task_id}] 识别完成")
                                break
                            else:
                                await self._send_task_failed(
                                    websocket, task_id, "Connection not started"
                                )
                        else:
                            await self._send_task_failed(
                                websocket,
                                task_id,
                                f"Invalid message name: {message_name}",
                            )

                    except json.JSONDecodeError as e:
                        logger.error(f"[{task_id}] JSON解析错误: {e}")
                        await self._send_task_failed(
                            websocket, task_id, f"Message Not Json: {message}"
                        )
                    except WebSocketDisconnect:
                        # 客户端断开，向外层抛出
                        logger.debug(f"[{task_id}] 处理消息时检测到客户端断开")
                        raise
                    except Exception as e:
                        logger.error(f"[{task_id}] 处理消息异常: {e}")
                        await self._send_task_failed(websocket, task_id, str(e))
                        break

                elif "bytes" in message:
                    if state == ConnectionState.STARTED:
                        audio_bytes = message["bytes"]

                        if not transcription_params:
                            await self._send_task_failed(
                                websocket, task_id, "StartTranscription not received"
                            )
                            continue

                        try:
                            # 将接收到的音频添加到缓冲区
                            audio_format = transcription_params.get("format", "pcm")
                            sample_rate = transcription_params.get("sample_rate", 16000)
                            incoming_audio = self._convert_audio_bytes_to_array(
                                audio_bytes, audio_format, sample_rate, task_id
                            )

                            audio_buffer = np.concatenate([audio_buffer, incoming_audio])
                            if len(audio_buffer) > max_buffer_size:
                                logger.warning(
                                    f"[{task_id}] WebSocket音频缓冲区超过限制，保留最近 {max_buffer_size} samples"
                                )
                                audio_buffer = audio_buffer[-max_buffer_size:]

                            standard_chunk_sizes = [3840, 9600]
                            selected_chunk_size = None
                            for chunk_size in sorted(standard_chunk_sizes, reverse=True):
                                if len(audio_buffer) >= chunk_size:
                                    selected_chunk_size = chunk_size
                                    break

                            if selected_chunk_size is None:
                                continue

                            while len(audio_buffer) >= selected_chunk_size:
                                chunk_start_time = audio_time

                                audio_chunk = audio_buffer[:selected_chunk_size]
                                audio_buffer = audio_buffer[selected_chunk_size:]

                                effective_rms_threshold = settings.ASR_NEARFIELD_RMS_THRESHOLD
                                if sentence_active:
                                    effective_rms_threshold = settings.ASR_NEARFIELD_RMS_THRESHOLD * 0.6

                                is_nearfield, filter_metrics = is_nearfield_voice(
                                    audio_chunk,
                                    sample_rate=sample_rate,
                                    rms_threshold=effective_rms_threshold,
                                    enable_filter=settings.ASR_ENABLE_NEARFIELD_FILTER,
                                )

                                is_sentence_end = False
                                if not is_nearfield:
                                    if logger.isEnabledFor(logging.DEBUG):
                                        logger.debug(
                                            f"[{task_id}] 远场声音已过滤 - "
                                            f"RMS: {filter_metrics['rms_energy']:.6f} (阈值: {effective_rms_threshold:.6f})"
                                        )

                                    chunk_duration_ms = int(len(audio_chunk) / sample_rate * 1000)
                                    audio_time += chunk_duration_ms

                                    if sentence_active:
                                        result_text = ""
                                        result_text_raw = ""
                                        is_silence_frame = False
                                    else:
                                        continue

                                else:
                                    if logger.isEnabledFor(logging.DEBUG) and filter_metrics.get('enabled', True):
                                        logger.debug(
                                            f"[{task_id}] 近场声音检测通过 - "
                                            f"RMS: {filter_metrics['rms_energy']:.6f} (阈值: {effective_rms_threshold:.6f})"
                                        )

                                    (
                                        result_text,
                                        result_text_raw,
                                        is_silence_frame,
                                        audio_cache,
                                        audio_time,
                                    ) = await self._process_audio_chunk(
                                        audio_chunk,
                                        audio_cache,
                                        transcription_params,
                                        audio_time,
                                        task_id,
                                        is_final=False,
                                    )

                                max_empty_count = max(
                                    3,
                                    (
                                        transcription_params.get(
                                            "max_sentence_silence", 800
                                        )
                                        * 2
                                    )
                                    // 600,
                                )

                                if not result_text:
                                    empty_result_count += 1
                                    if (
                                        sentence_active
                                        and empty_result_count >= max_empty_count
                                    ):
                                        is_sentence_end = True
                                        logger.debug(
                                            f"[{task_id}] 连续空结果，判断句子结束"
                                        )
                                else:
                                    empty_result_count = 0

                                # 检测到静音帧且当前有正在识别的句子，触发SentenceEnd
                                if (
                                    is_silence_frame
                                    and sentence_active
                                    and sentence_texts_raw
                                ):
                                    is_sentence_end = True
                                    logger.debug(f"[{task_id}] 检测到静音帧，判断句子结束")

                                if is_sentence_end and sentence_active:
                                    (
                                        _,
                                        flush_result_text_raw,
                                        _,
                                        audio_cache,
                                        audio_time,
                                    ) = await self._process_audio_chunk(
                                        np.array([], dtype=np.float32),
                                        audio_cache,
                                        transcription_params,
                                        audio_time,
                                        task_id,
                                        is_final=True,
                                    )

                                    if flush_result_text_raw:
                                        if (
                                            not sentence_texts_raw
                                            or flush_result_text_raw
                                            != sentence_texts_raw[-1]
                                        ):
                                            sentence_texts_raw.append(flush_result_text_raw)

                                    sentence_index += 1
                                    sentence_duration = audio_time - sentence_start_time
                                    full_sentence_text = "".join(sentence_texts_raw)

                                    if transcription_params.get(
                                        "enable_punctuation_prediction", True
                                    ):
                                        full_sentence_text = (
                                            await self._apply_final_punctuation_to_sentence(
                                                full_sentence_text, task_id
                                            )
                                        )

                                    logger.debug(
                                        f"[{task_id}] 句子结束 #{sentence_index}: '{full_sentence_text}' "
                                        f"({sentence_duration}ms)"
                                    )
                                    await self._send_sentence_end(
                                        websocket,
                                        task_id,
                                        sentence_index,
                                        audio_time,
                                        full_sentence_text,
                                        sentence_start_time,
                                        enable_itn=transcription_params.get(
                                            "enable_inverse_text_normalization", True
                                        ),
                                    )
                                    sentence_active = False
                                    sentence_start_time = 0
                                    last_sentence_text = ""
                                    sentence_texts = []
                                    sentence_texts_raw = []
                                    empty_result_count = 0
                                    audio_cache = {}
                                elif result_text:
                                    if result_text != last_sentence_text:
                                        last_sentence_text = result_text
                                        if (
                                            not sentence_texts
                                            or result_text != sentence_texts[-1]
                                        ):
                                            sentence_texts.append(result_text)
                                        if (
                                            not sentence_texts_raw
                                            or result_text_raw != sentence_texts_raw[-1]
                                        ):
                                            sentence_texts_raw.append(result_text_raw)

                                        if not sentence_active:
                                            sentence_active = True
                                            sentence_start_time = chunk_start_time
                                            sentence_texts = [result_text]
                                            sentence_texts_raw = [result_text_raw]
                                            empty_result_count = 0
                                            logger.debug(
                                                f"[{task_id}] 句子开始 #{sentence_index + 1}"
                                            )
                                            await self._send_sentence_begin(
                                                websocket,
                                                task_id,
                                                sentence_index + 1,
                                                sentence_start_time,
                                            )

                                        if transcription_params.get(
                                            "enable_intermediate_result", True
                                        ):
                                            # 发送当前句子的累计完整文本（去重拼接）
                                            accumulated_text = "".join(sentence_texts)
                                            await self._send_transcription_result_changed(
                                                websocket,
                                                task_id,
                                                sentence_index + 1,
                                                audio_time,
                                                accumulated_text,
                                            )

                        except WebSocketDisconnect:
                            # 客户端断开，向外层抛出
                            logger.debug(f"[{task_id}] 音频处理时检测到客户端断开")
                            raise
                        except Exception as e:
                            logger.error(f"[{task_id}] 音频处理异常: {e}")
                            await self._send_task_failed(
                                websocket, task_id, f"Audio processing failed: {str(e)}"
                            )
                    else:
                        await self._send_task_failed(
                            websocket, task_id, "Connection not started"
                        )

        except WebSocketDisconnect:
            logger.warning(f"[{task_id}] 客户端主动断开WebSocket连接")
        except Exception as e:
            # 检查是否是WebSocket连接相关的异常
            error_msg = str(e)
            if (
                "Cannot call \"receive\" once a disconnect message has been received" in error_msg
                or "WebSocket is not connected" in error_msg
                or "Need to call \"accept\" first" in error_msg
            ):
                logger.warning(f"[{task_id}] WebSocket连接已断开: {e}")
            else:
                logger.error(f"[{task_id}] WebSocket ASR连接处理异常: {e}")
                with suppress(Exception):
                    await self._send_task_failed(websocket, task_id, str(e))

    def _parse_start_transcription(self, data: dict, task_id: str) -> dict:
        """解析StartTranscription消息参数"""
        payload = data.get("payload", {})
        params = {
            "format": payload.get("format", "pcm"),
            "sample_rate": payload.get("sample_rate", 16000),
            "enable_intermediate_result": payload.get(
                "enable_intermediate_result", True
            ),
            "enable_punctuation_prediction": payload.get(
                "enable_punctuation_prediction", True
            ),
            "enable_inverse_text_normalization": payload.get(
                "enable_inverse_text_normalization", True
            ),
            "max_sentence_silence": payload.get("max_sentence_silence", 800),
        }
        logger.info(f"[{task_id}] StartTranscription参数解析成功: {params}")
        return params

    def _is_silence_frame(
        self, audio_array: np.ndarray, threshold: float = 0.001
    ) -> bool:
        """检测音频帧是否为静音（优化版）

        Args:
            audio_array: 音频数据数组（float32，范围-1.0到1.0）
            threshold: 静音阈值，低于此值视为静音

        Returns:
            True表示静音帧，False表示有效语音
        """
        if len(audio_array) == 0:
            return True

        # 使用更快的最大振幅检测，避免计算RMS
        max_amplitude = np.max(np.abs(audio_array))

        # 如果最大振幅都很低，直接判定为静音，无需进一步计算
        return max_amplitude < threshold * 2

    @staticmethod
    def _should_apply_realtime_punc(text: str) -> bool:
        """Realtime PUNC currently uses a Chinese punctuation model."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    async def _process_audio_chunk(
        self,
        audio_array: np.ndarray,
        cache: Dict,
        params: dict,
        current_audio_time: int,
        task_id: str,
        is_final: bool = False,
    ) -> tuple[str, str, bool, Dict, int]:
        """处理音频块，返回带标点文本、无标点文本、是否静音帧、缓存、音频时长"""
        try:
            asr_engine = self._ensure_asr_engine()

            sample_rate_value = params.get("sample_rate", 16000)
            if isinstance(sample_rate_value, (list, tuple)):
                sample_rate_value = sample_rate_value[0] if sample_rate_value else 16000
            if isinstance(sample_rate_value, str):
                sample_rate_value = sample_rate_value.strip()
                if sample_rate_value.isdigit():
                    sample_rate_value = int(sample_rate_value)
                else:
                    raise Exception(f"无效的采样率参数: {sample_rate_value}")
            try:
                sample_rate = int(sample_rate_value)
            except (TypeError, ValueError):
                raise Exception(f"无效的采样率类型: {sample_rate_value}")

            audio_array = np.asarray(audio_array, dtype=np.float32)

            chunk_duration_ms = int(len(audio_array) / sample_rate * 1000)
            new_audio_time = current_audio_time + chunk_duration_ms

            # 计算音频能量用于调试
            max_amplitude = np.max(np.abs(audio_array)) if len(audio_array) > 0 else 0
            mean_amplitude = np.mean(np.abs(audio_array)) if len(audio_array) > 0 else 0
            logger.debug(
                f"[{task_id}] 音频块信息: samples={len(audio_array)}, "
                f"duration={chunk_duration_ms}ms, max={max_amplitude:.4f}, mean={mean_amplitude:.6f}"
            )

            # 只在音频块足够大（>=400ms）时才检测静音帧，避免对小块音频进行检测增加延迟
            # 静音帧检测主要用于主动结束句子，不需要对每个小块都检测
            is_silence = False
            if chunk_duration_ms >= 400:
                is_silence = self._is_silence_frame(audio_array)
                logger.debug(f"[{task_id}] 静音帧检测: is_silence={is_silence}")

            # 根据实际音频样本数自适应调整chunk_size
            # chunk_stride = chunk_size[1], FunASR期望: samples ≈ chunk_stride * 960
            # 支持的标准chunk_stride: 4 (3840 samples, 240ms), 10 (9600 samples, 600ms)
            num_samples = len(audio_array)

            # 计算最接近的chunk_stride
            if num_samples == 3840:
                chunk_stride = 4
            elif num_samples == 9600:
                chunk_stride = 10
            else:
                # 自动选择最接近的标准stride
                chunk_stride = round(num_samples / 960)
                chunk_stride = max(4, min(chunk_stride, 10))  # 限制在4-10之间

            chunk_size = [0, chunk_stride, 5]
            encoder_chunk_look_back = 4
            decoder_chunk_look_back = 1

            logger.debug(
                f"[{task_id}] 使用chunk_size={chunk_size} (samples={num_samples}, "
                f"stride={chunk_stride}, expected={chunk_stride * 960})"
            )

            # 使用线程池执行模型推理，避免阻塞事件循环
            # 将 asr_engine 转换为 FunASREngine 以访问 realtime_model
            funasr_engine = cast("FunASREngine", asr_engine)
            realtime_model = funasr_engine.realtime_model
            if realtime_model is None:
                raise Exception("实时模型未加载")
            # 主ASR推理加全局锁，避免并发连接串音
            result = await run_sync(
                realtime_model.generate,
                input=audio_array,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back,
            )

            logger.debug(f"[{task_id}] ASR模型返回结果: {result}")

            result_text_raw = ""
            result_text_with_punc = ""

            if result and len(result) > 0:
                result_text_raw = result[0].get("text", "").strip()
                result_text_with_punc = result_text_raw

                # 如果启用实时标点且用户要求标点，则应用实时标点恢复（仅用于中间结果展示）
                if (
                    result_text_raw
                    and settings.ASR_ENABLE_REALTIME_PUNC
                    and params.get("enable_punctuation_prediction", True)
                    and not params.get("_disable_realtime_punc", False)
                    and self._should_apply_realtime_punc(result_text_raw)
                ):
                    try:
                        punc_realtime_model = get_global_punc_realtime_model(
                            asr_engine.device
                        )
                        if punc_realtime_model:
                            def _apply_realtime_punc():
                                with get_punc_realtime_inference_lock():
                                    return punc_realtime_model.generate(
                                        input=result_text_raw,
                                        cache={},
                                    )

                            punc_result = await run_sync(_apply_realtime_punc)
                            if punc_result and len(punc_result) > 0:
                                result_text_with_punc = (
                                    punc_result[0].get("text", result_text_raw).strip()
                                )
                    except Exception as e:
                        params["_disable_realtime_punc"] = True
                        logger.warning(f"[{task_id}] 实时标点恢复失败: {e}")

            if result_text_with_punc:
                logger.debug(f"[{task_id}] 识别: '{result_text_with_punc}'")

            return (
                result_text_with_punc,
                result_text_raw,
                is_silence,
                cache,
                new_audio_time,
            )

        except Exception as e:
            logger.exception(f"[{task_id}] 音频块处理失败: {e}")
            raise e

    async def _apply_final_punctuation_to_sentence(
        self, text: str, task_id: str
    ) -> str:
        """对完整句子应用最终标点恢复（使用离线标点模型添加完整标点包括句末标点）"""
        if not text:
            return text

        try:
            asr_engine = self._ensure_asr_engine()
            punc_model = get_global_punc_model(asr_engine.device)

            if punc_model is None:
                logger.info(f"[{task_id}] 标点模型未加载，返回原文本")
                return text

            logger.debug(f"[{task_id}] 应用标点恢复: '{text}'")
            def _apply_final_punc():
                with get_punc_inference_lock():
                    return punc_model.generate(input=text, cache={})

            result = await run_sync(_apply_final_punc)

            if result and len(result) > 0:
                punctuated_text = result[0].get("text", text).strip()
                logger.debug(f"[{task_id}] 标点恢复结果: '{punctuated_text}'")
                return punctuated_text
            else:
                return text

        except Exception as e:
            logger.warning(f"[{task_id}] 标点恢复失败: {e}")
            return text

    def _convert_audio_bytes_to_array(
        self, audio_bytes: bytes, audio_format: str, sample_rate: int, task_id: str
    ) -> np.ndarray:
        """将音频字节转换为numpy数组

        Args:
            audio_bytes: 音频字节数据
            audio_format: 音频格式（pcm或wav）
            sample_rate: 采样率
            task_id: 任务ID

        Returns:
            float32的numpy数组，范围-1.0到1.0
        """
        if audio_format == "pcm":
            audio_array = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
        elif audio_format == "wav":
            audio_io = io.BytesIO(audio_bytes)
            audio_array, sr = sf.read(audio_io)
            if sr != sample_rate:
                logger.warning(f"[{task_id}] WAV采样率 {sr} 与配置 {sample_rate} 不一致")
        else:
            raise Exception(f"暂不支持的音频格式: {audio_format}")

        return np.asarray(audio_array, dtype=np.float32)

    def _build_event(
        self,
        task_id: str,
        name: str,
        payload: Optional[dict[str, Any]] = None,
        status: int = AliyunASRStatus.SUCCESS,
        status_text: Optional[str] = None,
    ) -> dict[str, Any]:
        response = {
            "header": {
                "message_id": AliyunASRWSHeader.generate_message_id(),
                "task_id": task_id,
                "namespace": AliyunASRNamespace.SPEECH_TRANSCRIBER,
                "name": name,
                "status": status,
            }
        }
        if status == AliyunASRStatus.SUCCESS:
            response["header"]["status_message"] = "GATEWAY|SUCCESS|Success."
        elif status_text:
            response["header"]["status_text"] = status_text
        if payload is not None:
            response["payload"] = payload
        return response

    async def _send_event(self, websocket, task_id: str, response: dict) -> None:
        try:
            await websocket.send_text(json.dumps(response, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"[{task_id}] 发送WebSocket事件失败，客户端可能已断开: {e}")
            raise WebSocketDisconnect()

    async def _send_transcription_started(
        self, websocket, task_id: str, session_id: str
    ):
        await self._send_event(
            websocket,
            task_id,
            self._build_event(
                task_id,
                AliyunASRMessageName.TRANSCRIPTION_STARTED,
                {"session_id": session_id},
            ),
        )

    async def _send_sentence_begin(
        self, websocket, task_id: str, index: int, time: int
    ):
        await self._send_event(
            websocket,
            task_id,
            self._build_event(
                task_id,
                AliyunASRMessageName.SENTENCE_BEGIN,
                {"index": index, "time": time},
            ),
        )

    async def _send_transcription_result_changed(
        self, websocket, task_id: str, index: int, time: int, result: str
    ):
        await self._send_event(
            websocket,
            task_id,
            self._build_event(
                task_id,
                AliyunASRMessageName.TRANSCRIPTION_RESULT_CHANGED,
                {"index": index, "time": time, "result": result},
            ),
        )

    async def _send_sentence_end(
        self,
        websocket,
        task_id: str,
        index: int,
        time: int,
        result: str,
        begin_time: int = 0,
        enable_itn: bool = False,
    ):
        if enable_itn and result:
            logger.debug(f"[{task_id}] 应用ITN: {result}")
            result = apply_itn_to_text(result)
            logger.debug(f"[{task_id}] ITN结果: {result}")

        await self._send_event(
            websocket,
            task_id,
            self._build_event(
                task_id,
                AliyunASRMessageName.SENTENCE_END,
                {
                    "index": index,
                    "time": time,
                    "result": result,
                    "begin_time": begin_time,
                },
            ),
        )

    async def _send_transcription_completed(self, websocket, task_id: str):
        await self._send_event(
            websocket,
            task_id,
            self._build_event(task_id, AliyunASRMessageName.TRANSCRIPTION_COMPLETED),
        )

    async def _send_task_failed(self, websocket, task_id: str, reason: str):
        with suppress(Exception):
            await websocket.send_text(
                json.dumps(
                    self._build_event(
                        task_id,
                        AliyunASRMessageName.TASK_FAILED,
                        status=AliyunASRStatus.TASK_FAILED,
                        status_text=reason,
                    ),
                    ensure_ascii=False,
                )
            )
            logger.error(f"[{task_id}] 发送TaskFailed: {reason}")
