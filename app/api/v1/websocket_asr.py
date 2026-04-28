# -*- coding: utf-8 -*-
"""WebSocket ASR API routes."""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ...services.qwen3_websocket_asr import Qwen3ASRService
from ...services.websocket_asr import AliyunWebSocketASRService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws/v1/asr", tags=["WebSocket ASR"])


@router.websocket("")
@router.websocket("/funasr")
async def funasr_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    service = AliyunWebSocketASRService()
    task_id = f"funasr_ws_{int(time.time())}_{id(websocket)}"

    try:
        await service.handle_connection(websocket, task_id)
    except WebSocketDisconnect:
        logger.info("[%s] Client disconnected", task_id)
    except Exception as exc:
        logger.error("[%s] Connection error: %s", task_id, exc)
    finally:
        await service.cleanup()


_qwen3_service = Qwen3ASRService()


@router.websocket("/qwen")
async def qwen_asr_websocket(
    websocket: WebSocket,
    task_id: Optional[str] = None,
) -> None:
    if task_id is None:
        task_id = str(uuid.uuid4())[:8]
    await _qwen3_service.handle_connection(websocket, task_id)
