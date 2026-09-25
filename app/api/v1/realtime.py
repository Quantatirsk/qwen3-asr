"""R2T2 WebSocket transcription and capability discovery."""

from pathlib import Path

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse

from app.core.security import validate_openai_token
from app.services.realtime.client import get_capabilities
from app.services.realtime.gateway import handle_stream
from app.services.realtime.protocol import StreamError

router = APIRouter(tags=["R2T2 Realtime"])
STATIC = Path(__file__).resolve().parents[2] / "static"


@router.get("/realtime", include_in_schema=False)
async def page():
    return FileResponse(STATIC / "realtime.html")


@router.get("/realtime/recorder.js", include_in_schema=False)
async def recorder():
    return FileResponse(STATIC / "recorder.js", media_type="text/javascript")


@router.websocket("/v1/stream")
async def stream(websocket: WebSocket):
    await handle_stream(websocket)


@router.get("/v1/config")
async def config(request: Request):
    if not validate_openai_token(request)[0]:
        return JSONResponse(
            {"error": "Invalid authentication", "code": "invalid_api_key"},
            status_code=401,
        )
    try:
        return await get_capabilities()
    except StreamError as error:
        return JSONResponse(
            {"error": str(error), "code": error.code}, status_code=error.status
        )
