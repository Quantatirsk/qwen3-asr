"""Manage API and inference processes in one shared Python environment."""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("single-container")
ENGINE_URL = "http://127.0.0.1:8001/health"
API_URL = "http://127.0.0.1:8000/stream/v1/asr/health"


@dataclass
class Service:
    name: str
    command: list[str]
    url: str
    ready_key: str
    env: dict[str, str]


def healthy(url: str, ready_key: str) -> bool:
    try:
        headers = {}
        if url == API_URL and os.environ.get("API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["API_KEY"]
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.load(response)
            return (
                response.status == 200
                and isinstance(payload, dict)
                and payload.get(ready_key) is True
            )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def check_children(children: list[tuple[str, subprocess.Popen]]) -> None:
    for name, process in children:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"{name} exited unexpectedly with status {code}")


def stop_child(name: str, process: subprocess.Popen, grace_seconds: float) -> None:
    logger.info("Stopping %s (pid=%s)", name, process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        logger.warning("%s did not stop in time; killing its process group", name)
    finally:
        # Also reap descendants if a failed group leader left inference workers.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run(
    services: list[Service],
    stop: threading.Event,
    *,
    startup_timeout: float = 600,
    grace_seconds: float = 20,
) -> int:
    children = []
    try:
        for service in services:
            if stop.is_set():
                return 0
            logger.info("Starting %s", service.name)
            process = subprocess.Popen(
                service.command, env=service.env, start_new_session=True
            )
            children.append((service.name, process))
            deadline = time.monotonic() + startup_timeout
            while not stop.is_set():
                check_children(children)
                if healthy(service.url, service.ready_key):
                    logger.info("%s ready", service.name)
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{service.name} startup timed out")
                stop.wait(0.5)
        if not stop.is_set():
            logger.info("Single-container ASR ready on port 8000")
        while not stop.wait(0.5):
            check_children(children)
        return 0
    except Exception:
        logger.exception("ASR service failed; stopping the whole container")
        return 1
    finally:
        # Stop admissions before shutting down the model and releasing GPU memory.
        for name, process in reversed(children):
            stop_child(name, process, grace_seconds)


def services() -> list[Service]:
    engine_env = dict(os.environ)
    engine_env.pop("PYTHONHOME", None)
    engine_env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "VLLM_PLUGINS": "",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
    api_env = dict(os.environ)
    api_env.pop("PYTHONHOME", None)
    api_env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "VLLM_PLUGINS": "",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "R2T2_URL": "http://127.0.0.1:8001",
        }
    )
    return [
        Service(
            "r2t2",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.services.realtime.server:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                "8001",
                "--ws-max-size",
                "65536",
                "--ws-max-queue",
                "8",
                "--ws-ping-interval",
                "10",
                "--ws-ping-timeout",
                "20",
            ],
            ENGINE_URL,
            "ready",
            engine_env,
        ),
        Service(
            "api",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--ws-max-size",
                "65536",
                "--ws-max-queue",
                "8",
            ],
            API_URL,
            "model_loaded",
            api_env,
        ),
    ]


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    if not (os.environ.get("HF_ENDPOINT") or "").strip():
        os.environ.pop("HF_ENDPOINT", None)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args()
    if args.healthcheck:
        return (
            0
            if healthy(ENGINE_URL, "ready") and healthy(API_URL, "model_loaded")
            else 1
        )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s"
    )
    from app.core.config import settings
    from app.core.device import detect_device

    try:
        device = detect_device(settings.DEVICE)
    except (RuntimeError, ValueError) as exc:
        logger.error("Invalid inference device: %s", exc)
        return 1
    os.environ["DEVICE"] = device
    logger.info(
        "Inference device: %s; shared offline/streaming R2T2 and independent aligner",
        device,
    )
    from app.bootstrap import ensure_models_downloaded

    if not ensure_models_downloaded():
        return 1
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    return run(services(), stop)


if __name__ == "__main__":
    raise SystemExit(main())
