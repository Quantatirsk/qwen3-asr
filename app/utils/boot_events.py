# -*- coding: utf-8 -*-
"""Structured startup events for the optional terminal dashboard."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_BOOT_EVENT_PREFIX = "__FUNASR_BOOT__"


def boot_events_enabled() -> bool:
    return (os.getenv("FUNASR_BOOT_EVENTS") or "").strip() == "1"


def emit_boot_event(event: str, **payload: Any) -> None:
    if not boot_events_enabled():
        return

    data = {"event": event, **payload}
    sys.stdout.write(_BOOT_EVENT_PREFIX + json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def get_boot_event_prefix() -> str:
    return _BOOT_EVENT_PREFIX
