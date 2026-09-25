# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen forced alignment text preparation and monotonic timestamp repair.

Adapted from QwenLM/Qwen3-ASR, qwen_asr/inference/qwen3_forced_aligner.py
(Qwen3ForceAlignProcessor). Keeps the upstream longest nondecreasing
subsequence and interpolation policy, with empty-input and duration guards.
Text is never reordered. Valid neighboring spans may share a boundary.
"""

import math
import unicodedata
from itertools import pairwise


def split_alignment_units(text: str) -> list[str]:
    """Use official Chinese/mixed-language units; punctuation has no voice span."""
    units = []
    for word in text.split():
        cleaned = "".join(
            c
            for c in word
            if c == "'" or unicodedata.category(c).startswith(("L", "N"))
        )
        buffer = []
        for char in cleaned:
            code = ord(char)
            is_cjk = (
                0x4E00 <= code <= 0x9FFF
                or 0x3400 <= code <= 0x4DBF
                or 0x20000 <= code <= 0x2A6DF
                or 0x2A700 <= code <= 0x2B73F
                or 0x2B740 <= code <= 0x2B81F
                or 0x2B820 <= code <= 0x2CEAF
                or 0xF900 <= code <= 0xFAFF
            )
            if is_cjk:
                if buffer:
                    units.append("".join(buffer))
                    buffer.clear()
                units.append(char)
            else:
                buffer.append(char)
        if buffer:
            units.append("".join(buffer))
    return units


def repair_timestamps(timestamps: list[float], duration_ms: float) -> list[float]:
    """Repair flattened start/end predictions without swapping text positions."""
    if not math.isfinite(duration_ms) or duration_ms < 0:
        raise ValueError("Invalid alignment audio duration")
    if any(not math.isfinite(value) for value in timestamps):
        raise ValueError("Non-finite forced alignment timestamp")
    data = [min(duration_ms, max(0.0, value)) for value in timestamps]
    if all(a <= b for a, b in pairwise(data)):
        return data

    # Match upstream's anchor selection, including ties between subsequences.
    count = len(data)
    lengths = [1] * count
    parents = [-1] * count
    for i in range(1, count):
        for j in range(i):
            if data[j] <= data[i] and lengths[j] + 1 > lengths[i]:
                lengths[i] = lengths[j] + 1
                parents[i] = j
    normal = [False] * count
    index = lengths.index(max(lengths))
    while index != -1:
        normal[index] = True
        index = parents[index]

    result = data.copy()
    index = 0
    while index < count:
        if normal[index]:
            index += 1
            continue
        end = index
        while end < count and not normal[end]:
            end += 1
        left = result[index - 1] if index else None
        right = data[end] if end < count else None
        for k in range(index, end):
            if left is None:
                result[k] = right
            elif right is None:
                result[k] = left
            elif end - index <= 2:
                result[k] = left if k - index + 1 <= end - k else right
            else:
                result[k] = left + (right - left) * (k - index + 1) / (end - index + 1)
        index = end
    # Keep fractional milliseconds: truncation could put a repaired value before
    # a neighboring anchor clipped to a fractional audio duration.
    return result
