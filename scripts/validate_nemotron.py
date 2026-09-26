"""Validate the isolated GPU API and export private listening artifacts.

Run with python -m scripts.validate_nemotron; authentication uses API_KEY only.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import html
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import httpx

from scripts.benchmark.realtime_smoke import main as realtime_smoke
from scripts.benchmark.realtime_smoke import text_distance


def normalize(text: str) -> str:
    return "".join(char for char in text.casefold() if char.isalnum())


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def audit(payload: dict, words_enabled: bool, diarization_enabled: bool) -> dict:
    duration = payload["duration"]
    segments = payload["segments"]
    words = payload.get("words") or []
    raw = payload.get("speaker_segments") or []
    assert math.isfinite(duration) and duration > 0, "Invalid audio duration"
    assert payload["text"] and segments, "Empty speech transcription"
    assert normalize("".join(s["text"] for s in segments)) == normalize(
        payload["text"]
    ), "Segment text does not reconstruct full text"

    previous_end = 0.0
    for segment in segments:
        start, end = segment["start"], segment["end"]
        assert all(math.isfinite(value) for value in (start, end))
        assert -0.003 <= start <= end <= duration + 0.003, "Segment outside audio"
        assert start >= previous_end - 0.003, "Overlapping ASR segments"
        previous_end = end
        assert not (
            segment.get("speaker") and segment.get("speaker_candidates")
        ), "An uncertain segment was assigned to one speaker"
    if words_enabled:
        assert words, "Word timestamps were requested but absent"
        assert normalize("".join(w["word"] for w in words)) == normalize(
            payload["text"]
        ), "Aligned words lost or added text"
    else:
        assert not words, "Internal alignment leaked into public word timestamps"
    previous_end = 0.0
    segment_index = 0
    spans: set[tuple[str, float, float]] = set()
    for word in words:
        start, end = word["start"], word["end"]
        assert all(math.isfinite(value) for value in (start, end))
        assert -0.003 <= start <= end <= duration + 0.003, "Word outside audio"
        assert start >= previous_end - 0.003, "Word timestamps moved backwards"
        previous_end = end
        while (
            segment_index + 1 < len(segments)
            and segments[segment_index]["end"] < end - 0.003
        ):
            segment_index += 1
        segment = segments[segment_index]
        assert (
            segment["start"] - 0.003 <= start <= end <= segment["end"] + 0.003
        ), "Word escaped its segment; possible relative/absolute offset error"
        if end > start and normalize(word["word"]):
            key = (normalize(word["word"]), start, end)
            assert key not in spans, "Duplicated positive-duration word/time span"
            spans.add(key)
    speakers = {span["speaker"] for span in raw}
    assert len(speakers) <= 8, "Nemotron exceeded its eight output speaker channels"
    if diarization_enabled:
        assert raw, "Expected raw activity for the meeting recording"
    else:
        assert payload.get("speaker_segments") is None, "Diarization flag ignored"
        assert all(
            s.get("speaker") is None and not s.get("speaker_candidates")
            for s in segments
        ), "Speaker annotations leaked with diarization disabled"

    events: dict[float, Counter] = defaultdict(Counter)
    by_speaker: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for span in raw:
        start, end = span["start"], span["end"]
        assert 0 <= start < end <= duration + 0.003, "Activity outside audio"
        assert 0 <= span["confidence"] <= 1, "Invalid activity confidence"
        events[start][span["speaker"]] += 1
        events[end][span["speaker"]] -= 1
        by_speaker[span["speaker"]].append((start, end))
    active: Counter = Counter()
    overlap = previous = 0.0
    for at, changes in sorted(events.items()):
        if sum(count > 0 for count in active.values()) > 1:
            overlap += at - previous
        active.update(changes)
        previous = at
    returns = []
    for speaker, intervals in by_speaker.items():
        intervals.sort()
        for before, after in zip(intervals, intervals[1:]):
            if after[0] - before[1] >= 30:
                returns.append(
                    {
                        "speaker": speaker,
                        "previous_start": before[0],
                        "previous_end": before[1],
                        "return_start": after[0],
                        "return_end": after[1],
                        "gap_seconds": after[0] - before[1],
                    }
                )
    return {
        "duration_seconds": duration,
        "segments": len(segments),
        "words": len(words),
        "raw_speakers": len(speakers),
        "raw_segments": len(raw),
        "raw_overlap_seconds": round(overlap, 3),
        "unknown_segments": sum(s.get("speaker") is None for s in segments),
        "candidate_segments": sum(bool(s.get("speaker_candidates")) for s in segments),
        "multiple_candidate_segments": sum(
            len(s.get("speaker_candidates") or []) > 1 for s in segments
        ),
        "late_returns_for_listening": sorted(returns, key=lambda r: -r["gap_seconds"]),
        "structural_word_duplication": False,
        "accuracy_note": "Structural checks only; no manual DER/JER or word accuracy ground truth.",
    }


def export_listening(output: Path, audio: Path, payload: dict, summary: dict) -> None:
    media = output / ("audio" + audio.suffix.lower())
    if media.resolve() != audio.resolve():
        shutil.copy2(audio, media)
    subtitles, rows = [], []
    for index, segment in enumerate(payload["segments"], 1):
        speaker = segment.get("speaker") or "Unknown"
        candidates = segment.get("speaker_candidates") or []
        if candidates:
            speaker += " (" + ", ".join(candidates) + ")"
        start, end = segment["start"], segment["end"]
        subtitles.append(
            f"{index}\n{timestamp(start)} --> {timestamp(end)}\n"
            f"[{speaker}] {segment['text']}\n"
        )
        rows.append(
            f'<tr><td><button data-start="{start}" data-end="{end}">'
            f"{start:.2f} - {end:.2f}</button></td><td>{html.escape(speaker)}</td>"
            f"<td>{html.escape(segment['text'])}</td></tr>"
        )
    (output / "transcript.srt").write_text("\n".join(subtitles), encoding="utf-8")
    raw_rows, rectangles = [], []
    raw = payload.get("speaker_segments") or []
    speakers = list(dict.fromkeys(span["speaker"] for span in raw))
    colors = [
        "#2563eb",
        "#059669",
        "#dc2626",
        "#9333ea",
        "#d97706",
        "#0891b2",
        "#db2777",
        "#475569",
    ]
    duration = payload["duration"]
    for span in raw:
        start, end, speaker = span["start"], span["end"], span["speaker"]
        lane = speakers.index(speaker)
        rectangles.append(
            f'<rect x="{start / duration * 1000:.3f}" y="{lane * 26 + 3}" '
            f'width="{max(0.25, (end-start) / duration * 1000):.3f}" height="20" '
            f'fill="{colors[lane]}" role="button" tabindex="0" '
            f'data-start="{start}" data-end="{end}"><title>'
            f"{html.escape(speaker)} {start:.2f} - {end:.2f}</title></rect>"
        )
        raw_rows.append(
            f'<tr><td><button data-start="{start}" data-end="{end}">'
            f"{start:.2f} - {end:.2f}</button></td><td>{html.escape(speaker)}</td>"
            f"<td>{span['confidence']:.3f}</td></tr>"
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nemotron validation</title><style>
body{max-width:1200px;margin:24px auto;padding:0 18px;font:16px system-ui;color:#172033}
audio{width:100%;position:sticky;top:0;background:white;padding:10px 0}
table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}
button{cursor:pointer;white-space:nowrap;padding:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
svg{width:100%;background:#f1f5f9}rect[role=button]{cursor:pointer}a{color:#2563eb}
</style><h1>Nemotron + R2T2 + ForcedAligner</h1>
<p>Click a transcript or raw activity interval to play it. Unknown means speaker attribution is uncertain;
overlap activity does not guarantee both voices were transcribed. Labels are session-local.
Brief interjections are included in the main speaker paragraph; the raw activity timeline remains unmerged.</p>
"""
    page += f'<audio id="audio" preload="none" data-src="{html.escape(media.name)}"></audio>'
    page += '<p id="audio-status" role="status" aria-live="polite">Loading audio for seeking...</p>'
    page += "<p>Zero-duration timestamps play 0.25 seconds of context on either side; displayed timestamps remain unchanged.</p>"
    page += '<p><a href="result.json">Response JSON</a> | <a href="transcript.srt">SRT</a></p>'
    page += (
        "<h2>Raw speaker activity</h2><p>Lane order: "
        + html.escape(", ".join(speakers))
        + ". Each lane retains overlapping activity.</p>"
    )
    page += f'<svg viewBox="0 0 1000 {max(26, len(speakers)*26)}" aria-label="Raw speaker timeline">'
    page += "".join(rectangles) + "</svg>"
    page += (
        "<details><summary>Validation and late-return listening points</summary><pre>"
    )
    page += (
        html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre></details>"
    )
    returns = summary.get("late_returns_for_listening", [])[:10]
    if returns:
        page += "<h2>Longest silent gaps: compare voices</h2>"
        page += "<p>Compare the two clips yourself; matching model labels do not prove matching identity.</p>"
        page += "<table><tr><th>Model label</th><th>Gap</th><th>Before</th><th>After</th></tr>"
        for point in returns:
            before_start = max(point["previous_start"], point["previous_end"] - 5)
            after_end = min(point["return_end"], point["return_start"] + 5)
            page += (
                f"<tr><td>{html.escape(point['speaker'])}</td><td>{point['gap_seconds']:.1f}s</td>"
                f'<td><button data-start="{before_start}" data-end="{point["previous_end"]}">'
                f"Before {before_start:.2f}s</button></td>"
                f'<td><button data-start="{point["return_start"]}" data-end="{after_end}">'
                f"After {point['return_start']:.2f}s</button></td></tr>"
            )
        page += "</table>"
    page += "<h2>Transcript</h2><table><tr><th>Seconds</th><th>Main speaker</th><th>Text</th></tr>"
    page += (
        "".join(rows)
        + "</table><details><summary>Raw activity intervals</summary><table>"
    )
    page += "".join(raw_rows) + "</table></details>"
    page += """<script>
const audio = document.getElementById('audio');
const status = document.getElementById('audio-status');
const intervals = [...document.querySelectorAll('[data-start]')];
let ready = false, stopAt = null, stopFrame = null;
function setReady(value) {
ready = value; audio.controls = value;
intervals.forEach(element => {
if (element instanceof HTMLButtonElement) element.disabled = !value;
element.setAttribute('aria-disabled', String(!value));
if (element.tagName.toLowerCase() === 'rect') element.setAttribute('tabindex', value ? '0' : '-1');
});
}
function playInterval(element) {
if (!ready) return;
let start = Number(element.dataset.start), end = Number(element.dataset.end);
if (end <= start) { start = Math.max(0, start - 0.25); end = Math.min(audio.duration, end + 0.25); }
audio.currentTime = start; stopAt = end;
audio.play().catch(error => { status.textContent = 'Playback failed: ' + error.message; });
}
setReady(false);
intervals.forEach(element => {
element.addEventListener('click', () => playInterval(element));
if (element.tagName.toLowerCase() === 'rect') element.addEventListener('keydown', event => {
if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); playInterval(element); }});
});
function stopAtBoundary() {
if (stopAt !== null && audio.currentTime >= stopAt) { audio.pause(); stopAt = null; }
}
function checkStop() {
stopAtBoundary();
stopFrame = audio.paused ? null : requestAnimationFrame(checkStop);
}
audio.addEventListener('timeupdate', stopAtBoundary);
audio.addEventListener('play', () => { if (stopFrame === null) stopFrame = requestAnimationFrame(checkStop); });
audio.addEventListener('loadedmetadata', () => { setReady(true); status.textContent = 'Audio ready. Select an interval to listen.'; });
audio.addEventListener('error', () => { setReady(false); status.textContent = 'Audio could not be decoded. Download the audio to listen locally.'; });
(async () => {
try {
const response = await fetch(audio.dataset.src);
if (!response.ok) throw new Error('HTTP ' + response.status);
audio.src = URL.createObjectURL(await response.blob()); audio.load();
} catch (error) { status.textContent = 'Audio loading failed: ' + error.message; }
})();
</script></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def transcribe(
    client: httpx.Client,
    base_url: str,
    audio: Path,
    output: Path,
    diarization: bool = True,
    words: bool = True,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with audio.open("rb") as source:
        response = client.post(
            base_url.rstrip("/") + "/v1/audio/transcriptions",
            data={
                "model": "arbitrary-client-model-name",
                "response_format": "verbose_json",
                "enable_speaker_diarization": str(diarization).lower(),
                "word_timestamps": str(words).lower(),
            },
            files={"file": (audio.name, source, "application/octet-stream")},
        )
    elapsed = time.perf_counter() - started
    (output / "response.txt").write_text(response.text, encoding="utf-8")
    response.raise_for_status()
    payload = response.json()
    write_json(output / "result.json", payload)
    summary = audit(payload, words, diarization)
    summary["request_seconds"] = round(elapsed, 3)
    summary["real_time_factor"] = round(elapsed / payload["duration"], 4)
    write_json(output / "summary.json", summary)
    export_listening(output, audio, payload, summary)
    return payload, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:4175")
    parser.add_argument("--audio-300", type=Path, required=True)
    parser.add_argument("--audio-full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-stream", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    short = args.output / "audio_15s.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(args.audio_300),
            "-t",
            "15",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(short),
        ],
        check=True,
    )
    api_key = os.getenv("API_KEY", "")
    headers = {"Authorization": "Bearer " + api_key} if api_key else {}
    report: dict = {"cases": {}, "failures": [], "streaming": "not_run"}
    payloads = {}
    with httpx.Client(
        headers=headers, timeout=httpx.Timeout(3600, connect=15)
    ) as client:
        for name, audio, diarization, words in [
            ("short_full", short, True, True),
            ("short_internal_alignment", short, True, False),
            ("short_no_diarization", short, False, True),
            ("short_plain", short, False, False),
            ("five_minutes", args.audio_300, True, True),
            ("full_recording", args.audio_full, True, True),
        ]:
            try:
                timings = []
                run_count = 3 if name.startswith("short_") else 1
                for run_index in range(run_count):
                    destination = args.output / name
                    if run_index < run_count - 1:
                        destination /= "warmup" if run_index == 0 else "measured_1"
                    payload, summary = transcribe(
                        client, args.base_url, audio, destination, diarization, words
                    )
                    timings.append(summary["request_seconds"])
                if run_count == 3:
                    summary["warmup_seconds"] = timings[0]
                    summary["measured_seconds"] = timings[1:]
                    summary["median_seconds"] = statistics.median(timings[1:])
                    write_json(args.output / name / "summary.json", summary)
                report["cases"][name] = summary
                payloads[name] = payload
                print(
                    json.dumps(
                        {
                            "case": name,
                            "seconds": summary["request_seconds"],
                            "segments": summary["segments"],
                            "words": summary["words"],
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                report["failures"].append({"case": name, "error": str(error)})
                print(f"Validation failed: {name}: {type(error).__name__}", flush=True)
            write_json(args.output / "report.json", report)
    if "short_full" in payloads:
        baseline = payloads["short_full"]
        report["flag_comparisons"] = {}
        for name in ("short_internal_alignment", "short_no_diarization", "short_plain"):
            if name in payloads:
                report["flag_comparisons"][name] = {
                    "normalized_character_difference": text_distance(
                        baseline["text"], payloads[name]["text"]
                    ),
                    "median_seconds": report["cases"][name]["median_seconds"],
                }
    cases = report["cases"]
    if all(
        name in cases
        for name in (
            "short_plain",
            "short_no_diarization",
            "short_internal_alignment",
            "short_full",
        )
    ):
        plain, aligned, internal, full = (
            cases[name]["median_seconds"]
            for name in (
                "short_plain",
                "short_no_diarization",
                "short_internal_alignment",
                "short_full",
            )
        )
        report["overhead_seconds"] = {
            "alignment_only": round(aligned - plain, 3),
            "diarization_and_internal_alignment": round(internal - plain, 3),
            "diarization_with_alignment_already_enabled": round(full - aligned, 3),
            "returning_public_words": round(full - internal, 3),
            "method": "Each flag combination: one excluded warmup, two measured requests, median. Signed differences include timing noise.",
        }
    if not args.skip_stream:
        try:
            with (args.output / "streaming.log").open("w", encoding="utf-8") as log:
                with redirect_stdout(log):
                    asyncio.run(
                        realtime_smoke(
                            argparse.Namespace(
                                url=args.base_url,
                                audio=[short],
                                seconds=15,
                                concurrency=4,
                                api_key=api_key,
                                output=args.output / "streaming.json",
                            )
                        )
                    )
            print(
                "Streaming regression passed; private details saved to artifacts.",
                flush=True,
            )
            report["streaming"] = "passed"
        except Exception as error:
            report["failures"].append({"case": "streaming", "error": str(error)})
            report["streaming"] = "failed"
    write_json(args.output / "report.json", report)
    links = "".join(
        f'<li><a href="{name}/index.html">{name}</a></li>' for name in report["cases"]
    )
    (args.output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Nemotron validation</title>'
        "<h1>Nemotron validation</h1><p>Model output requires listening review; no accuracy ground truth.</p>"
        '<p><a href="report.json">Report</a> | <a href="streaming.json">Streaming regression</a></p>'
        f"<ul>{links}</ul></html>",
        encoding="utf-8",
    )
    if report["failures"]:
        raise SystemExit(
            f"{len(report['failures'])} validation cases failed; see report.json"
        )


if __name__ == "__main__":
    main()
