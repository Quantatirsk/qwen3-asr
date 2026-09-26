"""Verify the full diarization, independent transcription and alignment workflow."""

import argparse
import json
import os
import time
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--samples", nargs="+", default=["zh", "en", "mixed", "long", "silence"]
    )
    args = parser.parse_args()
    report = []
    headers = (
        {"Authorization": "Bearer " + os.environ["API_KEY"]}
        if os.environ.get("API_KEY")
        else {}
    )
    for name in args.samples:
        path = args.audio_dir / (name + ".wav")
        started = time.perf_counter()
        try:
            with path.open("rb") as audio:
                response = requests.post(
                    args.url + "/v1/audio/transcriptions",
                    headers=headers,
                    files={"file": (path.name, audio, "audio/wav")},
                    data={
                        "model": "arbitrary-client-model",
                        "response_format": "verbose_json",
                        "enable_speaker_diarization": "true",
                        "word_timestamps": "true",
                    },
                    timeout=180,
                )
            response.raise_for_status()
            payload = response.json()
            if name != "silence":
                assert (
                    payload["text"] and payload["segments"] and payload["words"]
                ), payload
                assert all(s["speaker"] for s in payload["segments"]), payload
            else:
                assert not payload["text"], payload
            for word in payload.get("words") or []:
                assert (
                    0 <= word["start"] <= word["end"] <= payload["duration"] + 0.002
                ), word
            result = {
                "audio": name,
                "status": response.status_code,
                "seconds": time.perf_counter() - started,
                "payload": payload,
            }
        except Exception as error:
            result = {
                "audio": name,
                "seconds": time.perf_counter() - started,
                "error": str(error),
            }
        report.append(result)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "payload"}, ensure_ascii=False
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
