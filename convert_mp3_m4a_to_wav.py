#!/usr/bin/env python3
"""
Convert all .mp3/.m4a files from:
  data/audio/mp3/
to .wav files under:
  data/audio/

Example mapping:
  data/audio/mp3/spk_01/sample1.mp3
->data/audio/spk_01/sample1.wav
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import whisper


PREFERRED_INPUT_ROOTS = [Path("data/audio/mp3"), Path("data/mp3")]
OUTPUT_ROOT = Path("data/audio")
SUPPORTED_EXTENSIONS = {".mp3", ".m4a"}
TARGET_SAMPLE_RATE = 16000


def save_wav_16k_mono(audio_float32: np.ndarray, output_path: Path) -> None:
    clipped = np.clip(audio_float32, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(pcm16.tobytes())


def collect_input_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def build_output_path(input_file: Path, input_root: Path) -> Path:
    relative = input_file.relative_to(input_root)
    return (OUTPUT_ROOT / relative).with_suffix(".wav")


def convert_one_file(input_file: Path, output_file: Path) -> None:
    audio = whisper.load_audio(str(input_file))  # 16kHz mono float32
    save_wav_16k_mono(audio, output_file)


def detect_input_root() -> Path | None:
    for root in PREFERRED_INPUT_ROOTS:
        if root.exists() and root.is_dir():
            return root
    return None


def main() -> int:
    detected_root = detect_input_root()
    if detected_root is None:
        checked = ", ".join(str(p) for p in PREFERRED_INPUT_ROOTS)
        print(f"[ERROR] Input folder not found. Checked: {checked}")
        return 1

    input_root = detected_root.resolve()
    output_root = OUTPUT_ROOT.resolve()

    files = collect_input_files(input_root)
    if not files:
        print(f"[INFO] No .mp3/.m4a files found in: {input_root}")
        return 0

    print(f"[INFO] Found {len(files)} files to convert")
    print(f"[INFO] Output root: {output_root}")

    ok = 0
    fail = 0

    for idx, input_file in enumerate(files, start=1):
        output_file = build_output_path(input_file, input_root)
        try:
            convert_one_file(input_file, output_file)
            ok += 1
            print(f"[{idx}/{len(files)}] OK   {input_file} -> {output_file}")
        except Exception as exc:
            fail += 1
            print(f"[{idx}/{len(files)}] FAIL {input_file}")
            print(f"                Reason: {exc}")

    print("\n[SUMMARY]")
    print("\n\n")
    print(f"Converted: {ok}")
    print(f"Failed: {fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
