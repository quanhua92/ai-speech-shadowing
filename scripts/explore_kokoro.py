"""Explore Kokoro TTS: synthesize text to a 24kHz WAV using the Kokoro-82M model.

This is an exploration harness (Phase 6 TTS preview), not production code.
First run downloads the ~330MB Kokoro-82M weights from HuggingFace into the
HF cache (~/.cache/huggingface by default).

Usage:
    uv run python scripts/explore_kokoro.py --text "Hello world"
    uv run python scripts/explore_kokoro.py --text "Xin chào" --lang v --voice vf_nuoi
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/explore_kokoro.py --text "..."

Notes:
    - On Apple Silicon, set PYTORCH_ENABLE_MPS_FALLBACK=1 (some ops lack MPS kernels).
    - `--lang` is the Kokoro single-letter language code (a=US English, b=British,
      e=Spanish, f=French, j=Japanese, z=Mandarin, v=Vietnamese, etc.). The chosen
      voice must match the language family.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000
DEFAULT_OUT_DIR = Path("tmp/audio")
DEFAULT_VOICE = "af_heart"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--text", required=True, help="Text to synthesize.")
    p.add_argument(
        "--lang",
        default="a",
        help="Kokoro single-letter language code (default: a = American English).",
    )
    p.add_argument("--voice", default=DEFAULT_VOICE, help="Voice name (default: af_heart).")
    p.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed multiplier (default: 1.0)."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output WAV path (default: {DEFAULT_OUT_DIR}/<voice>_<idx>.wav).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading Kokoro pipeline (lang_code={args.lang!r})...", file=sys.stderr)
    pipeline = KPipeline(lang_code=args.lang)

    out_dir = args.out.parent if args.out else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Synthesizing: {args.text!r} with voice {args.voice!r}", file=sys.stderr)
    generator = pipeline(args.text, voice=args.voice, speed=args.speed)

    n_chunks = 0
    for i, (graphemes, phonemes, audio) in enumerate(generator):
        out_path = args.out if args.out is not None else out_dir / f"{args.voice}_{i}.wav"
        sf.write(out_path, audio, SAMPLE_RATE)
        n_chunks += 1
        print(f"[chunk {i}] {graphemes!r} -> phonemes: {phonemes}", file=sys.stderr)
        dur = len(audio) / SAMPLE_RATE
        print(f"  wrote {out_path} ({dur:.2f}s, {SAMPLE_RATE}Hz)", file=sys.stderr)

    print(f"Done. {n_chunks} chunk(s) written to {out_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
