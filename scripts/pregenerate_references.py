"""Pre-generate all 30 default references in every US + UK English voice.

Run locally or during ``docker build`` (after prewarm + project install):
    python scripts/pregenerate_references.py

Produces ~330 reference WAVs (~55 MB). Existing references (e.g. af_heart) are
skipped (cached). Each new voice downloads its .pt on first use (~3 MB).
"""

from __future__ import annotations

import sys
from pathlib import Path

US_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael"]
UK_VOICES = ["bf_emma", "bf_isabella", "bf_alice", "bm_george", "bm_lewis"]
LANG_FOR_VOICE = {**{v: "a" for v in US_VOICES}, **{v: "b" for v in UK_VOICES}}


def main() -> int:
    from ai_speech_shadowing.tts.generator import (
        ReferenceConfig,
        ReferenceManager,
        parse_sentence_list,
    )

    default_txt = Path("data/default.txt")
    if not default_txt.is_file():
        print("data/default.txt not found — run from project root.", file=sys.stderr)
        return 1

    sentences = parse_sentence_list(default_txt)
    mgr = ReferenceManager(ReferenceConfig(base_dir=Path("data/references")))
    all_voices = US_VOICES + UK_VOICES

    total, skipped = 0, 0
    for voice in all_voices:
        lang = LANG_FOR_VOICE[voice]
        for text in sentences:
            slug = mgr.voice_profile(lang=lang, voice=voice)
            existing = mgr.audio_file(_slug_from_text(text), slug)
            if existing.is_file():
                skipped += 1
                continue
            mgr.generate(text, voice=voice, lang=lang)
            total += 1
        print(f"  done: {voice} ({total + skipped} total)", flush=True)

    print(
        f">>> pregenerated {total} new, skipped {skipped} existing "
        f"({total + skipped} total references)",
        flush=True,
    )
    return 0


def _slug_from_text(text: str) -> str:
    from ai_speech_shadowing.tts.generator import slugify

    return slugify(text)


if __name__ == "__main__":
    sys.exit(main())
