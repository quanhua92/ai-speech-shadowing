"""Build-time helper: pre-download the Kokoro + Wav2Vec2 models into HF_HOME.

Uses raw ``transformers`` / ``kokoro`` imports (NOT the project wrappers) so it
can run during ``docker build`` right after dependencies are installed —
*before* the application source is copied. That keeps this layer cached on
the dependency lock, so editing source code doesn't re-download ~1.5 GB of
models.

Also downloads all US + UK English Kokoro voices so the pregenerate step can
run without network. Idempotent.

Prewarms every registered phoneme backend (default ``slplab-l2`` + ``espeak``)
so either can be selected at runtime via ``PHONEME_MODEL`` without a cold
download. The espeak model's ``vocab.json`` is also fetched because the G2P
reference-side tokenizer reads it directly.
"""

from __future__ import annotations

import sys

# Hardcode model IDs so this script does NOT depend on application source.
# This lets it run during Docker build before src/ is copied (the Dockerfile
# runs this script between dependency install and source copy).
MODELS: dict[str, str] = {
    "slplab-l2": "slplab/wav2vec2-large-robust-L2-english-phoneme-recognition",
    "espeak": "facebook/wav2vec2-lv-60-espeak-cv-ft",
}

# US + UK English voices to pre-download (each ~3 MB, one-time)
US_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael"]
UK_VOICES = ["bf_emma", "bf_isabella", "bf_alice", "bm_george", "bm_lewis"]


def main() -> int:
    # Prewarm every registered phoneme backend. Each loads its own processor /
    # model weights; the espeak backend additionally needs vocab.json (the G2P
    # tokenizer reads it). AutoProcessor/AutoModelForCTC cover both families.
    from huggingface_hub import hf_hub_download
    from kokoro import KPipeline
    from transformers import AutoModelForCTC, AutoProcessor

    for key, mid in MODELS.items():
        print(f">>> prewarming phoneme model [{key}] {mid}…", flush=True)
        AutoProcessor.from_pretrained(mid)
        AutoModelForCTC.from_pretrained(mid)
        # not every backend ships a top-level vocab.json (the ARPAbet model
        # keeps its vocab inside the tokenizer); silence fetch failure.
        try:
            hf_hub_download(mid, "vocab.json")
        except Exception:
            print(f"    [warn] could not fetch vocab.json for {mid}", flush=True)

    print(">>> prewarming Kokoro TTS model + all US/UK English voices…", flush=True)
    p = KPipeline(lang_code="a")
    for v in US_VOICES:
        list(p("hello", voice=v))
        print(f"    downloaded voice: {v}", flush=True)
    p = KPipeline(lang_code="b")
    for v in UK_VOICES:
        list(p("hello", voice=v))
        print(f"    downloaded voice: {v}", flush=True)

    print(">>> prewarm complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
