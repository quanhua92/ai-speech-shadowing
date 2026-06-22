"""Build-time helper: pre-download the Kokoro + Wav2Vec2 models into HF_HOME.

Uses raw ``transformers`` / ``kokoro`` imports (NOT the project wrappers) so it
can run during ``docker build`` right after dependencies are installed —
*before* the application source is copied. That keeps this layer cached on the
dependency lock, so editing source code doesn't re-download ~1.5 GB of models.

Also downloads all US + UK English Kokoro voices so the pregenerate step can
run without network. Idempotent.
"""

from __future__ import annotations

import sys

WAV2VEC2 = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# US + UK English voices to pre-download (each ~3 MB, one-time)
US_VOICES = ["af_heart", "af_bella", "af_nicole", "af_sky", "am_adam", "am_michael"]
UK_VOICES = ["bf_emma", "bf_isabella", "bf_alice", "bm_george", "bm_lewis"]


def main() -> int:
    from huggingface_hub import hf_hub_download
    from kokoro import KPipeline
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    print(">>> prewarming Wav2Vec2 phoneme model (~1.2 GB)...", flush=True)
    Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2)
    Wav2Vec2ForCTC.from_pretrained(WAV2VEC2)
    hf_hub_download(WAV2VEC2, "vocab.json")

    print(">>> prewarming Kokoro TTS model + all US/UK English voices...", flush=True)
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
