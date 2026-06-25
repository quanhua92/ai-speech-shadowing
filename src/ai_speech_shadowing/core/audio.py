"""AudioSample dataclass and WAV I/O with format validation.

The :class:`AudioSample` is the canonical, typed container passed between all
downstream engine modules::

    sample: AudioSample  # (waveform: np.ndarray, sample_rate: int)

Downstream consumers (phoneme, prosody, fluency) always receive an AudioSample
that has been normalized to mono float32 at ``TARGET_SAMPLE_RATE`` (16 kHz).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from os import PathLike

TARGET_SAMPLE_RATE: int = 16000
"""The canonical sample rate for all downstream ML models (Wav2Vec2, etc.)."""

# Plausibility bounds used to reject pathological uploads before they reach the
# DSP pipeline. A WAV declaring sample_rate=1 made librosa.resample allocate
# len*16000/1 samples (~4 GB from a 122 KB upload); bounding the rate caps the
# amplification at 16x, and the duration cap bounds downstream Wav2Vec2/MFCC
# compute.
MIN_SAMPLE_RATE: int = 1000
MAX_SAMPLE_RATE: int = 192_000
MAX_DURATION_SECONDS: float = 120.0


class AudioLoadError(ValueError):
    """Raised when an audio file cannot be loaded or fails validation."""


@dataclass(frozen=True, slots=True)
class AudioSample:
    """A typed container for a chunk of PCM audio.

    Attributes:
        waveform: ``float32`` array, shape ``(n_samples,)`` for mono or
            ``(n_samples, n_channels)`` for multi-channel. Values nominally in
            ``[-1, 1]``.
        sample_rate: Sample rate in Hz.

    The dataclass is frozen so it can be safely shared across pipeline stages;
    transforms return *new* AudioSample instances rather than mutating.
    """

    waveform: np.ndarray
    sample_rate: int = TARGET_SAMPLE_RATE

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, np.ndarray):
            raise TypeError(f"waveform must be a numpy.ndarray, got {type(self.waveform).__name__}")
        if self.waveform.dtype != np.float32:
            object.__setattr__(self, "waveform", self.waveform.astype(np.float32))
        if self.waveform.ndim not in (1, 2):
            raise ValueError(
                f"waveform must be 1D (mono) or 2D (multi-channel); got {self.waveform.ndim}D"
            )
        if self.waveform.shape[0] == 0:
            raise ValueError("waveform must contain at least one sample")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive; got {self.sample_rate}")
        if not (MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE):
            raise ValueError(
                f"sample_rate {self.sample_rate} outside plausible range "
                f"[{MIN_SAMPLE_RATE}, {MAX_SAMPLE_RATE}] Hz"
            )
        max_samples = int(self.sample_rate * MAX_DURATION_SECONDS)
        if self.waveform.shape[0] > max_samples:
            raise ValueError(
                f"audio too long: {self.duration:.1f}s exceeds the "
                f"{MAX_DURATION_SECONDS:.0f}s limit"
            )
        object.__setattr__(self, "sample_rate", int(self.sample_rate))

    @property
    def num_samples(self) -> int:
        return int(self.waveform.shape[0])

    @property
    def channels(self) -> int:
        return 1 if self.waveform.ndim == 1 else int(self.waveform.shape[1])

    @property
    def is_mono(self) -> bool:
        return self.waveform.ndim == 1

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.num_samples / self.sample_rate

    @classmethod
    def from_wav(cls, path: str | PathLike[str]) -> AudioSample:
        """Load and validate an audio file (WAV/FLAC/OGG — anything soundfile reads).

        Raises:
            AudioLoadError: If the file is missing, unreadable, empty, or has an
                invalid sample rate.
        """
        p = Path(path)
        if not p.is_file():
            raise AudioLoadError(f"file not found: {p}")
        try:
            info = sf.info(str(p))
        except RuntimeError as e:
            raise AudioLoadError(f"unreadable audio header in {p}: {e}") from e
        if info.samplerate <= 0:
            raise AudioLoadError(f"invalid sample rate {info.samplerate} in {p}")
        if info.frames <= 0:
            raise AudioLoadError(f"empty audio (0 frames) in {p}")

        try:
            data, sr = sf.read(str(p), dtype="float32", always_2d=False)
        except RuntimeError as e:
            raise AudioLoadError(f"failed to decode {p}: {e}") from e
        try:
            return cls(waveform=np.ascontiguousarray(data), sample_rate=int(sr))
        except ValueError as e:
            raise AudioLoadError(f"invalid audio in {p}: {e}") from e

    MAX_UNCOMPRESSED_BYTES: ClassVar[int] = 256 * 1024 * 1024
    """Safety cap on decoded PCM size to prevent decompression bombs. 256 MB
    covers 120 s x 192 kHz x 2 ch x 4 bytes ~ 184 MB with headroom."""
    MAX_DECOMPRESSION_RATIO: ClassVar[int] = 50
    """Reject if estimated decoded size / compressed size exceeds this ratio.
    A 5 MB upload with ratio 50 caps decoded size at 250 MB."""

    @classmethod
    def from_bytes(cls, data: bytes) -> AudioSample:
        """Load and validate audio straight from a byte string (e.g. an upload).

        Raises:
            AudioLoadError: If the bytes can't be decoded or are empty.
        """
        import io

        if not data:
            raise AudioLoadError("empty audio buffer")
        # Estimate uncompressed size via header info before decoding.
        try:
            info = sf.info(io.BytesIO(data))
        except RuntimeError as e:
            raise AudioLoadError(f"unreadable audio header: {e}") from e
        estimated = info.frames * (info.channels or 1) * 4
        if estimated > cls.MAX_UNCOMPRESSED_BYTES:
            raise AudioLoadError(
                f"audio would decode to {estimated // (1024 * 1024)} MB, "
                f"exceeds {cls.MAX_UNCOMPRESSED_BYTES // (1024 * 1024)} MB limit"
            )
        if len(data) > 0 and estimated // len(data) > cls.MAX_DECOMPRESSION_RATIO:
            raise AudioLoadError(
                f"decompression ratio {estimated // len(data)}x exceeds "
                f"max {cls.MAX_DECOMPRESSION_RATIO}x"
            )
        try:
            arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        except RuntimeError as e:
            raise AudioLoadError(f"failed to decode audio buffer: {e}") from e
        if sr <= 0 or arr.size == 0:
            raise AudioLoadError("invalid or empty audio buffer")
        try:
            return cls(waveform=np.ascontiguousarray(arr), sample_rate=int(sr))
        except ValueError as e:
            raise AudioLoadError(f"invalid audio: {e}") from e

    def to_wav(
        self,
        path: str | PathLike[str],
        *,
        subtype: str = "FLOAT",
    ) -> None:
        """Write the sample to a WAV file.

        ``subtype`` defaults to ``"FLOAT"`` (32-bit IEEE float) so audio
        round-trips losslessly — the engine never wants quantization loss
        between pipeline stages. Use ``"PCM_16"`` for distribution WAVs.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), self.waveform, int(self.sample_rate), subtype=subtype)
