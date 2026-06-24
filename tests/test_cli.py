"""Tests for the Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ai_speech_shadowing.cli import app
from ai_speech_shadowing.core.history import save_report

runner = CliRunner()


def test_preprocess_command_writes_16k_mono(tmp_path: Path, stereo_48000_wav: Path) -> None:
    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["preprocess", str(stereo_48000_wav), "-o", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    assert "16000 Hz" in result.stdout
    assert "1ch" in result.stdout


def test_preprocess_default_output_suffix(tmp_path: Path, mono_44100_wav: Path) -> None:
    result = runner.invoke(app, ["preprocess", str(mono_44100_wav)])
    assert result.exit_code == 0, result.stdout
    expected = mono_44100_wav.with_suffix(".preprocessed.wav")
    assert expected.is_file()


def test_preprocess_disable_normalize(tmp_path: Path, quiet_wav: Path) -> None:
    out = tmp_path / "out.wav"
    result = runner.invoke(
        app,
        ["preprocess", str(quiet_wav), "-o", str(out), "--normalize", "none"],
    )
    assert result.exit_code == 0, result.stdout


def test_preprocess_missing_file_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["preprocess", str(tmp_path / "nope.wav")])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# Global flags
# --------------------------------------------------------------------------- #
def test_global_verbose_flag_accepted() -> None:
    result = runner.invoke(app, ["--verbose", "version"])
    assert result.exit_code == 0


def test_global_quiet_flag_accepted() -> None:
    result = runner.invoke(app, ["--quiet", "version"])
    assert result.exit_code == 0


# --------------------------------------------------------------------------- #
# record (mocked sounddevice — no real microphone)
# --------------------------------------------------------------------------- #
def test_record_writes_wav(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sd = pytest.importorskip("sounddevice")

    def fake_rec(n: int, **kwargs: object) -> np.ndarray:
        return np.zeros((n, 1), dtype="float32")

    monkeypatch.setattr(sd, "rec", fake_rec)
    monkeypatch.setattr(sd, "wait", lambda: None)

    out = tmp_path / "rec.wav"
    result = runner.invoke(app, ["record", str(out), "--duration", "0.1", "--sample-rate", "8000"])
    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    assert "wrote" in result.stdout


# --------------------------------------------------------------------------- #
# report (list + view)
# --------------------------------------------------------------------------- #
def _seed_report(history_dir: Path) -> str:
    from ai_speech_shadowing.core.feedback import build_report
    from ai_speech_shadowing.core.fluency import DtwResult, FluencyDiff, PauseInfo
    from ai_speech_shadowing.core.phoneme import diff_phonemes
    from ai_speech_shadowing.core.prosody import PitchStats, ProsodyDiff

    pitch = PitchStats(
        f0_contour=np.zeros(1, dtype=np.float64),
        times=np.zeros(1, dtype=np.float64),
        mean_hz=200.0,
        median_hz=200.0,
        min_hz=100.0,
        max_hz=300.0,
        range_hz=200.0,
        std_hz=20.0,
        voiced_ratio=1.0,
        pitch_floor=75.0,
        pitch_ceiling=500.0,
    )
    prosody = ProsodyDiff(
        reference=pitch,
        hypothesis=pitch,
        pitch_range_ratio=1.0,
        monotone=False,
        monotone_threshold=0.5,
        score=1.0,
    )
    fluency = FluencyDiff(
        dtw=DtwResult(0.0, 10, 0.0),
        score=1.0,
        reference_pauses=PauseInfo(0, 0.0, ()),
        hypothesis_pauses=PauseInfo(0, 0.0, ()),
        syllable_rate_reference=2.0,
        syllable_rate_hypothesis=2.0,
        syllable_rate_ratio=1.0,
    )
    path = save_report(
        build_report(diff_phonemes(["a", "b"], ["a", "b"]), prosody, fluency),
        history_dir=history_dir,
    )
    return path.stem


def test_report_lists_saved_reports(tmp_path: Path) -> None:
    history = tmp_path / "history"
    rid = _seed_report(history)
    result = runner.invoke(app, ["report", "--history-dir", str(history)])
    assert result.exit_code == 0, result.stdout
    assert rid in result.stdout
    assert "100/100" in result.stdout


def test_report_view_by_id(tmp_path: Path) -> None:
    history = tmp_path / "history"
    rid = _seed_report(history)
    result = runner.invoke(app, ["report", rid, "--history-dir", str(history)])
    assert result.exit_code == 0, result.stdout
    assert "Composite: 100/100" in result.stdout


def test_report_view_json(tmp_path: Path) -> None:
    history = tmp_path / "history"
    rid = _seed_report(history)
    import json

    result = runner.invoke(app, ["report", rid, "--history-dir", str(history), "--format", "json"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["id"] == rid


def test_report_empty(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "--history-dir", str(tmp_path / "history")])
    assert result.exit_code == 0
    assert "no saved reports" in result.stdout


def test_report_missing_id_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["report", "eval_nope", "--history-dir", str(tmp_path / "history")])
    assert result.exit_code != 0


# --------------------------------------------------------------------------- #
# batch (opt-in slow: loads the phoneme model)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_batch_evaluates_directory(
    tmp_path: Path, kokoro_ref_wav: Path, mono_44100_wav: Path
) -> None:
    # two "recordings" (reuse fixtures) in a dir, evaluate against the Kokoro ref
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "r1.wav").write_bytes(mono_44100_wav.read_bytes())
    (recordings / "r2.wav").write_bytes(mono_44100_wav.read_bytes())
    history = tmp_path / "history"

    result = runner.invoke(
        app,
        [
            "batch",
            str(kokoro_ref_wav),
            str(recordings),
            "--history-dir",
            str(history),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Evaluated 2 recording(s)" in result.stdout
    assert (history / "eval_").parent == history
    # two reports saved
    from ai_speech_shadowing.core.history import list_reports

    assert len(list_reports(history)) == 2


# --------------------------------------------------------------------------- #
# backfill-phonemes (fast: G2P is mocked so no misaki / HF vocab load)
# --------------------------------------------------------------------------- #
def _seed_ref_without_phonemes(base: Path, slug: str, text: str) -> None:
    """Plant a minimal reference folder with text but no phonemes block."""
    d = base / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"text": text, "default_speaker": "af_heart"}))


def _seed_ref_with_phonemes(base: Path, slug: str, text: str, tokens: list[str]) -> None:
    d = base / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(
        json.dumps(
            {
                "text": text,
                "default_speaker": "af_heart",
                "phonemes": {
                    "tokens": tokens,
                    "source": "kokoro-g2p",
                    "notation": "espeak-wav2vec2",
                },
            }
        )
    )


class TestBackfillPhonemes:
    """Verify the backfill CLI command without invoking misaki / HF vocab.

    The G2P callable is monkeypatched to a deterministic stub so the tests stay
    fast (no model download, no espeak vocab load). The integration with the
    real misaki is exercised end-to-end by the slow test below.
    """

    def test_backfill_populates_missing_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_speech_shadowing.core import g2p as g2p_mod

        monkeypatch.setattr(
            g2p_mod,
            "text_to_espeak_tokens",
            lambda s: ("h", "ə", "l", "oʊ") if "hello" in s.lower() else ("x",),
        )
        _seed_ref_without_phonemes(tmp_path, "hello-world", "Hello world")

        result = runner.invoke(app, ["backfill-phonemes", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.stdout
        assert "WROTE" in result.stdout
        assert "1 updated" in result.stdout

        meta = json.loads((tmp_path / "hello-world" / "metadata.json").read_text())
        assert meta["phonemes"]["tokens"] == ["h", "ə", "l", "oʊ"]
        assert meta["phonemes"]["source"] == "kokoro-g2p"

    def test_backfill_skips_existing_without_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_speech_shadowing.core import g2p as g2p_mod

        monkeypatch.setattr(g2p_mod, "text_to_espeak_tokens", lambda s: ("NEW",))
        _seed_ref_with_phonemes(tmp_path, "hello", "Hello", ["OLD"])

        result = runner.invoke(app, ["backfill-phonemes", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.stdout
        assert "HAVE" in result.stdout
        assert "1 skipped" in result.stdout

        # Unchanged — the existing block was not overwritten.
        meta = json.loads((tmp_path / "hello" / "metadata.json").read_text())
        assert meta["phonemes"]["tokens"] == ["OLD"]

    def test_backfill_force_overwrites(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_speech_shadowing.core import g2p as g2p_mod

        monkeypatch.setattr(g2p_mod, "text_to_espeak_tokens", lambda s: ("NEW", "toks"))
        _seed_ref_with_phonemes(tmp_path, "hello", "Hello", ["OLD"])

        result = runner.invoke(app, ["backfill-phonemes", "--output-dir", str(tmp_path), "--force"])
        assert result.exit_code == 0, result.stdout
        assert "WROTE" in result.stdout
        assert "1 updated" in result.stdout

        meta = json.loads((tmp_path / "hello" / "metadata.json").read_text())
        assert meta["phonemes"]["tokens"] == ["NEW", "toks"]

    def test_backfill_dry_run_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_speech_shadowing.core import g2p as g2p_mod

        monkeypatch.setattr(g2p_mod, "text_to_espeak_tokens", lambda s: ("h", "ə"))
        _seed_ref_without_phonemes(tmp_path, "hi", "Hi")

        result = runner.invoke(
            app, ["backfill-phonemes", "--output-dir", str(tmp_path), "--dry-run"]
        )
        assert result.exit_code == 0, result.stdout
        assert "WOULD" in result.stdout
        assert "dry-run" in result.stdout

        # No write happened.
        meta = json.loads((tmp_path / "hi" / "metadata.json").read_text())
        assert "phonemes" not in meta

    def test_backfill_idempotent_second_run_is_noop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_speech_shadowing.core import g2p as g2p_mod

        monkeypatch.setattr(g2p_mod, "text_to_espeak_tokens", lambda s: ("h", "ə", "l", "oʊ"))
        _seed_ref_without_phonemes(tmp_path, "hello-world", "Hello world")

        # First run populates.
        r1 = runner.invoke(app, ["backfill-phonemes", "--output-dir", str(tmp_path)])
        assert r1.exit_code == 0
        assert "1 updated" in r1.stdout

        # Second run skips (block now present).
        r2 = runner.invoke(app, ["backfill-phonemes", "--output-dir", str(tmp_path)])
        assert r2.exit_code == 0
        assert "1 skipped" in r2.stdout
        assert "0 updated" in r2.stdout

    def test_backfill_empty_dir_reports_no_references(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["backfill-phonemes", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.stdout
        assert "no references" in result.stdout

    def test_backfill_skips_refs_without_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ai_speech_shadowing.core import g2p as g2p_mod

        monkeypatch.setattr(g2p_mod, "text_to_espeak_tokens", lambda s: ("x",))
        d = tmp_path / "no-text"
        d.mkdir(parents=True)
        # metadata.json exists but has no text field — must SKIP, not crash.
        (d / "metadata.json").write_text(json.dumps({"default_speaker": "af_heart"}))

        result = runner.invoke(app, ["backfill-phonemes", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, result.stdout
        assert "SKIP" in result.stdout
        assert "1 failed" in result.stdout  # counted as a failure to backfill


@pytest.mark.slow
def test_backfill_uses_real_misaki_g2p(tmp_path: Path) -> None:
    """Integration test: the backfill command runs real misaki G2P on the
    reference text and produces sensible espeak tokens.

    Catches regressions where the CLI accidentally passes the raw text into the
    phoneme-normalization path (which would yield character-level junk tokens
    instead of real phonemes — a real bug that shipped undetected by the mocked
    fast tests until this test was added).
    """
    _seed_ref_without_phonemes(tmp_path, "hello-world", "Hello world")

    result = runner.invoke(app, ["backfill-phonemes", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "1 updated" in result.stdout

    meta = json.loads((tmp_path / "hello-world" / "metadata.json").read_text())
    tokens = meta["phonemes"]["tokens"]
    # Sanity: real G2P for "Hello world" starts with /h/ and contains the GOAT
    # diphthong /oʊ/ (not character-level "o" + "ʊ" junk). Also asserts the
    # tokens came from the espeak vocab, not raw text characters.
    assert tokens[0] == "h"
    assert "oʊ" in tokens
    # The character "ʊ" alone is NOT an espeak token in this vocabulary — its
    # presence would indicate the bug where raw text was tokenized char-by-char.
    assert "ʊ" not in tokens
    assert meta["phonemes"]["source"] == "kokoro-g2p"
    assert meta["phonemes"]["notation"] == "espeak-wav2vec2"
