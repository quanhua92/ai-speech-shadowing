# CLI Interface

> **Phase 7 deliverable.** The developer-facing command-line workflow. The CLI
> grew command-by-command through Phases 1–6; Phase 7 adds the global output
> flags, `record`, `batch`, `report`, progress bars, and report persistence.

## Install & overview

```bash
uv sync
ai-speech-shadowing --help
```

Commands:

| Command | Purpose |
| --- | --- |
| `version` | Print the installed version. |
| `preprocess` | Mono → 16 kHz → trim → normalize a file. |
| `phoneme` | Extract the IPA phoneme sequence (Wav2Vec2-CTC). |
| `prosody` | F0 pitch statistics (mean, range, voiced ratio). |
| `fluency` | MFCC + DTW comparison of two files. |
| `evaluate` | **Full pipeline** → unified `FeedbackReport` (saved to history). |
| `generate-reference` | Kokoro TTS reference (single `--text` or `--list` batch). |
| `backfill-phonemes` | Populate cached G2P phonemes for references that predate capture-at-gen. |
| `record` | Record user audio from the microphone. |
| `batch` | Evaluate every recording in a directory against one reference. |
| `report` | List saved reports, or view one in detail. |

## Global flags

`--verbose / -v` and `--quiet / -q` are app-level options on every command
(parsed before the subcommand):

```bash
ai-speech-shadowing --verbose evaluate ref.wav user.wav   # debug logging
ai-speech-shadowing --quiet   evaluate ref.wav user.wav   # warnings only
```

For machine-readable output, the data-producing commands take `--format json`
(e.g. `evaluate --format json`). `report <id> --format json` prints the stored
report as JSON.

## The practice loop

```bash
# 1. create a native reference (or generate one)
ai-speech-shadowing generate-reference --text "The quick brown fox"

# 2. record your attempt
ai-speech-shadowing record attempt.wav --duration 4

# 3. evaluate → unified report (auto-saved to data/history/)
ai-speech-shadowing evaluate data/references/the-quick-brown-fox/audio/kokoro-en-us/ref.wav \
    attempt.wav

# 4. review past attempts
ai-speech-shadowing report
ai-speech-shadowing report eval_ce49e658
```

Sample `report <id>` output:
```
Report eval_ce49e658  (2026-06-22T12:56:53+00:00)
Composite: 100/100 🟢 good
Pronunciation 100 | Intonation 100 | Fluency 100
Feedback:
  • Great job — your delivery closely matches the reference.
```

## Commands in detail

### `evaluate`

```bash
ai-speech-shadowing evaluate <reference.wav> <user.wav> \
    [--format terminal|json|markdown] \
    [--weights 0.4,0.3,0.3] \
    [--no-preprocess] [--no-save] [--history-dir DIR]
```

Runs phoneme + prosody + fluency and prints the unified report. By default the
report is persisted to the history directory as `eval_<id>.json` (pass
`--no-save` to skip). The first call downloads/loads the ~1.2 GB Wav2Vec2 model.

### `generate-reference`

See [`tts-reference.md`](tts-reference.md). `--text "…"` for a single reference,
`--list sentences.txt` for a batch; `--voice`, `--lang`, `--output-dir`,
`--force`.

### `backfill-phonemes`

```bash
ai-speech-shadowing backfill-phonemes [--output-dir DIR] [--force] [--dry-run]
```

One-shot migration command for references created before the
capture-at-synthesis change. For every reference under `--output-dir` with a
stored `text` field, runs misaki G2P on the text and persists the normalized
espeak tokens to `metadata.json["phonemes"]` — the same field
`generate-reference` populates at synthesis time. References that already have
the block are skipped unless `--force` is given.

Idempotent and safe to re-run. `--dry-run` previews without writing. Use once
after upgrading to populate existing references without re-synthesizing their
audio:

```bash
ai-speech-shadowing backfill-phonemes            # populate missing
ai-speech-shadowing backfill-phonemes --force    # recompute everything
ai-speech-shadowing backfill-phonemes --dry-run  # preview only
```

The first call downloads the Wav2Vec2 `vocab.json` (small) into the HF cache;
subsequent calls reuse it.

### `record`

```bash
ai-speech-shadowing record <out.wav> [--duration 5] [--sample-rate 16000]
```

Records mono `float32` from the default input device via `sounddevice` and
writes a WAV. Ctrl-C aborts. (Requires a working microphone / PortAudio.)

### `batch`

```bash
ai-speech-shadowing batch <reference.wav> <recordings_dir> [--history-dir DIR]
```

Evaluates every audio file (`*.wav`, `*.flac`, `*.ogg`, `*.aiff`) in the
directory against the single reference, shows a `rich` progress bar, saves a
report per recording, and prints a summary table sorted by score:

```
Evaluated 3 recording(s) (reports saved to data/history):
   88/100  good       attempt-3.wav
   74/100  fair       attempt-1.wav
   61/100  fair       attempt-2.wav
```

The phoneme model is loaded **once** and reused across all recordings.

### `report`

```bash
ai-speech-shadowing report [--history-dir DIR]            # list all
ai-speech-shadowing report <id> [--history-dir DIR]       # view one
ai-speech-shadowing report <id> --format json             # raw JSON
```

Lists saved reports (`id  score  grade  timestamp`) or prints one in summary or
JSON form. History lives under `data/history/` by default.

## Report persistence

CLI evaluations are saved under the fixed `_cli` user bucket
(`data/history/_cli/eval_*.json`). The `report` command scans **all** user
directories (the all-users view). Reports are JSON via
`ai_speech_shadowing.core.history`; each function takes a `user_id` parameter
(`None` = all users). See [`storage.md`](storage.md) for the full layout, the
per-browser identity model, and the retention/cleanup configuration.

| Function | Purpose |
| --- | --- |
| `save_report(report, *, history_dir=…, user_id=…)` | Write `eval_<id>.json`, return the path |
| `list_reports(history_dir=…, user_id=…)` | `list[HistoryEntry]` (id, timestamp, composite score/grade) |
| `load_report(id, history_dir=…, user_id=…)` | Report dict, or `None` |
| `delete_report(id, history_dir=…, user_id=…)` | `True` if removed |
| `cleanup_old_reports(history_dir=…, retention_days=…)` | Delete reports older than N days; return count |
| `format_summary(data)` | Compact terminal view of a saved dict |

## Progress bars

Long operations report progress with [`rich`](https://rich.readthedocs.io/):

- **`batch`** — a live progress bar (bar, %, ETA) as each recording is scored.
- **model loading** — `evaluate`/`batch`/`phoneme` print a `Loading model…`
  notice on `stderr` so the pause isn't silent. (The transformer's internal
  weight download isn't hookable for a finer-grained bar.)

## Design decisions

- **`evaluate` saves by default.** The Phase 7 workflow is *practice → review
  history*, so persisting is the common case; `--no-save` opts out for
  scripting/CI.
- **`evaluate()` the function stays pure.** Only the CLI command writes to disk,
  so the engine remains unit-testable without touching the filesystem.
- **Global flags control logging only.** `--verbose`/`--quiet` set the Python
  logging level (debug / warning); per-command `--format json` handles
  machine output rather than a competing global `--json`.
- **`batch` loads the model once.** Reusing one `PhonemeExtractor` across the
  directory turns an O(n) model-load cost into O(1).
- **History is plain JSON files.** No DB dependency; trivially inspectable,
  greppable, and served verbatim by the future API.

## Test coverage

- `tests/test_history.py` (fast): save writes `eval_*.json` with id/timestamp;
  load round-trip; missing → `None`; list (empty, sorted, skips malformed);
  delete (hit/miss); `format_summary` content.
- `tests/test_cli.py` (fast): global `--verbose`/`--quiet` accepted; `record`
  writes a WAV with `sounddevice` mocked; `report` list / view-by-id /
  view-json / empty / missing-id; `backfill-phonemes` (G2P mocked) covers
  populate, skip-existing, `--force`, `--dry-run`, idempotency, empty dir,
  missing-text, and `--help`.
- `tests/test_cli.py::test_batch_evaluates_directory` (opt-in slow): evaluates
  a directory of two recordings against a Kokoro reference, asserts the summary
  line and that two reports were saved.
- `tests/test_cli.py::test_backfill_uses_real_misaki_g2p` (opt-in slow):
  regression test that the backfill command actually runs misaki G2P (not just
  raw-text normalization) and produces real espeak diphthongs/affricates.
