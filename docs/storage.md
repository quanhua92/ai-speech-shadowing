# History Storage & Per-User Identity

> How evaluation reports are persisted, scoped per-browser, and aged out.
> The canonical reference for the identity model, storage layout, and retention.

## Overview

Every evaluation is saved as a JSON report (plus the user's recorded audio as a
`.wav`). Reports are **scoped per-browser** via a hashed identity cookie, so
each user only sees their own history. A background task deletes reports older
than a configurable retention window.

```
data/history/
  a1b2c3…ef/              ← one folder per browser (sha256 of the cookie token)
    eval_221572d6.json
    eval_221572d6.wav
  _cli/                   ← CLI evaluations (no cookie)
    eval_…json
  eval_13280587.json      ← legacy flat files (pre-feature; cleanup ages out)
```

## Per-user identity (`api/identity.py`)

Each browser is assigned a stable identity on first visit via the `user_id`
cookie. The cookie carries a random **uuid4 token**; the on-disk user identifier
is its **SHA-256 digest** — the raw token is never written to disk.

| | |
| --- | --- |
| **Cookie name** | `user_id` |
| **Cookie value** | `uuid4().hex` (122-bit random token) |
| **On-disk identifier** | `sha256(token).hexdigest()` (64 hex chars) |
| **`httpOnly`** | `True` — JavaScript cannot read the cookie (XSS defense) |
| **`secure`** | `True` only when `ENV=production` — HTTPS-only in prod, plain HTTP for local dev |
| **`sameSite`** | `lax` — CSRF defense (the Origin-check middleware is the primary guard) |
| **`max_age`** | 1 year (the data itself is aged out by the retention cleanup) |

### Security model

The hash-on-disk design means a **storage compromise alone cannot forge a valid
cookie**: an attacker who reads `data/history/` gets SHA-256 digests, not the
tokens needed to impersonate a user (finding a token whose hash matches a digest
is computationally infeasible).

Combined with `httpOnly` (XSS can't steal the cookie) and `secure`-in-prod
(network can't sniff it), this covers the main hijack vectors.

> **What it is not:** this is *not authentication*. Anyone holding a valid
> cookie can read that user's history — there is no login, no expiry-on-idle,
> no per-eval access control. It is a privacy boundary between browsers sharing
> one server, not a security boundary against a determined attacker.

### Middleware flow (`api/app.py`)

```
read cookie "user_id"
  ├─ present + valid 64-hex → request.state.user_id = value
  ├─ present (raw uuid)     → request.state.user_id = sha256(value)
  └─ absent                 → mint token, request.state.user_id = sha256(token),
                              Set-Cookie on the response
```

Route handlers read `request.state.user_id` for all storage operations. The
browser auto-sends the cookie on every same-origin request (including
`<audio>` elements), so the frontend needs no JavaScript changes.

## Storage API (`core/history.py`)

Every function takes a `user_id: str | None` parameter:

| Function | `user_id` set (API path) | `user_id=None` (CLI / `report`) |
| --- | --- | --- |
| `save_report(report, *, history_dir, user_id)` | writes `history_dir/user_id/eval_*.json` | writes `history_dir/_cli/eval_*.json` |
| `list_reports(history_dir, user_id)` | globs `history_dir/user_id/eval_*.json` | `rglob` across all subdirectories + top-level |
| `load_report(id, history_dir, user_id)` | checks only `history_dir/user_id/` | searches all subdirectories + top-level |
| `delete_report(id, history_dir, user_id)` | same scoping | same |
| `report_path(id, history_dir, user_id, *, suffix)` | `history_dir/user_id/{id}{suffix}` | top-level fallback |
| `compute_stats(history_dir, user_id, *, period_days)` | scoped to user dir | all users |

**Isolation guarantee:** when `user_id` is set, a user can only read, list, or
delete their own reports. User A cannot access user B's history (load/delete
return `None`/`False`/404).

**Path safety:** both `report_id` and `user_id` are validated against strict
regexes (`[A-Za-z0-9_-]+`) and `resolve()`-checked for containment under
`history_dir` — defense in depth against directory traversal.

### CLI

CLI evaluations (no browser cookie) save under the fixed `_cli` bucket. The
`report` command (list / view) passes `user_id=None`, scanning every user's
directory — the all-users admin view.

## Daily cleanup (`cleanup_old_reports`)

A background `asyncio` task (spawned in the app lifespan) deletes reports whose
`created_at` is older than the retention window. It runs once shortly after
startup, then repeats on an interval.

```python
def cleanup_old_reports(history_dir, retention_days=7) -> int:
    # rglob("eval_*.json") — per-user subdirs AND legacy flat files
    # delete .json + .wav where created_at < now - retention_days
    # remove empty user directories; return count
```

### Environment variables

| Env var | Default | Effect |
| --- | --- | --- |
| `HISTORY_RETENTION_DAYS` | `7` | Reports older than this are deleted (JSON + WAV). `0` = keep forever |
| `HISTORY_CLEANUP_INTERVAL_HOURS` | `24` | How often the background sweep runs |
| `ENV` | _(unset)_ | `production` → cookies get `secure=True` (HTTPS-only) |
| `PHONEME_MODEL` | `slplab-l2` | Phoneme backend (see [phoneme-extraction.md](phoneme-extraction.md)) |

> **Legacy flat files:** reports saved before this feature shipped (at the top
> level of `data/history/`, not in a user subdirectory) are invisible to scoped
> per-user queries, but the cleanup sweep (`rglob`) still ages them out.

## Test coverage

- `tests/test_identity.py` — token generation, sha256 hashing, `is_valid_user_id`,
  `is_production` (env-based).
- `tests/test_history.py::TestUserScoping` — per-user isolation (A can't
  see/load/delete B's reports), `_cli` default bucket, `None` all-users scan,
  path validation with user segment.
- `tests/test_history.py::TestCleanup` — retention sweep (deletes old, keeps
  recent, `0`=noop, removes WAV alongside JSON, sweeps empty dirs, cleans legacy
  flat files).
- `tests/test_api.py::TestUserIdentityCookie` — first-visit `Set-Cookie`,
  cookie reuse, two cookies see disjoint history, cross-user load → 404.
