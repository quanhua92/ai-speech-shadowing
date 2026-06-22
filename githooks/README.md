# githooks/

Project-local Git hooks. Activate them once after cloning:

```bash
git config core.hooksPath githooks
```

Git will now run the hooks in this directory instead of the default `.git/hooks`.

## Hooks

| Hook          | Runs                                   | When             |
| ------------- | -------------------------------------- | ---------------- |
| `pre-commit`  | `ruff check` + `ruff format --check`   | Before a commit  |

## Bypassing

Skip the hooks for a single command:

```bash
git commit --no-verify
```

## Notes

- The hooks call `uv run ...`, so they require `uv` and a synced environment
  (`uv sync`). They will fail fast if `uv` is not on PATH.
- The pre-commit hook does **not** auto-fix. If `ruff format --check` fails,
  run `uv run ruff format .` and re-stage.
