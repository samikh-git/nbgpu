# nbgpu CLI reference for agents

Every command accepts `--json`. In `--json` mode stdout is exactly one JSON
document; progress and warnings go to stderr. Errors are one line on stderr
(`nbgpu: error: ...`) with a non-zero exit code and no JSON.

## Commands

### `nbgpu run NOTEBOOK [--backend NAME] [--max-lines N] [--project-root DIR] [--json]`

Runs `python NOTEBOOK` on a runtime and exits with the notebook's exit code.
Uses the notebook's running session if it has exactly one (and `--backend`,
if given, matches it, and `--project-root` is not given); otherwise creates a
one-off runtime, pushes the project, runs, and terminates it.

```json
{"exit_code": 0, "output": "stdout then stderr", "truncated": false, "full_output_path": null}
```

`truncated: true` means `output` holds the first and last halves of
`--max-lines` (default 200) around a marker line, and `full_output_path` is a
local log file with everything.

### `nbgpu start NOTEBOOK [--detach] [--gpu G] [--cpu N] [--memory M] [--idle-timeout D] [--max-runtime D] [--backend NAME] [--project-root DIR] [--json]`

Creates the runtime, uploads the project, starts marimo, waits until it is
ready. With `--detach` it then returns; without, it keeps syncing every
second until interrupted (the runtime keeps running after that).

```json
{
  "session_id": "b390261f",
  "notebook": "/abs/path/nb.py",
  "backend": "modal",
  "runtime_id": "sb-...",
  "url": "https://...?access_token=...",
  "local_url": null,
  "rate": {"usd_per_hour": 1.1, "complete": true}
}
```

- `url` contains a secret token.
- `local_url` is the fast localhost proxy; null with `--detach`.
- `rate` is null when nothing is known about the price; `complete: false`
  means part of the shape (e.g. CPU on an unknown provider) is not priced.
- Flag formats: `--gpu A10G` or `H100:2`; `--memory 32GiB` (a bare number is
  MiB); durations `20m`, `1h30m`, `90s` (a bare number is seconds).

### `nbgpu sync NOTEBOOK [--json]` and `nbgpu stop NOTEBOOK [--force] [--json]`

`sync` is one two-way pass. `stop` does a final pull, terminates the runtime
and removes the record. Both print a sync report:

```json
{"pushed": [], "pulled": ["result.json"], "merged": [], "conflicts": [], "skipped": [], "warnings": []}
```

- `merged`: both sides changed and merged cleanly (written to both).
- `conflicts`: open conflicts; a `.remote` copy sits next to each.
- `skipped`: left alone this round (changed mid-sync, too large, unreadable,
  unsafe path). Sync again.
- `warnings` on `stop` can list packages installed during the session that
  the header does not name, and edits lost because the runtime was already
  gone.

`stop` exits non-zero with open conflicts unless `--force`.

### `nbgpu status [--backend NAME] [--json]`

```json
{
  "sessions": [{
    "session_id": "b390261f", "notebook": "/abs/path/nb.py", "backend": "modal",
    "gpu": "A10G", "running": true, "state": "running", "uptime_s": 312.4,
    "cost_usd": 0.1, "conflicts": [], "last_sync_at": 1794711155.99,
    "warning": null, "unsynced": []
  }],
  "orphans": [{"backend": "modal", "runtime_id": "sb-...", "uptime_s": 900.0, "session_tag": "..."}],
  "other_owners": []
}
```

`state` is one of:

| state | meaning | what to do |
|-------|---------|------------|
| `running` | marimo is up | nothing |
| `starting` | a start is in progress, or was interrupted | wait; if no start is running, `stop` |
| `idle-unsynced` | idle timeout reached but browser edits are not pulled yet | `sync` to keep them |
| `idle-stopped` | the watchdog stopped marimo; the runtime may still exist | `stop` (pulls, terminates) |
| `not-running` | the runtime is gone; only the record is left | `stop` (clears the record) |

`unsynced` lists remote paths with edits the laptop has not pulled.
`cost_usd` is null when the rate is unknown.

### `nbgpu stop --orphans [--backend NAME] [--yes] [--json]`

Terminates runtimes this state directory created that no record tracks.
`--json` requires `--yes`. Runtimes younger than two minutes are skipped (a
start may still own them).

```json
{"orphans": [...], "terminated": ["sb-..."], "skipped": [], "failed": {}}
```

### `nbgpu open NOTEBOOK [--json]`

Prints the `start` shape for the running session and resumes the sync loop.
Blocks; run it in the background.

## Environment variables

| Variable | Effect |
|----------|--------|
| `NBGPU_BACKEND` | Backend when `--backend` is not given (`modal`, `local`). Overrides the header's `provider`. |
| `NBGPU_STATE_DIR` | Where session records, tokens and sync state live. Default `$XDG_STATE_HOME/nbgpu`, else `~/.local/state/nbgpu`. |
| `NBGPU_NO_PROXY` | Any value: do not start the localhost proxy. |
| `NBGPU_NO_HYPERLINKS` | Any value: plain URLs instead of OSC 8 links. |
| `NO_COLOR` | Any value: no colors. |

## Limits

| What | Limit |
|------|-------|
| Project size at start | 2,000 files, 500 MB (ignored files not counted) |
| One file pushed | 50 MB (larger files are skipped) |
| New remote file pulled | 1 MiB, suffix in `.py .md .toml .txt .json .yaml .yml .cfg .ini` |
| Max runtime on Modal | 24h |
| Never synced | `.git`, `.venv`, `__pycache__`, `node_modules`, `.nbgpu`, `__marimo__`, anything `.gitignore`d or in the header's `exclude` |
