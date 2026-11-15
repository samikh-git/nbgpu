# CLI reference

```text
nbgpu start   NOTEBOOK   start a session, print its URL, keep syncing
nbgpu open    NOTEBOOK   print a running session's URL, resume syncing
nbgpu sync    NOTEBOOK   one sync pass
nbgpu run     NOTEBOOK   run the notebook as a script remotely
nbgpu status             every session, cost so far, untracked runtimes
nbgpu stop    NOTEBOOK   final pull, terminate, remove the record
nbgpu stop --orphans     terminate runtimes that lost their record
```

Conventions that hold for every command:

- `--json` prints exactly one JSON document on stdout. Progress and warnings
  go to stderr.
- An error is one line on stderr, `nbgpu: error: ...`, with no traceback and
  a non-zero exit code.
- Ctrl-C exits with 130; SIGTERM and SIGHUP are handled the same way and
  exit with 128 + the signal number. Cleanup (final pull, terminating a
  half-started runtime) runs first in all three cases.

## `nbgpu start`

```text
nbgpu start NOTEBOOK [--gpu G] [--cpu N] [--memory M] [--idle-timeout D]
                     [--max-runtime D] [--backend NAME] [--project-root DIR]
                     [--detach] [--json]
```

Creates the runtime, uploads the project, starts marimo and waits until it
answers. Prints the URL (with its access token) and the hourly rate.

Without `--detach` it then runs the sync loop: a two-way sync every second,
live dependency installs, and a localhost proxy that serves marimo's frontend
from your machine so the page loads quickly. Ctrl-C does a final pull and
leaves the runtime running.

With `--detach` it returns as soon as marimo is ready. Nothing syncs in the
background: edits made in the browser reach your disk only when you run
`nbgpu sync`, `nbgpu open` or `nbgpu stop`.

| Option | Meaning |
|--------|---------|
| `--gpu` | e.g. `A10G`, `H100:2`. Overrides the header. |
| `--cpu` | CPU cores. |
| `--memory` | e.g. `32GiB`; a bare number is MiB. |
| `--idle-timeout` | e.g. `20m`; a bare number is seconds. |
| `--max-runtime` | e.g. `8h`; a bare number is seconds. |
| `--backend` | `modal` or `local`. Default: `$NBGPU_BACKEND`, else the header's `provider`. |
| `--project-root` | Directory to upload. Default: the nearest ancestor of the notebook with a `pyproject.toml` or `.git`, else the notebook's directory. |
| `--detach` | Return once marimo is ready. |

`start` refuses when the notebook already has a running session; use `open`.

If anything fails after the runtime exists, the runtime is terminated before
the error is reported.

JSON:

```json
{
  "session_id": "b390261f",
  "notebook": "/abs/path/train.py",
  "backend": "modal",
  "runtime_id": "sb-...",
  "url": "https://...?access_token=...",
  "local_url": "http://127.0.0.1:2718?access_token=...",
  "rate": {"usd_per_hour": 1.1, "complete": true}
}
```

`local_url` is null when there is no proxy (`--detach`, or `NBGPU_NO_PROXY`).
`rate` is null when no price is known; `complete: false` means part of the
shape could not be priced.

## `nbgpu open`

```text
nbgpu open NOTEBOOK [--json]
```

Prints the URL of the notebook's running session and resumes the sync loop,
including the proxy and live installs. Dependency edits made while no loop
was running are picked up on the first tick. JSON output has the `start`
shape.

## `nbgpu sync`

```text
nbgpu sync NOTEBOOK [--json]
```

One two-way pass. Prints what moved:

```text
↑ pushed   train.py
↓ pulled   results.json
⇅ merged   utils.py
✗ CONFLICT model.py  (remote version saved next to it as .remote; delete that copy to resolve)
```

JSON:

```json
{"pushed": [], "pulled": [], "merged": [], "conflicts": [], "skipped": [], "warnings": []}
```

`skipped` lists paths left alone this round: changed while the pass ran, too
large, unreadable, or unsafe to write. They are retried on the next pass.

## `nbgpu run`

```text
nbgpu run NOTEBOOK [--backend NAME] [--max-lines N] [--project-root DIR] [--json]
```

Runs `python NOTEBOOK` on a runtime, prints its output (stdout, then stderr)
and exits with the notebook's exit code.

- If the notebook has exactly one running session, the run happens there,
  after a sync, in a new process next to the marimo kernel.
- Otherwise a runtime is created for the run and terminated afterwards. In
  that case only the output comes back; files the run writes do not.
- Giving `--project-root`, or a `--backend` different from the session's,
  forces a one-off runtime.

Output longer than `--max-lines` (default 200) is shown as its first and last
halves around a marker line, and written in full to a log file under the
state directory.

JSON:

```json
{"exit_code": 0, "output": "...", "truncated": false, "full_output_path": null}
```

## `nbgpu status`

```text
nbgpu status [--backend NAME] [--json]
```

```text
b390261f  /Users/me/proj/train.py
  modal, gpu A10G, ● running 0h12m, rate $1.10/h, cost so far $0.22
  last sync: ...
```

Also listed:

- **Untracked runtimes**: created from this state directory but with no
  session record. They are still billing. `nbgpu stop --orphans` ends them.
- **Runtimes of other owners**: nbgpu runtimes in the same provider account
  created from another machine or state directory. Listed only.

Only backends that a record uses, or that `--backend` / `$NBGPU_BACKEND`
name, are contacted.

JSON:

```json
{
  "sessions": [{
    "session_id": "b390261f", "notebook": "/abs/path/train.py", "backend": "modal",
    "gpu": "A10G", "running": true, "state": "running", "uptime_s": 742.1,
    "cost_usd": 0.22, "conflicts": [], "last_sync_at": 1794711155.99,
    "warning": null, "unsynced": []
  }],
  "orphans": [{"backend": "modal", "runtime_id": "sb-...", "uptime_s": 900.0, "session_tag": "..."}],
  "other_owners": []
}
```

`state` is `running`, `starting`, `idle-unsynced`, `idle-stopped` or
`not-running`; see [Sessions, idle and cost](sessions-and-cost.md#session-states).

## `nbgpu stop`

```text
nbgpu stop NOTEBOOK [--force] [--json]
```

Final pull, then terminate the runtime and remove the session record. Prints
the sync report (same JSON shape as `sync`).

- With open conflicts it refuses. Resolve them, or pass `--force` to keep
  your local versions and the `.remote` copies and terminate anyway.
- If the final pull fails, the runtime and the record are kept so nothing is
  lost. `--force` terminates anyway.
- If the runtime is already gone, the record is removed and a warning says
  that browser edits after the last successful sync could not be recovered.
- A warning lists packages that were installed during the session but are
  not in the notebook's dependencies.

### `nbgpu stop --orphans`

```text
nbgpu stop --orphans [--backend NAME] [--yes] [--json]
```

Lists untracked runtimes of this state directory and terminates them after a
confirmation (`--yes` skips it; `--json` requires `--yes`). Runtimes younger
than two minutes are skipped because a start may still own them. Runtimes of
other owners are never touched.

```json
{"orphans": [...], "terminated": ["sb-..."], "skipped": [], "failed": {}}
```

## Environment variables

| Variable | Effect |
|----------|--------|
| `NBGPU_BACKEND` | Backend to use when `--backend` is not given. Overrides the header's `provider`. |
| `NBGPU_STATE_DIR` | Where session records, tokens, sync state and run logs are kept. Default: `$XDG_STATE_HOME/nbgpu`, else `~/.local/state/nbgpu`. Must be outside any project you upload. |
| `NBGPU_NO_PROXY` | Set to anything to skip the localhost proxy and use the remote URL. |
| `NBGPU_NO_HYPERLINKS` | Set to anything to print plain URLs instead of clickable terminal links. |
| `NO_COLOR` | Set to anything to turn colors off. |
