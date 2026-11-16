# CLI reference

```text
nbgpu start   NOTEBOOK   start a session, print its URL, keep syncing
nbgpu open    NOTEBOOK   print a running session's URL, resume syncing
nbgpu sync    NOTEBOOK   one sync pass, then install header changes
nbgpu run     NOTEBOOK   run the notebook as a script remotely, bring back its files
nbgpu exec    NOTEBOOK   run code or cells in the session's live kernel
nbgpu status             every session, cost so far, untracked runtimes
nbgpu cost               what the provider actually billed
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
                     [--detach] [--fresh] [--json]
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
| `--fresh` | Build the environment even if the notebook has a matching [snapshot](warm-starts.md#filesystem-snapshots). |

If the notebook has a matching filesystem snapshot, the runtime boots from
it instead of building the environment; the first progress line says
"from the snapshot of ...".

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
  "rate": {"usd_per_hour": 1.1, "complete": true, "prices_checked": "2026-11-15", "prices_source": "live"},
  "from_snapshot": null,
  "timings": {"create": 11.4, "upload": 1.3, "ready": 1.4, "total": 14.1}
}
```

`local_url` is null when there is no proxy (`--detach`, or `NBGPU_NO_PROXY`).
`from_snapshot` is the snapshot image the runtime booted from, or null.
`timings` are seconds per step; see [Warm starts](warm-starts.md#seeing-what-a-start-costs).
`rate` is null when no price is known; `complete: false` means part of the
shape could not be priced. `prices_source` says where the prices came from
(`live`: your account's rates, `list`: the built-in table, `file`:
`NBGPU_PRICES`) and `prices_checked` how current they are; see
[Cost](sessions-and-cost.md#cost).

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

Then it installs any dependencies added to the header since the runtime last
had them, and waits for the install:

```text
✓ env  installed polars in the runtime
```

This is how a `--detach` session gets header changes. See
[Environments](environments.md#without-a-sync-loop).

JSON:

```json
{"pushed": [], "pulled": [], "merged": [], "conflicts": [], "skipped": [], "warnings": [], "env": []}
```

`skipped` lists paths left alone this round: changed while the pass ran, too
large, unreadable, or unsafe to write. They are retried on the next pass.
`env` holds the dependency messages, without the symbol.

## `nbgpu run`

```text
nbgpu run NOTEBOOK [--backend NAME] [--max-lines N] [--pull PATTERN]... [--project-root DIR] [--json]
```

Runs `python NOTEBOOK` on a runtime, prints its output (stdout, then stderr)
and exits with the notebook's exit code.

- If the notebook has exactly one running session, the run happens there:
  sync, install any header changes, then run in a new process next to the
  marimo kernel. A failed install stops the run before it starts.
- Otherwise a runtime is created for the run and terminated afterwards.
- Giving `--project-root`, or a `--backend` different from the session's,
  forces a one-off runtime.

Afterwards, in both cases, the files the run wrote come back:

- One sync pass brings back what [`nbgpu sync`](sync.md#what-comes-back)
  would: changes to files you have, and new small text files such as
  `metrics.json`.
- `--pull PATTERN` brings back anything else matching the pattern: plots,
  checkpoints, files in gitignored folders. Patterns use `.gitignore`
  syntax, relative to the project root (where the run's working directory
  is): `--pull 'plots/'`, `--pull '*.png'`, `--pull outputs/model.pt`.
  Repeat the flag for several. Each file may be up to 200 MB; anything
  bigger belongs on a [volume](notebook-header.md#volumes).

`--pull` never overwrites a project file that sync manages, because that
could lose local edits; sync handles those. It creates missing files, and
replaces ignored files from earlier runs.

```text
trained
↓ pulled   metrics.json
↓ pulled   plots/curve.png
```

Output longer than `--max-lines` (default 200) is shown as its first and last
halves around a marker line, and written in full to a log file under the
state directory.

JSON:

```json
{"exit_code": 0, "output": "...", "truncated": false, "full_output_path": null,
 "pulled": ["metrics.json", "plots/curve.png"], "warnings": [], "env": []}
```

`warnings` lists files that matched but were not brought back (too large,
unsafe path), sync conflicts, and sync errors after the run. None of them
change the exit code.

## `nbgpu exec`

```text
nbgpu exec NOTEBOOK CODE            evaluate CODE in the live kernel
nbgpu exec NOTEBOOK -f FILE         the code from a file (- for stdin)
nbgpu exec NOTEBOOK --cell NAME...  run cells and what depends on them
nbgpu exec NOTEBOOK --all           run every cell
nbgpu exec NOTEBOOK --list          the cells, with ids and status
      [--sync | --no-sync] [--max-lines N] [--json]
```

Works inside the marimo kernel of the notebook's running session, where
the notebook's variables already are. A model loaded on the GPU stays
loaded: asking about it takes a fraction of a second instead of re-running
the notebook. `run`, by contrast, starts a new Python process each time.

```bash
nbgpu exec train.py "model.config.hidden_size"
nbgpu exec train.py "losses[-5:]"
nbgpu exec train.py --cell evaluate
```

```text
768
[0.412, 0.398, 0.391, 0.388, 0.385]
```

**Evaluating code.** The code runs in marimo's scratchpad, with the
notebook's variables in scope. If its last statement is an expression,
`repr()` of the value is printed, the way a Python prompt shows it. `print`
works as usual, and so does top-level `await`. Errors show a traceback
with the code's own line numbers and exit 1.

Names the code defines are thrown away afterwards and never become
notebook variables, but objects it changes in place stay changed:
`model.eval()` or `history.append(...)` affect the live kernel. To keep
something, put it in a cell.

**Running cells.** `--cell` takes a cell's function name (`def evaluate():`
in the file) or its id from `--list`, and is repeatable. marimo then also
runs the cells that depend on it. Cells above it are not re-run, so an
edit to the evaluation cell re-evaluates without reloading the model. A
failing cell prints marimo's traceback and exits 1.

**Your edits.** With `--cell` and `--all`, `exec` syncs first (as `run`
does), installs header changes, and waits until marimo has loaded the new
file, about a second, so the cell runs the code you just wrote. A bare
expression does not sync, which keeps it fast. `--sync` and `--no-sync`
override either default.

**A kernel that has not run.** Headless marimo starts its kernel only when
something connects, with every cell unrun. If no cell has run yet, `exec`
first runs the whole notebook and says so (`ran_notebook`), so the
variables exist. After that, nothing re-runs unless asked.

**Sharing the kernel.** It is the same kernel the browser shows. A browser
opened later is a normal editor. While a cell runs in the browser, `exec`
waits for it. Ctrl-C interrupts the code `exec` started.

JSON:

```json
{"ok": true, "value": "768", "error": null, "output": "", "truncated": false,
 "full_output_path": null, "cells": [], "kernel_created": false, "ran_notebook": false,
 "pushed": [], "warnings": [], "env": []}
```

`value` is the last expression's `repr()` (null if none). `output` is
everything printed, stdout and stderr in the order they arrived, bounded
like `run`'s. `error` is one line when `ok` is false: the exception, or
`cell NAME: ...` for each failing cell. `cells` lists all cells for `--list`
(`name`, `id`, `status`, `errors`, `first_line`) and the cells left in an
error state otherwise. The exit code is 0 when `ok`, else 1.

`exec` needs a running session (`nbgpu start`) and marimo 0.25 or newer,
which has the kernel execute API it uses.

## `nbgpu cost`

```text
nbgpu cost [--days N] [--backend NAME] [--json]
```

What the provider actually billed for nbgpu's runtimes over the last `N`
complete days (default 7, at most 31) plus today so far, per day and per
resource:

```text
billed for nbgpu's runtimes on modal, 2026-11-08 to 2026-11-15 20:00 UTC:
  2026-11-14  <$0.01
  2026-11-15  $0.45
  total       $0.46
  by resource: A10G $0.41, CPU $0.03, Memory <$0.01, Network Egress <$0.01
```

This is the bill, not an estimate, with two limits: it covers complete hours
only, and the provider's report can lag by an hour or more. `nbgpu status`
estimates what is running right now. Modal reports cost per App, and all of
nbgpu's runtimes share one, so the total covers every session rather than
breaking down by session.

It also refreshes the cached copy of your account's current prices (see
[Cost](sessions-and-cost.md#cost)).

JSON:

```json
{"backend": "modal", "start": "2026-11-08T00:00:00+00:00", "end": "2026-11-15T20:00:00+00:00",
 "days": [{"date": "2026-11-15", "usd": 0.45}], "by_resource": {"A10G": 0.41, "CPU": 0.03},
 "total_usd": 0.46, "prices_refreshed": true, "prices_error": null}
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
    "cost_usd": 0.22, "prices_checked": "2026-11-15", "prices_source": "live", "conflicts": [], "last_sync_at": 1794711155.99,
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
nbgpu stop NOTEBOOK [--force] [--snapshot | --no-snapshot] [--json]
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
- `--snapshot` saves the runtime's filesystem before terminating, so the
  next `start` boots from it (default: the header's
  [`snapshot`](notebook-header.md#snapshot) key). JSON adds `"snapshot":
  {"image", "created_at", "seconds", ...}`, or null. A failed snapshot is a
  warning; the runtime is terminated regardless.

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
| `NBGPU_PRICES` | Path to a price file that replaces the built-in one. See [Cost](sessions-and-cost.md#cost). |
| `NO_COLOR` | Set to anything to turn colors off. |
