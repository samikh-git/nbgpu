---
name: nbgpu-dev
description: Work on the nbgpu codebase itself (this repository). Use when changing anything under src/nbgpu or tests, adding a CLI command or flag, adding a `[tool.nbgpu]` header key, adding a provider backend, touching sync, the idle watchdog, the proxy or env sync, or running and debugging the test suite.
---

# Developing nbgpu

## Layout

| Module | Owns |
|--------|------|
| `cli.py` | Click commands, output, the sync loop. Thin: logic lives in `lifecycle`. |
| `lifecycle.py` | start / run / stop / status / orphans. The start sequence is the module docstring. |
| `spec.py` | Parsing and validating the PEP 723 header into `NotebookSpec`. |
| `header.py` | Rewriting only the `dependencies` line of a header. |
| `sync.py` | Two-way sync engine: hash compare, three-way merge, conflicts, manifest. |
| `ignore.py` | gitignore layering for what is left out of sync. |
| `envsync.py` | Live dependency installs and what cannot change in place. |
| `watchdog.py` | Runs **inside the runtime**; stops marimo when idle. Stdlib only. |
| `kernel.py` | `nbgpu exec`: marimo's kernel API from the laptop; ensure a kernel, run, parse results. |
| `kernel_runner.py` | Runs **inside the marimo kernel** (sent as code). Stdlib + marimo private APIs only. |
| `proxy.py` | Localhost proxy serving marimo's static assets from this machine. |
| `cost.py` | Price sources in order: `$NBGPU_PRICES`, cached live rates (`prices-live.json` in the state dir), `prices.toml`; rates, running totals, the "as of" disclaimer. |
| `session.py` | Session records on disk (state directory). |
| `backends/base.py` | The provider contract (`Backend` protocol). |
| `backends/modal.py`, `backends/local.py` | Implementations. |

Each module's docstring is its design document and is kept exact. Read it
before changing the module, and update it in the same change. "Decisions
beyond the contract" sections record choices the docstring's contract did not
force; add to them rather than leaving a decision implicit.

`docs/architecture.md` has the cross-module picture.

## Tests

```bash
uv run pytest                    # unit + contract (local backend) + e2e, ~1-2 min
uv run pytest tests/unit -q      # fast
uv run pytest tests/unit tests/contract   # skip the ones that start real marimo
NBGPU_MODAL_TESTS=1 uv run pytest -m modal   # creates real Modal sandboxes: costs money, ask first
```

- `tests/unit`: pure logic, with `tests/fakes.py` standing in for a backend.
- `tests/contract`: one suite run against every backend. A new backend must
  pass it unchanged.
- `tests/e2e`: real marimo processes through the local backend.

Never run the `modal` marker without the user's go-ahead. Tests shrink
timing constants by assigning module globals (`watchdog.CHECK_INTERVAL_S`,
`lifecycle.MARIMO_ARGV`); code reads those globals at call time for that
reason, so do not bind them as default arguments.

## Invariants not to break

- **A failed start leaves nothing billing.** After `backend.create`, every
  failure path terminates the runtime before raising. The provisional record
  is saved *before* `create` so a slow create is never taken for an orphan.
- **Sync never overwrites unseen content.** Every write is compare-and-swap
  against the hash the decision was made on. A conflict writes a copy, never
  over either side. Deletions are not propagated.
- **Edits come before output.** On shutdown the final pull runs before any
  printing, and printing never raises (`_say`).
- **`watchdog.py` imports nothing from nbgpu.** It is uploaded as a single
  file. Constants it shares with `sync.py` (`PULL_SUFFIXES`,
  `EXCLUDED_DIRS`) are duplicated on purpose; change both.
- **Only this owner's runtimes are ever terminated.** Orphan cleanup filters
  on the state directory's owner id.
- **Tokens stay out of the project tree**, locally (state dir, 0600) and
  remotely (next to `remote_root`, not in it).
- **`kernel_runner.py` imports nothing from nbgpu**, defines only
  `_nbgpu*` names (it runs in the user's kernel namespace), and must never
  wait inside the kernel for something the kernel itself has to do (a file
  reload): return `pending` and let `kernel.run_in_kernel` retry.
- **`exec` streams through `lifecycle.LiveLines`**, which must emit
  exactly the lines `result.output` holds (tested against
  `_bound_output`). The kernel id cache (`<id>.kernel`) is removed in
  `_forget`; a stale id surfaces as `UnknownKernelSession` before any
  output, so the retry never duplicates output.
- **`exec` leaves no websocket connected.** A connected client makes the
  user's browser a read-only viewer.
- **`--json` stdout is one JSON document.** Progress goes to stderr.
- **Missing capabilities are errors, never approximations** (`UnsupportedError`).

## Common changes

**A new `[tool.nbgpu]` key**: add to `KNOWN_KEYS` and parse it in `spec.py`
(with a `SpecError` naming the key); thread it through `ComputeSpec` or
`NotebookSpec`; honour it in each backend or raise `UnsupportedError`; decide
in `envsync.snapshot` whether a change mid-session is reported; test in
`tests/unit/test_spec.py`; document in `docs/notebook-header.md` and
`skills/nbgpu-notebook/SKILL.md`.

**A new CLI flag or command**: implement in `lifecycle.py`, wire in `cli.py`
with `--json`, update the `cli.py` docstring, `docs/cli.md` and
`skills/nbgpu/references/cli.md`.

**A new backend**: implement `Backend` from `backends/base.py`, register it
in `backends/__init__.py` (`KNOWN_BACKENDS`, lazy import in `get_backend`),
add it to the contract suite's fixtures in `tests/contract/conftest.py`, add
prices to `prices.toml`. Requirements that are easy to miss: `write_file`'s
check-and-replace must be one step inside the runtime; helper execs must set
`NBGPU_INTERNAL=1` so the watchdog ignores them; tags must be set atomically
at creation.

**Warm starts**: `spec.compute.cache_volume` → Modal mounts it at
`CACHE_MOUNT` with `cache_env()`. Snapshots: `lifecycle.stop(snapshot=)`
calls `backend.snapshot` after the final pull and before terminate (failure
is a warning, never keeps the runtime); records live in
`<state>/snapshots.json` keyed by notebook path; `matching_snapshot` keys on
backend, sorted applied deps, requires-python, local Python minor and
marimo version, never compute. A start from a snapshot must call
`_reset_runtime_dir` before uploading, or the old project and token leak in.
Modal memory snapshots were probed and rejected (no GPU sandboxes; only the
entrypoint's process tree survives). `cell-cache` symlinks the notebook's
`__marimo__/cache` on the runtime into the volume (`cell_cache_paths`,
`_link_cell_cache`, run after the upload in start and one-off run). Do not
switch it to PYTHONPYCACHEPREFIX: bytecode on the volume made `import torch`
take 210 s.

**Prices**: normally the account's live rates are used: `start` and `cost`
call `lifecycle.refresh_live_prices`, which on a `billing` backend caches
`current_prices()` once a day. `src/nbgpu/prices.toml` is the fallback:
Modal's list prices per second, copied by hand, with a `checked` date. When
updating it, compare every number with https://modal.com/pricing (or
`modal.Workspace.from_context().billing.rates()`) and bump `checked` even if
nothing changed; output says "may be out of date" once it is older than
`cost.STALE_AFTER_DAYS`. New Modal GPU names need a rate key in
`backends/modal.py` `_RATE_KEY_ALIASES` unless the key is the name
lower-cased with `-` as `_`.

**Env installs outside the loop**: `run` and `sync` call
`EnvWatcher.apply()`, which installs synchronously. The loop's `tick()` and
`apply()` share a per-session flock (`<id>.env.lock`) and both re-read the
applied state from the record after taking it; keep that, or a loop and a
`run` install twice or overwrite each other's record.

## Trying a change by hand

```bash
NBGPU_BACKEND=local NBGPU_STATE_DIR=/tmp/nbgpu-dev uv run nbgpu start some_cpu_notebook.py
```

The local backend needs no account and costs nothing, but has no GPU, no
volumes and no live installs. Use a notebook whose header has no `gpu`. A
separate `NBGPU_STATE_DIR` keeps test sessions away from real ones.

## Docs and skills travel with the code

User docs are in `docs/`, agent skills in `skills/` (usage) and
`.claude/skills/` (this one). A behaviour change that a user or agent would
notice is not done until those say the same thing as the code.
