# Architecture

This page is the map. Each module's docstring is the detailed, authoritative
description of that module and is kept in step with the code.

## The shape of a session

```text
 your machine                                   runtime (Modal sandbox)
 ───────────────────────────────                ─────────────────────────────────
 browser ──► LocalProxy ─────── tunnel ───────► marimo edit --watch notebook.py
              │  serves /assets locally                 │ reads/writes
              │                                         ▼
 editor ──► project files ◄── SyncEngine ─────► /workspace/project
 agent  ──►      │              (poll 1/s,
                 │               hash compare,          watchdog.py
                 ▼               CAS writes)            (idle detection)
            notebook header
                 │
            EnvWatcher ──── uv pip install ───────────► runtime's Python

 state dir: session records, tokens, sync state
```

Everything on the left runs inside one `nbgpu start` (or `nbgpu open`)
process. The runtime side is marimo, the watchdog, and short-lived helper
commands.

## Modules

| Module | Role |
|--------|------|
| `cli.py` | Commands, output formatting, and the sync loop that ties the pieces together each tick. |
| `lifecycle.py` | Start, run, stop, status, orphan detection. Owns the ordering guarantees. |
| `spec.py` | Header → `NotebookSpec` / `ComputeSpec`, with validation. |
| `header.py` | Rewrites the header's `dependencies` and nothing else. |
| `sync.py` | `SyncEngine`: the two-way sync algorithm. |
| `ignore.py` | Which paths are left out (gitignore layers + `exclude`). |
| `envsync.py` | `EnvWatcher`: header changes → live installs or "next start" notes. |
| `watchdog.py` | Idle detection. Runs on the runtime; standard library only. |
| `proxy.py` | `LocalProxy`: local static assets, forwarded API and websockets, package-request observation. |
| `cost.py` | Price sources (`NBGPU_PRICES`, cached live rates, `prices.toml`); hourly rate, cost so far. |
| `prices.toml` | The price table: list prices per second, with the date they were checked. |
| `session.py` | `Session` records and `SessionStore`. |
| `backends/base.py` | The `Backend` protocol. |
| `backends/modal.py` | Modal Sandboxes. |
| `backends/local.py` | Same contract on this machine, for tests and trials. |

Dependencies point one way: `cli` → `lifecycle` → (`sync`, `envsync`,
`session`, `spec`, `cost`) → `backends`. Backends know nothing about marimo,
sync or cost.

## The backend contract

A backend provides a small set of operations on a *runtime*:

| Operation | Purpose |
|-----------|---------|
| `create(request)` | Provision a runtime with the spec's environment and an empty project directory |
| `get(id)`, `find(tags)` | Look runtimes up; tags are how records and runtimes are matched |
| `exec(argv)`, `spawn(argv)` | Run to completion; start in the background |
| `expose(port)` | A URL on which the laptop reaches a port in the runtime |
| `read_file`, `write_file(expect=)`, `hash_files` | File transfer, with compare-and-swap writes |
| `terminate` | Stop and release |

Optional abilities are declared as capabilities (`gpu`, `volumes`,
`max-runtime`, `idle-timeout`, `live-install`, `billing`, `snapshot`). A backend that
lacks one raises `UnsupportedError` for a spec that needs it; nothing is
approximated. `billing` adds two operations: `current_prices()` (the
account's hourly rates, cached for estimates) and `spend(start, end,
resolution)` (what this tool's runtimes were billed, for `nbgpu cost`).
`snapshot` adds `snapshot(handle)`, returning an id that
`RuntimeRequest.base_image` boots instead of building the environment.
`spec.compute.cache_volume` is mounted at `CACHE_MOUNT` with `cache_env()`
by backends with volumes, and ignored by the others.

Everything else (starting marimo, sync, idle detection, cost) is built on
these operations, which is what keeps a new provider small.

## Start sequence

1. Parse the header; pick the project root; check size limits.
2. Save a provisional record (`starting`) **before** creating anything, so a
   slow or interrupted create is never mistaken for an orphan.
3. `backend.create` with tags for the session, the notebook and the owner.
4. In parallel: push the project, upload the token and the watchdog script.
5. Spawn marimo and the watchdog together; poll marimo's `/health` until
   ready, then confirm the watchdog is still alive.
6. Save the complete record.

Any failure after step 3 terminates the runtime before the error is raised.
Terminates are shielded from a second Ctrl-C and bounded by a timeout.

## Sync in one paragraph

Each pass hashes both sides and compares each file with the *base*, the last
content both sides agreed on. One side changed: copy it across. Both
changed: three-way merge against the base with `git merge-file`; if that
fails, write the remote version to a `.remote` copy and leave both originals
alone. Every write is a compare-and-swap on the hash the decision used, so
an edit that lands mid-pass causes a skip and a retry rather than a lost
update. State (bases, open conflicts, a stat cache) is a JSON file per
session, guarded by a file lock so `run` and `sync` can work beside a
running loop.

The engine also publishes a manifest of what the laptop has to the runtime.
The watchdog compares the project with it to tell whether stopping now would
strand edits.

## Why a local proxy

marimo's frontend is about 250 static files. Through a provider tunnel they
take seconds, and since every session gets a new tunnel hostname the browser
cache never helps. The proxy serves those files from the marimo installed
on the laptop, on a stable `127.0.0.1` origin, and forwards the page, API
and websocket. It serves locally only after confirming the remote marimo
reports the same version; the runtime's marimo is pinned to the local
version to make that the normal case.

Sitting in that position, it also sees marimo's package-install requests,
which is how UI installs get recorded in the header.

## Why a watchdog in the runtime

Provider idle timeouts count processes and connections. marimo is always
running and the sync loop connects every second, so they would never fire.
The watchdog measures what indicates use (file edits, busy cells, CPU, GPU),
ignores nbgpu's own helper commands (marked with `NBGPU_INTERNAL=1`), and
stops marimo. It works identically on every backend. Modal's own short idle
timeout then reaps the quiet sandbox if no laptop is around to do it.

## Security notes

- The marimo server requires a random token. It is stored in the state
  directory (0700, files 0600) and on the runtime beside the project
  directory, never inside it, so it is never synced or committed.
- The state directory may not be inside an uploaded project.
- Remote paths are validated before anything is written locally: no absolute
  paths, no `..`, nothing through a symlink.
- The proxy adds no access: marimo still checks the token on every request.
- Orphan cleanup only ever terminates runtimes carrying this state
  directory's owner id.

## Adding a backend

1. Implement the `Backend` protocol in `src/nbgpu/backends/<name>.py`.
2. Register it in `src/nbgpu/backends/__init__.py`.
3. Add it to the fixtures in `tests/contract/conftest.py`. The contract suite
   is the definition of correct; a backend passes it unchanged.
4. Add its list prices to `src/nbgpu/prices.toml`. If the provider can
   report prices and spend, add `billing` and implement `current_prices()`
   and `spend()`.

Points the contract is strict about: `write_file`'s check and replace must
be one atomic step inside the runtime; tags must be set at creation, not
after; helper commands must set `NBGPU_INTERNAL=1`.

## Tests

| Directory | What | Needs |
|-----------|------|-------|
| `tests/unit` | Pure logic against fakes | nothing |
| `tests/contract` | The backend contract, once per backend | Modal only with `NBGPU_MODAL_TESTS=1` |
| `tests/e2e` | Real marimo through the local backend | nothing |

## The shim

`shim/` is a separate prototype: a fake Python interpreter that lets the
stock marimo VS Code extension run its kernel over SSH. It shares no code
with `nbgpu`. See [`shim/README.md`](../shim/README.md).
