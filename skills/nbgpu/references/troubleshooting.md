# nbgpu troubleshooting

Match the one-line error on stderr against the left column.

| Error contains | Cause | Fix |
|----------------|-------|-----|
| `no session for ...` | `sync`/`open`/`stop` on a notebook that has no session | `nbgpu start`, or use `nbgpu run` for a one-off |
| `no kernel execute API` | `exec` against a marimo older than 0.25 on the runtime | Raise marimo in the header's dependencies, then stop and start |
| `did not load the pushed notebook` | `exec --cell` pushed an edit marimo could not load (often a syntax error) | `python -c "import ast; ast.parse(open('nb.py').read())"`, fix, run again |
| `no cell named ...` | `exec --cell` with a name that is not a cell function name | `nbgpu exec nb.py --list --json`; use a listed name or id |
| `NameError` from `exec` right after an edit | The name is defined in a cell that has not been re-run | `exec --cell` that cell, or `--all` |
| `already has a running session` | `start` on a notebook that is already up | Reuse it: `nbgpu run` / `nbgpu sync`; or `nbgpu stop` first |
| `N sessions are running for ...` | Duplicate sessions for one notebook | `nbgpu stop NOTEBOOK` until one is left |
| `unknown key '...' under [tool.nbgpu]` | Typo in the header (e.g. `gpus`) | Fix the key; the message lists the known ones |
| `Unknown Modal GPU ...` | GPU name Modal does not offer | Use one from the list in the message |
| `invalid duration` / `invalid memory` | Bad value format | `20m`, `1h30m`; `32GiB`, `500MB`, or a bare integer |
| `open sync conflicts: ...` | `stop` with unresolved conflicts | Merge and delete the `.remote` copies, or `stop --force` |
| `has an open sync conflict` (from `run`) | The notebook itself conflicts | Resolve it, then run again |
| `is no longer running` | Runtime hit max runtime, was idle-stopped and reaped, or was terminated elsewhere | `nbgpu stop NOTEBOOK` to clear the record, then start again |
| `has not finished starting` | A start is in flight or was interrupted | Wait, or `nbgpu stop NOTEBOOK` |
| `was stopped after ... without activity` | Idle watchdog stopped marimo | `nbgpu stop NOTEBOOK` (pulls files), then start again |
| mentions the state directory being uploaded | `NBGPU_STATE_DIR` is inside the project | Point it outside the project |
| too many files / too large | Project exceeds 2,000 files or 500 MB | Add patterns to `exclude` in the header or `.gitignore`, or pass a narrower `--project-root` |
| the notebook is ignored | A `.gitignore` or `exclude` pattern matches the notebook | Fix the pattern |
| Modal authentication / token errors | Not logged in to Modal | Ask the user to run `modal setup`; do not handle credentials yourself |
| `unknown backend` | Bad `--backend` / `NBGPU_BACKEND` / header `provider` | Use `modal` or `local` |
| `interrupted` (exit 130, or 128+signal) | The command was signalled | Check `nbgpu status --json` for a leftover session |

## The notebook fails remotely but works locally

- `ModuleNotFoundError`: the package is missing from the header's
  `dependencies`. Add it there.
- A local module is not found: it is outside the project root, or ignored.
  The project root is the nearest ancestor of the notebook with a
  `pyproject.toml` or `.git`; pass `--project-root` to widen it.
- A data file is missing: it is `.gitignore`d, over 50 MB, or outside the
  project root. Put data on a volume.
- `torch.cuda.is_available()` is False: the header has no `gpu`, or the
  backend is `local`.

## A start seems stuck

The first start with new dependencies builds an image (minutes for torch).
Progress lines go to stderr with elapsed seconds. If it fails, the runtime is
terminated before the error is raised, and the message includes marimo's last
output lines.

## Suspected leftover billing

```bash
nbgpu status --json
```

Sessions you no longer need: `nbgpu stop NOTEBOOK`. Entries under `orphans`:
confirm with the user, then `nbgpu stop --orphans --yes --json`. If status
warns that it could not list a backend's runtimes, say so: it means the check
was incomplete, not that nothing is running.
