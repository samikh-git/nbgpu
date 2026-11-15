---
name: nbgpu
description: Run a local marimo notebook on a remote GPU with the nbgpu CLI and read the results back. Use when the user wants to run, test or iterate on a marimo notebook (a .py file with `app = marimo.App()`) that needs a GPU or a remote machine, when a notebook has a `[tool.nbgpu]` header, or when they mention nbgpu, a remote session, a runtime that is "still billing", sync conflicts (`.remote` files) or orphaned runtimes.
---

# nbgpu: run marimo notebooks on remote GPUs

`nbgpu` takes a marimo notebook on this machine, provisions the compute its
header asks for (Modal by default), syncs the project both ways, and either
serves the marimo UI or runs the notebook as a script. Every command takes
`--json`, prints one JSON document on stdout, and on failure prints one line
on stderr and exits non-zero.

A running runtime costs money until it is stopped. Whatever else happens,
**leave nothing running that the user did not ask to keep**: finish with
`nbgpu status --json` and stop what you started.

## Pick a mode

| Goal | Command | Cost profile |
|------|---------|--------------|
| Run the notebook once, get output and files | `nbgpu run nb.py --json` (add `--pull 'plots/'` etc. for outputs) | Creates a runtime, runs, brings files back, terminates it. Nothing left behind. |
| Iterate (edit, run, edit, run) | `nbgpu start nb.py --detach --json`, then `nbgpu run nb.py --json` repeatedly, then `nbgpu stop nb.py --json` | One runtime billed from start to stop; each run skips provisioning. |
| The user wants the marimo UI | `nbgpu start nb.py` in the background (it never returns: it is the sync loop) | Billed until `nbgpu stop`, or the idle timeout. |

Prefer the one-off `run` unless you expect three or more runs, or provisioning
is slow (large dependencies, a scarce GPU). The first start with a new set of
dependencies builds an image and can take minutes; later ones reuse it.

## The iterate loop

```bash
nbgpu start nb.py --detach --json     # returns once marimo is ready
nbgpu run nb.py --json                # syncs edits up, installs header changes, runs, brings files back
# ... edit nb.py locally, run again ...
nbgpu stop nb.py --json               # final pull, terminate, remove the record
```

`run` executes `python nb.py` on the runtime, which runs every cell in
dependency order. What comes back:

```json
{"exit_code": 0, "output": "...", "truncated": false, "full_output_path": null,
 "pulled": ["metrics.json"], "warnings": [], "env": []}
```

- `output` is stdout followed by stderr. A cell's last expression is *not*
  printed in script mode: `print()` what you need to see.
- The command's own exit code is the notebook's exit code.
- Over `--max-lines` (default 200) the output keeps the first and last halves
  and `full_output_path` names a local file with all of it. Read that file
  with offsets rather than raising `--max-lines` blindly.
- Inside a session, `run` syncs first, so local edits are always what runs,
  and installs packages added to the header (listed in `env`). It starts a
  fresh Python process each time: no state carries over from the previous
  run or from the marimo kernel in the browser, and it shares the GPU with
  that kernel.

## Getting results back

- stdout/stderr come back in `output`. For anything structured, print JSON
  or write a small `.json` file.
- After every `run` (one-off or in a session), files the notebook wrote into
  the project come back and are listed in `pulled`: changes to files you
  have, and new files with a suffix in `.py .md .toml .txt .json .yaml .yml
  .cfg .ini` up to 1 MiB.
- For anything else (plots, `.npy`, a checkpoint, files in a gitignored
  folder), name it: `--pull 'plots/' --pull '*.pt'` (.gitignore syntax,
  relative to the project root, up to 200 MB per file). Matches that did not
  come back are explained in `warnings`.
- Bigger than that, or data you want to keep between sessions: write it to a
  volume (`volumes = { "/data" = "my-volume" }` in the header).
- Deletions are never propagated in either direction.

## Dependencies and compute

Both live in the notebook's header (see the `nbgpu-notebook` skill for the
full reference). To add a package, edit `dependencies` in the header; do not
`pip install` on the runtime, where it is lost on the next start.

- One-off `run`: the header is read each time, so the edit just works.
- In a session, `run` and `sync` install added packages before anything
  else, and wait for the install. A sync loop (`nbgpu start` or `nbgpu open`
  in the background) also installs them within seconds.
- If the install fails, `run` exits non-zero with uv's error and does not
  run the notebook. Fix the requirement in the header and run again.
- Compute (`gpu`, `cpu`, `memory`, `volumes`, Python version) never changes
  in a running session. Stop and start.

`--gpu`, `--cpu`, `--memory`, `--idle-timeout`, `--max-runtime` on `start`
override the header for one session without editing the file.

## Cost and cleanup

- `start --json` reports `rate.usd_per_hour`; `status --json` reports
  `cost_usd` so far per session. Tell the user the rate before starting
  anything above a few dollars an hour, and the total when you stop. These
  are estimates: `prices_source` says whether they came from the account's
  live rates (`live`) or the built-in table (`list`), and `prices_checked`
  how current they are.
- `nbgpu cost --json` reports what Modal actually billed nbgpu's runtimes
  (complete hours only; it can lag an hour). Use it when the user asks what
  something cost, not to track a session that is still running.
- Ask before choosing a bigger GPU than the header names.
- The idle watchdog stops marimo after `idle-timeout` (default 20m) without
  activity, and `max-runtime` (default 8h) is a hard cap. Treat both as
  backstops, not as the way sessions end.
- `status --json` → `orphans` lists runtimes this machine created that no
  record tracks (still billing). Report them. `nbgpu stop --orphans --yes
  --json` terminates them; ask first, since it is not reversible.
  `other_owners` are someone else's and are never touched.

## Conflicts

If both sides changed the same lines, sync reports the path under
`conflicts` and writes the remote version next to the local file as
`<stem>.remote<suffix>` (e.g. `nb.remote.py`). Neither side is overwritten.

1. Read both files and fold what is wanted from the `.remote` copy into the
   local file.
2. Delete the `.remote` copy. That is the resolution signal: the next sync
   pushes the local file.

`run` refuses while the notebook itself is in conflict, and `stop` refuses
with any open conflict. `stop --force` keeps the local versions and the
copies and terminates anyway; use it only when the user agrees the remote
edits can go.

## Rules

- Never print, log or commit the `url` from `start`/`open`: it carries the
  session's access token. Show it to the user only when they ask to open the
  notebook.
- Do not run `modal setup` or handle Modal credentials. If a command fails
  with an authentication error, ask the user to authenticate.
- `start` without `--detach`, and `open`, block forever. Run them only as
  background processes.
- `start` refuses if the notebook already has a running session: use
  `status --json` first and reuse it.
- To try the plumbing with no cloud account, set `NBGPU_BACKEND=local`. It
  runs on this machine with this machine's Python, supports no GPU or
  volumes, and rejects a header that asks for either.

For exact JSON shapes and every flag, read `references/cli.md`. When a
command fails, read `references/troubleshooting.md` before retrying.
