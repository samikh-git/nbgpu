# Sync and conflicts

A session keeps two copies of your project: the one on your machine and the
one on the runtime. Editing in your editor and editing in the marimo UI are
both first-class, so sync runs in both directions, and its one hard rule is
that it never overwrites content it has not seen.

## What is uploaded

At start, the **project root** is uploaded: the nearest ancestor of the
notebook that contains a `pyproject.toml` or a `.git` (searched below your
home directory only), or the notebook's own directory if there is none.
`--project-root DIR` picks it explicitly. A notebook sitting directly in your
home directory is refused rather than uploading all of it.

The project must be under 2,000 files and 500 MB after exclusions. Start
fails with a message otherwise.

## What is left out

Never synced, in either direction:

- `.git`, `.venv`, `__pycache__`, `node_modules`, `.nbgpu`, `__marimo__`
- anything matched by a `.gitignore` in the project (nested ones included)
  or by `.git/info/exclude`
- anything matched by `exclude` in the notebook's
  [header](notebook-header.md#exclude)
- single files over 50 MB (reported as skipped)

The rules come from the local project only. Your global git excludes are not
consulted, so a session behaves the same from any machine. Editing a
`.gitignore` during a session takes effect on the next pass.

## What comes back

Files that exist on both sides are always synced both ways.

A file that is *new* on the runtime is pulled only if it is small text: at
most 1 MiB, with one of these suffixes:

```text
.py .md .toml .txt .json .yaml .yml .cfg .ini
```

This keeps checkpoints, plots and datasets written by a notebook from
flooding your working tree. For those, mount a
[volume](notebook-header.md#volumes).

Deletions are not propagated in either direction. Deleting a file locally
leaves the remote copy, and the other way round.

## When it runs

| | |
|---|---|
| `nbgpu start`, `nbgpu open` | every second, until Ctrl-C |
| `nbgpu sync` | once |
| `nbgpu run` (in a session) | once, before running |
| `nbgpu stop` | once, before terminating |

With `nbgpu start --detach` and nothing else, there is no background sync.
Browser edits stay on the runtime until one of the commands above runs, and
are lost if the runtime ends first.

## How a pass decides

Each pass compares, per file, the content hash on each side with the last
version both sides agreed on:

| Local | Remote | Result |
|-------|--------|--------|
| unchanged | changed | pull |
| changed | unchanged | push |
| changed | changed | three-way merge |
| new | absent | push |
| absent | new | pull, if it is small text (above) |

A three-way merge uses `git merge-file`. Edits to different parts of a file
merge cleanly and the result is written to both sides. This is what makes it
safe for you to edit a helper function in your editor while a cell is being
changed in the browser.

## Conflicts

When both sides changed the same lines, neither is overwritten. The remote
version is written next to your file:

```text
train.py           your version, untouched
train.remote.py    the runtime's version
```

The path is reported as a conflict on every pass and in `nbgpu status` until
it is resolved.

To resolve:

1. Compare the two files and edit `train.py` into what you want.
2. Delete `train.remote.py`.

Deleting the copy is the signal. On the next pass your file is pushed. If
the remote changed again in the meantime, a fresh copy appears instead of
your push clobbering it.

While a conflict is open:

- `nbgpu run` refuses if the notebook itself is in conflict.
- `nbgpu stop` refuses. `nbgpu stop --force` keeps your local versions and
  the `.remote` copies, and terminates.

## Edits made at the same moment

The marimo UI autosaves about a second after you stop typing, and your
editor saves whenever you tell it to, so files change while a pass is
running. Every write is therefore conditional on the file still having the
content the decision was based on. If it does not, the path is skipped for
that round and shows up under `skipped`; the next pass sees both changes and
merges them.

## When the runtime disappears

If a runtime ends without a final pull (max runtime reached, terminated from
the provider's dashboard), edits made in the browser after the last
successful sync are gone. `nbgpu status` and `nbgpu stop` say so and show
the time of that last sync. Your local files are intact as of that time.

A running sync loop makes this window about one second wide. With
`--detach`, it is as wide as the time since you last ran `nbgpu sync`.
