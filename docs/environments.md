# Environments

The notebook's header is the single source of truth for its Python
environment. nbgpu's job is to keep the runtime matching the header, and to
keep the header matching what you install by other means.

## At start

The runtime image is built from:

- the header's `dependencies`
- the packages in any [`requirements`](notebook-header.md#requirements) files
- marimo, pinned to the version on your machine unless you pinned it
- the `watchdog` package (marimo uses it to notice file changes) and uv

on a Python chosen from [`requires-python`](notebook-header.md#requires-python).

Modal caches images by content. The first start with a given set of
dependencies builds the image (minutes, for something like torch); every
start after that reuses it (seconds). Changing one dependency rebuilds.

## While a session runs

A sync loop (`nbgpu start` or `nbgpu open`) watches for changes from both
directions.

### You edit the header

Add a package to `dependencies`, or to a requirements file the header names,
and save:

```text
↻ env  installing in the runtime: polars
✓ env  installed polars. New imports work now; restart the kernel (marimo's ⋯ menu → Restart kernel) to load upgraded packages
```

The install runs in the background with `uv pip install`; syncing continues
meanwhile. A newly added package can be imported straight away. An upgraded
one is on disk, but the running kernel still has the old version loaded
until you restart it.

If the install fails, the runtime's environment is unchanged and the message
carries uv's output. Fix the header and save to retry.

### You install from marimo's UI

When marimo offers to install a missing package, or you use its packages
panel, the package is installed on the runtime by marimo and nbgpu writes it
into the header on your machine:

```text
✓ env  recorded +polars in train.py's dependencies (changed in marimo)
```

Only the `dependencies` line is rewritten. The rest of the header and the
file are kept byte for byte.

This is seen by the localhost proxy, so it works when you use the `open:`
link. Installs made through the `remote:` link, or by anything else on the
runtime, are not recorded as they happen. Instead `nbgpu stop` lists them:

```text
warning: installed during the session but not in train.py's dependencies: polars; add them to its header ...
```

### What cannot change in place

| Change | What happens |
|--------|--------------|
| Remove a dependency | Reported; stays installed until the next start |
| `gpu`, `cpu`, `memory`, `volumes`, Python version | Reported; applies on the next start |

```text
! env  gpu changed (A10G → H100); it applies on the next start (nbgpu stop, then nbgpu start)
```

## Without a sync loop

Live installs are part of the sync loop. A session started with `--detach`
and left alone does not get them. `nbgpu open` resumes the loop, and its
first tick compares the header with what the runtime has, so edits made in
between are applied then.

A one-off `nbgpu run` builds its runtime from the header as it is at that
moment, so it always has the current dependencies.

## The local backend

The local backend does not build an environment. It runs with the Python
that nbgpu itself is installed in, which must already have what the notebook
imports. Dependency changes are reported with a note to restart rather than
installed, because that Python is yours and not a disposable one.
