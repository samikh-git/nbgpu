---
name: nbgpu-notebook
description: Write or edit a marimo notebook so it runs on a remote GPU with nbgpu. Use when creating a new marimo notebook for GPU work, adding or changing the `# /// script` header (dependencies, `[tool.nbgpu]` gpu/cpu/memory/volumes/idle-timeout), converting a script or Jupyter notebook into a marimo notebook, or fixing marimo structure errors (redefined variables, cells not running in script mode).
---

# Writing marimo notebooks for nbgpu

An nbgpu notebook is one plain Python file with two parts: a PEP 723 header
that declares dependencies and compute, and marimo cells. The header is the
single source of truth for the remote environment.

## Skeleton

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "torch"]
#
# [tool.nbgpu]
# gpu = "A10G"
# idle-timeout = "20m"
# max-runtime = "2h"
# ///

import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    return mo, torch


@app.cell
def _(torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    return (device,)


if __name__ == "__main__":
    app.run()
```

The `if __name__ == "__main__": app.run()` block is required: it is what
`nbgpu run` executes.

## Header reference

Every `[tool.nbgpu]` key is optional. An unknown key is an error, not ignored.

| Key | Example | Default | Notes |
|-----|---------|---------|-------|
| `provider` | `"modal"` | `"modal"` | `modal` or `local`. `--backend` and `$NBGPU_BACKEND` override it. |
| `gpu` | `"A10G"`, `"H100:2"` | none (CPU only) | Modal: T4, L4, A10, A10G, L40S, A100, A100-40GB, A100-80GB, RTX-PRO-6000, H100, H100!, H200, B200, B200+, B300. `:n` for several (up to 8; 4 for A10). |
| `cpu` | `8` | provider default | Physical cores; fractions allowed. |
| `memory` | `"32GiB"` | provider default | `MiB`, `GiB`, `MB`, `GB`; a bare integer is MiB. |
| `idle-timeout` | `"20m"` | `20m` | Stop after this long with no file changes, running cells, CPU or GPU use. |
| `max-runtime` | `"8h"` | `8h` | Hard cap. Modal allows at most 24h. |
| `volumes` | `{ "/data" = "my-datasets" }` | none | Absolute mount path → provider volume name. The volume must already exist (`modal volume create my-datasets`); nbgpu does not create it. |
| `exclude` | `["data/", "*.ckpt"]` | none | gitignore-style, relative to the project root; never uploaded or pulled. `.gitignore` is always respected. |
| `requirements` | `"requirements.txt"` | none | Path or list, relative to the notebook. Package lines and `-r` includes only. |

Also read from the header:

- `dependencies`: installed into the runtime image. List `marimo` bare (no
  version): nbgpu pins it to the local version so the UI matches.
- `requires-python`: picks the runtime's Python (3.11 to 3.14 on Modal). The
  local minor version is used when it satisfies the constraint.

Choose the smallest GPU that fits. Rough guide: T4/L4 for inference and small
fine-tunes, A10G for mid-size training, A100/H100 only when memory or speed
demands it. Set `max-runtime` near the expected run length so a hung job
cannot bill for eight hours.

## marimo rules that bite

marimo runs cells as a dependency graph, not top to bottom.

- **One definition per name.** A global variable may be assigned in exactly
  one cell. Reusing `df`, `fig`, `i`, `model` in two cells is an error.
- **Underscore names are cell-local.** `_tmp`, `_i`, `_fig` can repeat in
  every cell. Use them for loop variables and scratch values.
- **Signature = reads, return = writes.** A cell's parameters are the globals
  it uses; its `return` tuple is the globals it defines. Keep both accurate
  when editing by hand, including the trailing comma for one name:
  `return (device,)`.
- **Imports are definitions.** Import a module in one cell and return it;
  other cells take it as a parameter.
- **No mutation across cells.** marimo does not see `lst.append(x)` in
  another cell as a dependency. Mutate where you define, or make a new name.
- **No cycles** between cells.
- **Name the cells that matter.** `def train(...)` instead of `def _(...)`
  makes errors and diffs readable.

Check structure without running anything:

```bash
uvx marimo check nb.py
```

## Writing for script mode

`nbgpu run` executes the file as a script and returns stdout and stderr.

- The last expression of a cell is rendered in the UI but printed nowhere in
  script mode. `print()` the values you want back, ideally one JSON line per
  result.
- Write small results as `.json` / `.txt` / `.md` into the project directory
  to have them synced back. Large artifacts go to a volume path.
- Guard UI-only cells so the script does not block or fail:
  `mo.app_meta().mode == "script"` tells you which mode you are in.
- UI elements (`mo.ui.slider`) take their default value in script mode. Put
  the defaults you want a headless run to use.
- Exit non-zero on failure (let exceptions propagate, or `raise SystemExit(1)`):
  the exit code is how `nbgpu run` reports success.
- Fail fast: assert `torch.cuda.is_available()` in an early cell so a
  misconfigured header costs seconds, not a full data load.

## Editing a notebook that has a live session

Edit the local file. The sync loop pushes it within a second or two and
marimo (started with `--watch`) reloads the changed cells in the browser. Do
not edit the copy on the runtime. If the user is editing in the browser at
the same time, non-overlapping edits merge; overlapping ones produce a
`.remote` conflict copy (see the `nbgpu` skill).

Adding to `dependencies` while a sync loop runs installs the package live.
Changing `gpu`, `cpu`, `memory`, `volumes` or `requires-python` needs
`nbgpu stop` and `nbgpu start`.

## Converting from Jupyter

```bash
uvx marimo convert notebook.ipynb -o notebook.py
```

Then fix what the converter cannot: repeated variable names across cells
(rename or prefix with `_`), `%magic` and `!shell` lines, and
`display(...)` calls (make the value the cell's last expression). Add the
header last.
