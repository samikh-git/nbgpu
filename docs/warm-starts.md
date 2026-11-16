# Warm starts

A remote runtime starts empty. Without help, every session downloads the
same model weights again, recompiles the same kernels, and, after you
installed a package live, rebuilds the environment image. nbgpu keeps two
things between sessions to avoid that:

| | What it keeps | On by default | Cost |
|---|---|---|---|
| [Cache volume](#the-cache-volume) | Downloads and compile caches | Yes | Volume storage |
| [Filesystem snapshot](#filesystem-snapshots) | The runtime's whole disk: packages installed live, files written outside the project | No | Snapshot storage |
| [Cell cache](#the-cell-cache) | Results of blocks you wrap in `mo.persistent_cache` | No | Volume storage |

None of them keeps what is in memory: the kernel starts empty. The cell
cache comes closest, by restoring the results you chose to cache. See
[what is not kept](#what-is-not-kept).

## Seeing what a start costs

`nbgpu start` prints the time of each step, and `--json` reports them:

```json
"timings": {"create": 1.75, "upload": 1.11, "ready": 1.05, "total": 4.23},
"from_snapshot": "im-01M7GHXR4HGRM4E0Y5WVE1GRZK"
```

`create` is provisioning the runtime and its environment (minutes when an
image is built, seconds otherwise), `upload` is pushing the project, `ready`
is marimo starting. The notebook's own setup (downloading weights, loading
data) comes after and is not included; time your first `nbgpu run` for that.

## The cache volume

Every Modal runtime mounts a volume named `nbgpu-cache` at `/nbgpu-cache`,
created the first time it is needed, and points the standard cache
locations into it:

| Variable | Used by |
|----------|---------|
| `HF_HOME` | Hugging Face models, datasets, tokenizers |
| `TORCH_HOME` | `torch.hub` weights |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `torch.compile` and Triton kernels |
| `UV_CACHE_DIR` | Package downloads, so live installs are faster |
| `NBGPU_CACHE` | Yours: the mount path, for anything else worth keeping |

Libraries read these variables themselves, so a notebook needs no change.
The second session reads `from_pretrained("...")` weights from the volume
instead of downloading them.

```python
import os, torch
path = os.path.join(os.environ.get("NBGPU_CACHE", "."), "features.pt")
if os.path.exists(path):
    features = torch.load(path)
else:
    features = expensive_preprocessing()
    torch.save(features, path)
```

Header control:

```toml
cache = false              # no cache volume
cache = "team-cache"       # use (and create) another volume
```

Things to know:

- Writes reach the volume when the runtime ends. Two sessions running at
  the same time each see the cache as it was when they started, and
  whichever ends last wins for a file both wrote. Cache files are written
  under content-derived names, so this is rarely visible.
- The volume grows without limit. Inspect and clear it with Modal's CLI:
  `modal volume ls nbgpu-cache`, `modal volume rm nbgpu-cache huggingface -r`.
  Modal bills volume storage per GiB-month (`nbgpu cost` shows it as a
  resource when it appears).
- `/nbgpu-cache` is reserved: a `volumes` entry there is an error unless
  `cache = false`.
- The local backend ignores the key; its disk already persists.

## Filesystem snapshots

A snapshot saves the runtime's disk when the session stops. The next start
of the same notebook boots from it instead of building the environment.

```bash
nbgpu stop train.py --snapshot
```

```text
✓ saved a snapshot of the runtime's filesystem (1s); the next `nbgpu start train.py` resumes from it (--fresh to build instead)
```

To snapshot at every stop, including the sync loop's idle stop, put it in
the header:

```toml
snapshot = true
```

`--no-snapshot` skips it for one stop.

What it keeps: packages installed live (by the sync loop, `run`, `sync` or
marimo's UI), anything installed with `pip` by hand, and files written
anywhere on the runtime's disk outside the project. What it leaves out:
mounted volumes (the cache included), the project directory and nbgpu's
own files (those are removed when the snapshot boots, so the upload and a
new access token are all there is).

### When a snapshot is used

The next `nbgpu start` uses the notebook's snapshot only if the snapshot
was built from the same:

- backend,
- dependencies (as installed, live installs included; order does not
  matter),
- `requires-python`, this machine's Python minor version, and marimo
  version.

Compute may differ: a snapshot taken on a T4 boots on an H100. Otherwise the
environment is built as usual and the snapshot stays for when it matches
again. Each notebook keeps only its latest snapshot, for up to 29 days.

```bash
nbgpu start train.py --fresh    # build the environment, ignore the snapshot
```

If the provider no longer has the image, start says so and builds instead.

### What it saves in practice

Measured on Modal with a small CPU notebook after adding one package live:
a fresh start rebuilt the image and took 14 s; the start from the snapshot
took 4 s. The gap grows with the environment: rebuilding one with torch
takes minutes. Taking the snapshot added about a second to `stop`, on CPU
and on a T4 alike.

## The cell cache

marimo can save the variables a block of code creates and restore them the
next time that block would run with the same code and inputs:

```python
with mo.persistent_cache(name="embeddings"):
    embeddings = encoder.encode(corpus)   # minutes the first time
    print("computed")                     # not printed on a cache hit
```

marimo keeps these entries in `__marimo__/cache` next to the notebook. On a
runtime that folder is thrown away at stop, so the cache only helps within
one session. With `cell-cache`, nbgpu turns that folder into a link to the
cache volume, so the entries survive:

```toml
[tool.nbgpu]
cell-cache = true
```

It is off by default and has to be asked for, because it changes what a
notebook does on its second run: a cached block is skipped, side effects
and all.

How it behaves:

- A block is skipped when its code and the values it reads from other
  cells are unchanged; edit either and it runs again. Add
  `pin_modules=True` to also re-run when package versions change.
- Each project and notebook directory gets its own folder on the volume
  (`/nbgpu-cache/marimo/<project id>/...`), so notebooks with the same name
  in different projects never share entries.
- It works in the browser and in `nbgpu run` alike. Measured on Modal: an
  8-second block took 0.3 s in the next session, and imports were as fast
  as without it.
- It needs the cache volume (`cell-cache = true` with `cache = false` is an
  error) and does nothing on the local backend.
- It applies from the next start; turning it on mid-session prints a note.

What to cache, and what not:

- **Do:** embeddings, preprocessed or tokenised datasets, evaluation
  results, a trained model's `state_dict()`: anything that takes minutes to
  compute and seconds to load.
- **Don't:** loading a pretrained model. Pickling a 7B model writes about
  14 GB, slower than `from_pretrained` reading the weights from the cache
  volume. Cache what you compute with the model instead.
- Move GPU tensors to CPU before caching (`.cpu()`) if the notebook may run
  on a runtime without a GPU: a pickled CUDA tensor only loads on a GPU.

The entries are ordinary files; clear them with
`modal volume rm nbgpu-cache marimo -r`.

## What is not kept

The kernel's memory: variables, a model already on the GPU, an open
dataloader. Every session starts its kernel from nothing and runs the
notebook's cells again; with the [cell cache](#the-cell-cache), the cached
blocks among them are restored rather than recomputed.

Modal has memory snapshots that restore running processes, but they do not
support GPU sandboxes (Modal refuses with "Memory snapshots do not support
GPU sandboxes"), and they only restore the sandbox's main process tree,
which marimo is not part of today. See the [roadmap](roadmap.md).

An earlier version of the cell cache moved Python's whole cache prefix
(`PYTHONPYCACHEPREFIX`) onto the volume. Bytecode read from the volume made
`import torch` take 210 s instead of a few, so only marimo's cache folder
is moved.
