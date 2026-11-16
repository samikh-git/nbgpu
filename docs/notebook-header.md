# Notebook header

A notebook declares what it needs in a [PEP 723](https://peps.python.org/pep-0723/)
`# /// script` block at the top of the file. nbgpu reads three things from it:
`dependencies`, `requires-python`, and the `[tool.nbgpu]` table.

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "torch"]
#
# [tool.nbgpu]
# provider = "modal"
# gpu = "A10G"
# cpu = 8
# memory = "32GiB"
# idle-timeout = "20m"
# max-runtime = "8h"
# volumes = { "/data" = "my-datasets" }
# exclude = ["data/", "*.ckpt"]
# requirements = "requirements.txt"
# cache = true
# snapshot = false
# cell-cache = false
# ///
```

Every key is optional. A notebook with no header at all runs on a CPU-only
Modal runtime with the defaults below.

An unknown key under `[tool.nbgpu]` is an error that names the key, so a typo
such as `gpus` is caught instead of silently ignored.

## `[tool.nbgpu]` keys

### `provider`

Which backend runs the notebook: `"modal"` (default) or `"local"`. The
`--backend` flag and the `NBGPU_BACKEND` environment variable both override
it, in that order.

### `gpu`

The GPU type, optionally with a count: `"A10G"`, `"H100:2"`. Omit for a
CPU-only runtime. Names are case-insensitive.

On Modal:

| Name | Max count | Notes |
|------|-----------|-------|
| `T4`, `L4` | 8 | |
| `A10`, `A10G` | 4 | `A10G` is the older name for `A10` |
| `L40S` | 8 | |
| `A100`, `A100-40GB`, `A100-80GB` | 8 | `A100` requests 40 GB |
| `RTX-PRO-6000` | 8 | |
| `H100`, `H100!` | 8 | `H100!` disables Modal's automatic upgrade to H200 |
| `H200` | 8 | |
| `B200`, `B200+`, `B300` | 8 | |

An unknown name is rejected before anything is created, with the list of
known ones.

### `cpu`

Number of CPU cores, as a positive number (`8`, `0.5`). On Modal this is
physical cores. Omit for the provider's default.

### `memory`

`"32GiB"`, `"512MiB"`, `"32GB"`, `"500MB"`, or a bare integer meaning MiB.
Decimal units are converted to MiB and rounded up. Omit for the provider's
default.

### `idle-timeout`

How long the session may sit unused before it is stopped. Default `"20m"`.

Durations are `"90s"`, `"20m"`, `"8h"`, `"1h30m"`, or a bare integer meaning
seconds. Days are not accepted.

"Unused" means no file changes in the project, no running or queued cell, no
CPU use and no GPU use. An open browser tab alone does not count as activity.
See [Sessions, idle and cost](sessions-and-cost.md#idle-shutdown).

### `max-runtime`

Hard upper bound on the runtime's life, enforced by the provider. Default
`"8h"`. Modal allows at most 24 hours.

### `volumes`

A table of absolute mount path to provider volume name:

```toml
volumes = { "/data" = "my-datasets", "/checkpoints" = "run-42" }
```

The volume must already exist; nbgpu does not create it
(`modal volume create my-datasets`). Use volumes for datasets, checkpoints
and anything else too large or too binary to sync.

### `exclude`

gitignore-style patterns, relative to the project root, for files that should
be neither uploaded nor pulled even though git tracks them:

```toml
exclude = ["data/", "*.ckpt", "secrets.toml"]
```

`.gitignore` files are always respected; `exclude` is for what git tracks
but the runtime should not get. See [Sync and conflicts](sync.md#what-is-left-out).

### `requirements`

One path or a list of paths to requirements files, relative to the
notebook's directory. Their packages are installed together with
`dependencies`:

```toml
requirements = ["requirements.txt", "requirements-gpu.txt"]
```

Only package lines and `-r` includes are supported. Any other option
(`-e`, `--index-url`, ...) is an error naming the file and line, rather than
being silently dropped.

### `cache`

A volume for downloads and compile caches, kept between sessions and
mounted at `/nbgpu-cache`. `HF_HOME`, `TORCH_HOME`, the `torch.compile` and
Triton caches and uv's cache point into it. Default `true`, which uses a
volume named `nbgpu-cache`, created on first use.

```toml
cache = false          # off
cache = "team-cache"   # another volume (created if missing)
```

See [Warm starts](warm-starts.md#the-cache-volume).

### `snapshot`

`true` saves the runtime's filesystem at every stop, so the next start boots
from it instead of building the environment (packages installed live
included). Default `false`; `nbgpu stop --snapshot` does it for one stop.
See [Warm starts](warm-starts.md#filesystem-snapshots).

### `cell-cache`

`true` keeps marimo's own cache (blocks wrapped in `mo.persistent_cache`)
on the cache volume, so cached blocks are restored in later sessions instead
of recomputed. Default `false`: a cached block is skipped on later runs,
which you should choose knowingly. Needs `cache`. See
[Warm starts](warm-starts.md#the-cell-cache).

## `dependencies`

Installed into the runtime's image with uv when the session starts.

- List `marimo` without a version. nbgpu pins it to the version installed on
  your machine, so the frontend your browser loads matches the server. If you
  give it a version yourself, yours is used.
- marimo is added if you leave it out.
- While a session runs, additions are installed live. See
  [Environments](environments.md).

## `requires-python`

Selects the runtime's Python. Your local minor version is used when it
satisfies the constraint (or none is given); otherwise the highest supported
version that does. Modal supports 3.11 through 3.14.

## Overriding from the command line

`nbgpu start` accepts `--gpu`, `--cpu`, `--memory`, `--idle-timeout` and
`--max-runtime`. They take the same values as the header, win over it, and
go through the same validation. Use them to try a bigger GPU for one session
without touching the file:

```bash
nbgpu start train.py --gpu H100 --max-runtime 2h
```

## What changes take effect when

| Change | In a running session |
|--------|----------------------|
| Add a dependency | Installed live by the sync loop, or by the next `nbgpu sync` or `nbgpu run` |
| Upgrade a dependency | Installed the same way; restart the kernel to load it in the browser |
| Remove a dependency | Stays installed until the next start |
| `gpu`, `cpu`, `memory`, `volumes`, `requires-python` | Next start |
| `exclude` | Next start |
| `cache` | Next start |
| `snapshot` | Read at stop |
| `cell-cache` | Next start |
