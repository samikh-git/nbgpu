# nbgpu

Run a local [marimo](https://marimo.io) notebook on a remote GPU. The notebook
declares the compute it needs in its own header; `nbgpu start` provisions it,
syncs the project both ways, and serves the marimo UI on a local URL.

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "torch"]
#
# [tool.nbgpu]
# gpu = "A10G"
# ///
```

```bash
nbgpu start train.py     # opens marimo on an A10G; edits sync both ways
nbgpu run train.py       # or: run it as a script remotely and print the output
nbgpu stop train.py      # pull, terminate, stop paying
```

The notebook stays a plain Python file on your machine, in your git repo,
editable by you, your editor and your agents. Only the kernel is remote.

## Install

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), `git` (used for
three-way merges) and a [Modal](https://modal.com) account.

From a checkout:

```bash
uv tool install --editable .
```

Then log in to Modal once:

```bash
modal setup
```

## What it does

- **Compute in the header.** GPU, CPU, memory, volumes and timeouts live next
  to the dependencies in the notebook's PEP 723 block.
- **Two-way sync.** Local edits are pushed, edits made in the marimo UI are
  pulled, concurrent edits are merged three-way, and a real conflict leaves
  both versions on disk. Nothing is ever overwritten unseen.
- **Live environments.** Add a dependency to the header and it is installed
  in the running session; install one from marimo's UI and it is written
  back to the header.
- **Cost stays visible and bounded.** The hourly rate is printed at start,
  the running total in `status`, and an idle watchdog stops sessions nobody
  is using. Runtimes that lost their record are found and listed.
- **Built for agents.** Every command has `--json`, errors are one line, and
  `nbgpu run` turns a notebook into a remote script with an exit code.

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Install, first session, first run |
| [Notebook header](docs/notebook-header.md) | Every `[tool.nbgpu]` key |
| [CLI reference](docs/cli.md) | Commands, flags, JSON output, environment variables |
| [Sync and conflicts](docs/sync.md) | What moves, what does not, resolving conflicts |
| [Environments](docs/environments.md) | Dependencies, live installs, Python version |
| [Sessions, idle and cost](docs/sessions-and-cost.md) | Lifecycle, idle shutdown, orphans, pricing |
| [Using nbgpu with agents](docs/agents.md) | The agent loop and the bundled skills |
| [Architecture](docs/architecture.md) | How the pieces fit; adding a backend |
| [Troubleshooting](docs/troubleshooting.md) | Errors and what to do about them |

## Agent skills

Three skills ship in this repository:

| Skill | For |
|-------|-----|
| [`skills/nbgpu`](skills/nbgpu/SKILL.md) | Running notebooks remotely: modes, results, cost, conflicts |
| [`skills/nbgpu-notebook`](skills/nbgpu-notebook/SKILL.md) | Writing notebooks and headers that work remotely and in script mode |
| [`.claude/skills/nbgpu-dev`](.claude/skills/nbgpu-dev/SKILL.md) | Working on nbgpu itself |

They are active in this repository as-is. To use the first two in every
project, see [Using nbgpu with agents](docs/agents.md#installing-the-skills).

## Development

```bash
uv sync
uv run pytest                 # unit, contract (local backend) and e2e tests
NBGPU_MODAL_TESTS=1 uv run pytest -m modal   # creates real Modal sandboxes
```

The kernel-tunnel prototype for VS Code lives in [`shim/`](shim/README.md).
It is a separate experiment and not used by `nbgpu`.
