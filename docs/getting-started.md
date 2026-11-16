# Getting started

## Install

With Homebrew:

```bash
brew install samikh-git/tools/nbgpu
```

Or from a checkout of this repository, with
[uv](https://docs.astral.sh/uv/) and Python 3.11 or newer:

```bash
uv tool install --editable .
```

Either way you need `git` on your `PATH` (sync uses `git merge-file` for
three-way merges).

```bash
nbgpu --version
```

nbgpu runs on [Modal](https://modal.com) by default. Log in once:

```bash
modal setup
```

To try everything without an account first, use the [local
backend](#trying-it-without-a-cloud-account).

## A first notebook

Save this as `hello.py` inside a project directory (any directory with a
`pyproject.toml` or a `.git`):

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "torch"]
#
# [tool.nbgpu]
# gpu = "T4"
# idle-timeout = "10m"
# max-runtime = "1h"
# ///

import marimo

app = marimo.App()


@app.cell
def _():
    import torch
    print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
    return (torch,)


if __name__ == "__main__":
    app.run()
```

The comment block at the top is a standard [PEP 723](https://peps.python.org/pep-0723/)
script header. `dependencies` says what to install; `[tool.nbgpu]` says what
to run it on. See [Notebook header](notebook-header.md) for every key.

## Start a session

```bash
nbgpu start hello.py
```

```text
starting hello.py on modal...
    0.0s  creating the runtime (...)
    2.9s  uploading 12 files (...) from myproject/
    4.0s  starting marimo
    6.1s  ready
✓ marimo is running on modal (sb-...)
  open:   http://127.0.0.1:2718?access_token=...
  remote: https://....modal.host?access_token=...  (direct; slower to load, works without this terminal)
  rate:   $0.59/h  (estimate from your account's rates as of 2026-11-15)
  stop:   nbgpu stop hello.py   (the runtime keeps running and billing until then)
syncing every second; Ctrl-C to stop syncing (the runtime keeps running)
```

The first start with a new set of dependencies builds an image, which can
take a few minutes for something like torch. Later starts reuse it and take
seconds.

Open the `open:` link. That is marimo, running on the GPU, editing the file
on your laptop:

- Edit a cell in the browser and `hello.py` changes on disk a second later.
- Edit `hello.py` in your editor and the browser updates.
- Add `"numpy"` to `dependencies` and it is installed in the running session.

The terminal running `nbgpu start` is the sync loop. Leave it open.

## Stop

Ctrl-C in that terminal stops *syncing*. The runtime keeps running and
billing, so that you can close the laptop and come back. To come back:

```bash
nbgpu open hello.py
```

To finish:

```bash
nbgpu stop hello.py
```

This pulls any remaining edits, terminates the runtime and removes the
session record. If you forget, the idle watchdog stops the session after
`idle-timeout` without activity. See [Sessions, idle and cost](sessions-and-cost.md).

## Run as a script

```bash
nbgpu run hello.py
```

Runs `python hello.py` on the remote machine and prints its output, then
exits with the notebook's exit code. With no session running it creates a
runtime for the run and terminates it afterwards. With a session running it
runs there, after syncing your latest edits. Small text files the run
writes come back afterwards; `--pull 'plots/'` brings back others.

## See what is running

```bash
nbgpu status
```

Lists every session with its uptime, rate and cost so far, plus any runtime
that is running without a record. `nbgpu cost` shows what Modal actually
billed.

## Trying it without a cloud account

```bash
NBGPU_BACKEND=local nbgpu start notebook.py
```

The local backend runs the "remote" side on your own machine. It uses the
Python that nbgpu itself runs in rather than building an environment, and it
has no GPU and no volumes, so the notebook's header must not ask for either.
It is meant for trying the tool and for tests.

## Next

- [Notebook header](notebook-header.md): choosing GPUs, memory, volumes
- [Sync and conflicts](sync.md): what is uploaded and what comes back
- [Using nbgpu with agents](agents.md)
