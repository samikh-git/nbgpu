# marimo remote kernel shim

A prototype answering one question: can the stock marimo VS Code extension run
its kernel on another machine, with no fork and no patch?

`marimo_remote_kernel.py` is a fake Python interpreter. The extension's
language server (`marimo-lsp`) starts a kernel by running
`<python> -c <bootstrap> <notebook>`, writing kernel arguments to stdin, and
binding five ZeroMQ ports on `127.0.0.1` that the kernel connects back to. The
shim opens one SSH connection, reverse-forwards those ports through it,
rewrites the arguments to match, and relays stdio and interrupts. Every other
invocation (`--version`, `import marimo` probes) is forwarded to the remote
interpreter unchanged.

Standard library only; needs Python 3.11+ locally and `ssh` on `PATH`.

## Setup

```bash
python3 marimo_remote_kernel.py --mrk-install ~/.marimo-remote/gpu-box
```

This creates a directory shaped like a venv (`bin/python`, `pyvenv.cfg`) and a
`marimo-remote.toml` inside it. Edit the config:

```toml
host = "gpu-box"                                  # any ssh destination
python = "/home/me/project/.venv/bin/python"      # needs marimo + pyzmq
ssh_args = []

[paths]                                           # local prefix -> remote prefix
"/Users/me/project" = "/home/me/project"
```

SSH must work non-interactively (`ssh gpu-box true` with no prompt). Then pick
`~/.marimo-remote/gpu-box/bin/python` as the interpreter for the notebook.

The remote working directory is created if missing. Cell code travels over the
kernel queues, so the notebook file itself does not need to exist remotely;
local modules and data files the notebook reads do.

## Test

`tests/e2e.py` launches a kernel exactly as `marimo-lsp` does and checks cell
execution, dependent cells, interrupt, and teardown:

```bash
python tests/e2e.py <fake venv>/bin/python <local working dir>
```

Run it with an interpreter that has `marimo` and `pyzmq`. Set
`EXPECT_REMOTE_PYTHON` and `EXPECT_REMOTE_CWD` to also assert where the cell ran.

## Status

Verified with marimo 0.25.2 against an SSH server on the same machine (separate
interpreter, separate working directory, real reverse tunnels): launch, cell
output, reactive dependents, interrupt, and clean shutdown all work.

Not yet verified:

- **Inside VS Code.** The harness reproduces the language server's launch, not
  the extension. Whether the Python extension accepts the fake venv in its
  interpreter picker is untested.
- **A real remote host**, including latency and a dropped connection.
- **Hard kill.** If the shim is killed with SIGKILL, the remote kernel is
  expected to be left running; nothing reaps it yet.
- **Reconnect.** A dead tunnel ends the session; kernel state is lost.

## Things the kernel needs that are easy to miss

- It calls `os.setsid()`, which fails for a process-group leader. The command
  sshd starts is one, so the kernel must be a child of the remote shell, not
  `exec`ed in its place.
- `parent_pid` in the kernel arguments must be cleared; a local pid means
  nothing remotely.
- Interrupts are sent to the kernel's process group, and the pid reported on
  the `KERNEL_INFO` line must be a local one, because the caller signals it.
- This relies on `marimo._ipc.launch_kernel`, which marimo documents as a
  public entry point for `marimo-lsp`, and on the shape of `KernelArgs`, which
  is not a documented contract.
