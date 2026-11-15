#!/usr/bin/env python3
"""A fake Python interpreter that runs a marimo kernel on another machine.

The marimo VS Code extension (marimo-lsp) starts a kernel by running

    <python> -c <bootstrap> <notebook path>

writing the kernel arguments as JSON to stdin and reading a few status lines
from stdout. The kernel then connects *back* to five ZeroMQ ports that the
language server has bound on 127.0.0.1.

Point the extension at this script instead of a real interpreter and it will:

  * forward every ordinary invocation (version probes, `import marimo` checks)
    to the remote interpreter over SSH, and
  * for a kernel launch, open one SSH connection, reverse-forward the five
    ports through it, rewrite the kernel arguments to match, and relay
    stdin/stdout and interrupts.

Nothing in marimo or the extension is patched. Standard library only.

Usage:
    marimo_remote_kernel.py --mrk-install <dir>   # create a fake venv
    <dir>/bin/python ...                          # behaves like remote python
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

KERNEL_MARKER = "marimo._ipc.launch_kernel"
PID_MARKER = "MRK_PID"
CONFIG_NAME = "marimo-remote.toml"
PORT_KEYS = ("control", "ui_element", "completion", "win32_interrupt", "input", "stream")

EXAMPLE_CONFIG = """\
# ssh destination (anything `ssh` accepts, including a Host alias)
host = "gpu-box"
# interpreter on the remote machine; it must have marimo and pyzmq installed
python = "/home/me/project/.venv/bin/python"
# extra arguments for every ssh call
ssh_args = []

# local path prefix -> remote path prefix (working directory, notebook path)
[paths]
# "/Users/me/project" = "/home/me/project"
"""


def log(message: str) -> None:
    print(f"[marimo-remote] {message}", file=sys.stderr, flush=True)


def die(message: str) -> "typing.NoReturn":  # noqa: F821
    log(message)
    sys.exit(1)


# -- configuration -----------------------------------------------------------


def find_config() -> Path:
    here = Path(os.path.abspath(__file__)).parent
    candidates = [
        Path(p) for p in [os.environ.get("MARIMO_REMOTE_CONFIG")] if p
    ] + [
        here / CONFIG_NAME,
        here.parent / CONFIG_NAME,  # <fake venv>/marimo-remote.toml
        Path.home() / ".config" / "marimo-remote" / "config.toml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    die("no config found; looked in: " + ", ".join(map(str, candidates)))


class Config:
    def __init__(self, path: Path) -> None:
        data = tomllib.loads(path.read_text())
        try:
            self.host: str = data["host"]
            self.python: str = data["python"]
        except KeyError as missing:
            die(f"{path}: missing required key {missing}")
        self.ssh_args: list[str] = list(data.get("ssh_args", []))
        # Longest prefix first so nested mappings win.
        self.paths = sorted(
            ((l.rstrip("/"), r.rstrip("/")) for l, r in data.get("paths", {}).items()),
            key=lambda pair: -len(pair[0]),
        )

    def ssh(self, *extra: str) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", *self.ssh_args, *extra]

    @staticmethod
    def _swap(path: str, pairs: list[tuple[str, str]]) -> str:
        for src, dst in pairs:
            if path == src or path.startswith(src + "/"):
                return dst + path[len(src) :]
        return path

    def to_remote(self, path: str) -> str:
        return self._swap(path, self.paths)

    def to_local(self, path: str) -> str:
        return self._swap(path, [(r, l) for l, r in self.paths])


# -- plain invocations -------------------------------------------------------


def passthrough(config: Config, argv: list[str]) -> "typing.NoReturn":  # noqa: F821
    """Run `python <argv>` remotely with our stdio; never returns."""
    remote = "exec " + shlex.join([config.python, *argv])
    os.execvp("ssh", config.ssh(config.host, remote))


# -- kernel launch -----------------------------------------------------------


def wait_for_master(config: Config, sock: str, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            die(f"ssh exited with code {proc.returncode} before connecting")
        check = subprocess.run(
            config.ssh("-S", sock, "-O", "check", config.host),
            capture_output=True,
        )
        if check.returncode == 0:
            return
        time.sleep(0.1)
    proc.kill()
    die("timed out waiting for the ssh connection")


def reverse_forward(config: Config, sock: str, local_port: int) -> int:
    """Ask the remote sshd for a free port that tunnels back to local_port."""
    result = subprocess.run(
        config.ssh(
            "-S", sock, "-O", "forward",
            "-R", f"127.0.0.1:0:127.0.0.1:{local_port}",
            config.host,
        ),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        die(f"could not forward port {local_port}: {result.stderr.strip()}")
    return int(result.stdout.strip())


def run_kernel(config: Config, argv: list[str]) -> int:
    # argv is ["-c", <bootstrap>, <notebook path or "">]
    kernel_args = json.loads(sys.stdin.buffer.read())

    notebook = config.to_remote(argv[2]) if len(argv) > 2 and argv[2] else ""
    remote_cwd = config.to_remote(os.getcwd())
    # The kernel reports its own pid before anything else, so interrupts work
    # on marimo versions that predate the KERNEL_INFO line.
    code = f"import os; print({PID_MARKER!r}, os.getpid(), flush=True)\n" + argv[1]
    # No `exec`, and a trailing command so the shell cannot exec implicitly:
    # the kernel calls os.setsid(), which fails for a process-group leader,
    # and the command sshd starts is one.
    script = (
        f"mkdir -p {shlex.quote(remote_cwd)} && cd {shlex.quote(remote_cwd)} && "
        f"{shlex.join([config.python, '-c', code, notebook])}; exit $?"
    )

    # Unix socket paths are limited to ~100 bytes; keep the directory short.
    tmp = tempfile.mkdtemp(prefix="mrk-", dir="/tmp")
    sock = os.path.join(tmp, "s")
    proc = subprocess.Popen(
        config.ssh(
            "-M", "-S", sock,
            "-o", "ControlPersist=no",
            "-o", "ServerAliveInterval=15",
            config.host,
            "sh -c " + shlex.quote(script),
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    assert proc.stdin and proc.stdout

    remote_pid: int | None = None

    def remote_kill(sig: str) -> None:
        if remote_pid is None or proc.poll() is not None:
            return
        # marimo's kernel leads its own process group so that an interrupt
        # also reaches subprocesses of user code; fall back to the pid alone.
        command = (
            f"kill -{sig} -- -{remote_pid} 2>/dev/null || kill -{sig} {remote_pid}"
        )
        subprocess.Popen(
            config.ssh("-S", sock, config.host, command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def on_interrupt(_signum: int, _frame: object) -> None:
        remote_kill("INT")

    def on_terminate(_signum: int, _frame: object) -> None:
        remote_kill("TERM")
        # Give the remote kill a moment to go out before the connection drops.
        time.sleep(0.3)
        proc.terminate()

    signal.signal(signal.SIGINT, on_interrupt)
    signal.signal(signal.SIGTERM, on_terminate)
    signal.signal(signal.SIGHUP, on_terminate)

    try:
        wait_for_master(config, sock, proc)

        # The kernel connects to 127.0.0.1:<port> on *its* machine, so give it
        # remote ports that tunnel back to the language server's local ones.
        info = kernel_args["connection_info"]
        for key in PORT_KEYS:
            if info.get(key) is not None:
                info[key] = reverse_forward(config, sock, info[key])

        # The parent pid is meaningless on another machine; a kernel that
        # watches it would shut itself down immediately.
        kernel_args["parent_pid"] = None
        metadata = kernel_args.get("app_metadata") or {}
        if metadata.get("filename"):
            metadata["filename"] = config.to_remote(metadata["filename"])

        proc.stdin.write(json.dumps(kernel_args).encode())
        proc.stdin.close()

        out = sys.stdout.buffer
        seen_metadata = False
        for line in proc.stdout:
            text = line.decode("utf-8", errors="replace").strip()
            if text.startswith(PID_MARKER + " "):
                remote_pid = int(text.split()[1])
                continue
            if not seen_metadata and text.startswith("{"):
                # {"marimo_version": ..., "cache": <remote path>}
                seen_metadata = True
                try:
                    meta = json.loads(text)
                    if isinstance(meta.get("cache"), str):
                        meta["cache"] = config.to_local(meta["cache"])
                    line = (json.dumps(meta) + "\n").encode()
                except json.JSONDecodeError:
                    pass
            elif text.startswith("KERNEL_INFO "):
                # Report ourselves: the caller signals this pid, and a remote
                # pid would name an unrelated local process.
                remote_pid = int(text.split()[1])
                line = f"KERNEL_INFO {os.getpid()} {sys.argv[0]}\n".encode()
            out.write(line)
            out.flush()
        return proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
        shutil.rmtree(tmp, ignore_errors=True)


# -- install -----------------------------------------------------------------


def install(target: str) -> int:
    """Create a directory that looks enough like a venv to be selectable."""
    root = Path(target).absolute()
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    source = Path(os.path.abspath(__file__)).read_text().splitlines(keepends=True)
    # Pin the shebang to a real interpreter: `env python3` could resolve back
    # to this shim once the fake venv is on PATH.
    source[0] = f"#!{sys.executable}\n"
    for name in ("python", "python3"):
        shim = bin_dir / name
        shim.write_text("".join(source))
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (root / "pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).parent}\ninclude-system-site-packages = false\n"
    )
    config = root / CONFIG_NAME
    if not config.exists():
        config.write_text(EXAMPLE_CONFIG)
    print(f"Installed. Edit {config}, then select {bin_dir / 'python'} as the interpreter.")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["--mrk-install"]:
        if len(argv) != 2:
            die("usage: marimo_remote_kernel.py --mrk-install <dir>")
        return install(argv[1])
    config = Config(find_config())
    if len(argv) >= 2 and argv[0] == "-c" and KERNEL_MARKER in argv[1]:
        return run_kernel(config, argv)
    passthrough(config, argv)


if __name__ == "__main__":
    sys.exit(main())
