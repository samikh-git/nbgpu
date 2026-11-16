"""End-to-end check of the shim, driving a kernel the way marimo-lsp does.

Mirrors `launch_kernel` in marimo-lsp's `src/marimo_lsp/kernels/manager.py`:
bind the ZeroMQ queues locally, start `<executable> -c <bootstrap> <notebook>`
with the kernel arguments on stdin, then run cells over the queues.

Run with an interpreter that has marimo and pyzmq:

    python tests/e2e.py <shim python> <local working dir>

Prints one PASS/FAIL line per check and exits non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import time

import marimo._ipc as ipc
from marimo._ast.app_config import _AppConfig
from marimo._config.manager import get_default_config_manager
from marimo._runtime.commands import AppMetadata, ExecuteCellsCommand
from marimo._session.managers import IPCQueueManagerImpl

# Copied from marimo-lsp (kernels/manager.py) so the launch is byte-identical.
BOOTSTRAP = """\
import json, os, runpy, sys, marimo
try:
    from pathlib import Path
    from marimo._session.state.serialize import get_session_cache_file
    notebook = Path(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else None
    cache = (
        os.path.abspath(os.fspath(get_session_cache_file(notebook)))
        if notebook is not None and notebook.is_file()
        else None
    )
except Exception:
    cache = None
print(json.dumps({"marimo_version": marimo.__version__, "cache": cache}), flush=True)
sys.argv[:] = [sys.argv[0]]
runpy.run_module("marimo._ipc.launch_kernel", run_name="__main__")
"""

failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}", flush=True)


def wait_for(stream: "queue.Queue", needle: str, timeout: float) -> str | None:
    """Return the first kernel message containing `needle`, if any."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            message = stream.get(timeout=0.2)
        except queue.Empty:
            continue
        text = message.decode("utf-8", "replace") if isinstance(message, bytes) else str(message)
        if needle in text:
            return text
    return None


def main() -> int:
    shim, workdir = sys.argv[1], os.path.abspath(sys.argv[2])
    notebook = os.path.join(workdir, "notebook.py")

    queues, info = ipc.QueueManager.create()
    manager = IPCQueueManagerImpl.from_ipc(queues)
    stream = manager.stream_queue
    args = ipc.KernelArgs(
        connection_info=info,
        configs={},
        app_metadata=AppMetadata(
            query_params={}, filename=notebook, cli_args={}, argv=None,
            app_config=_AppConfig(),
        ),
        user_config=get_default_config_manager(current_path=None).get_config(
            hide_secrets=False
        ),
        log_level=30,
        profile_path=None,
    )

    started = time.monotonic()
    proc = subprocess.Popen(
        [shim, "-c", BOOTSTRAP, notebook],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=workdir,
    )
    proc.stdin.write(args.encode_json())
    proc.stdin.close()

    metadata = proc.stdout.readline().decode().strip()
    ready = proc.stdout.readline().decode().strip()
    check("metadata line is JSON with a marimo version", '"marimo_version"' in metadata, metadata)
    check("kernel reports KERNEL_READY", ready == "KERNEL_READY", repr(ready))
    if ready != "KERNEL_READY":
        proc.kill()
        return 1
    info_line = proc.stdout.readline().decode().strip()
    check(
        "KERNEL_INFO names the shim's pid, not a remote one",
        info_line.split()[:2] == ["KERNEL_INFO", str(proc.pid)],
        info_line,
    )
    print(f"      launch took {time.monotonic() - started:.1f}s", flush=True)

    # 1. Run a cell and see where it actually ran.
    manager.put_control_request(
        ExecuteCellsCommand(
            cell_ids=["probe"],
            codes=[
                "import json, os, sys\n"
                "print('PROBE' + json.dumps({'exe': sys.executable, 'cwd': os.getcwd(),"
                " 'pid': os.getpid(), 'file': __file__}))"
            ],
        )
    )
    message = wait_for(stream, "PROBE{", 30)
    check("cell output comes back over the tunnelled queues", message is not None)
    if message is None:
        proc.kill()
        return 1
    probe = {}
    if message:
        # The payload is JSON nested inside a JSON message; unescape then cut.
        raw = message.replace('\\"', '"').replace("\\\\", "\\")
        start = raw.index("PROBE{") + 5
        probe = json.JSONDecoder().raw_decode(raw[start:])[0]
        print(f"      kernel: {probe}", flush=True)
        expect_exe = os.environ.get("EXPECT_REMOTE_PYTHON")
        expect_cwd = os.environ.get("EXPECT_REMOTE_CWD")
        if expect_exe:
            check("cell ran in the remote interpreter", probe["exe"] == expect_exe, probe["exe"])
        if expect_cwd:
            check("working directory was mapped", os.path.realpath(probe["cwd"]) == os.path.realpath(expect_cwd), probe["cwd"])
            check("notebook __file__ was mapped", probe["file"].startswith(expect_cwd), probe["file"])

    # 2. Reactivity: a second cell that depends on the first.
    manager.put_control_request(
        ExecuteCellsCommand(cell_ids=["a", "b"], codes=["x = 20", "print('SUM', x + 22)"])
    )
    check("dependent cells run", wait_for(stream, "SUM 42", 30) is not None)

    # 3. Interrupt a long-running cell, signalling the pid marimo-lsp holds.
    manager.put_control_request(
        ExecuteCellsCommand(
            cell_ids=["slow"],
            codes=["import time\nprint('SLOW_STARTED')\ntime.sleep(120)\nprint('SLOW_FINISHED')"],
        )
    )
    check("long cell starts", wait_for(stream, "SLOW_STARTED", 30) is not None)
    sent = time.monotonic()
    os.kill(proc.pid, signal.SIGINT)
    interrupted = wait_for(stream, "nterrupt", 20)
    check(
        "interrupt reaches the remote kernel",
        interrupted is not None,
        f"{time.monotonic() - sent:.1f}s" if interrupted else "",
    )

    # 4. The kernel survives the interrupt.
    manager.put_control_request(
        ExecuteCellsCommand(cell_ids=["after"], codes=["print('STILL', 'ALIVE')"])
    )
    check("kernel still usable after interrupt", wait_for(stream, "STILL ALIVE", 30) is not None)

    # 5. Terminate the way marimo-lsp does and make sure nothing is left behind.
    proc.terminate()
    try:
        proc.wait(timeout=10)
        check("shim exits on SIGTERM", True)
    except subprocess.TimeoutExpired:
        check("shim exits on SIGTERM", False)
        proc.kill()
    if probe.get("pid"):
        time.sleep(1.5)
        alive = subprocess.run(["kill", "-0", str(probe["pid"])], capture_output=True).returncode == 0
        check("remote kernel process is gone", not alive, f"pid {probe['pid']}")

    manager.close_queues()
    print("ALL PASSED" if not failures else f"{failures} FAILED", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
