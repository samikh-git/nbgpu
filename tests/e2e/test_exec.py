"""End-to-end: `nbgpu exec` against real marimo kernels on the local backend.

What these tests rely on about marimo 0.25.2 (probed 2026-11-15; see
nbgpu.kernel): headless `marimo edit` has no kernel until a client connects;
POST /api/kernel/execute runs code in the scratchpad of a session named by
`Marimo-Stable-Session-Id`; `marimo._code_mode` runs cells; `--watch`
reloads a changed notebook between kernel requests.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from nbgpu import kernel, lifecycle
from nbgpu.kernel import KernelClient
from tests.e2e.test_local_session import cli_cleanup, json_out, nbgpu  # noqa: F401 - cli_cleanup is a fixture

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(300)]

NAMED_NOTEBOOK = '''\
# /// script
# [tool.nbgpu]
# provider = "local"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def load():
    state = {"loads": 0, "steps": 0}
    state["loads"] += 1
    print("loading")
    return (state,)


@app.cell
def step(state):
    state["steps"] += 1
    factor = 2
    return (factor,)


if __name__ == "__main__":
    app.run()
'''


def ex(session, backend, store, tmp_path: Path, **kw):
    return lifecycle.exec_in_kernel(session, backend=backend, store=store, output_dir=tmp_path / "runs", **kw)


@pytest.fixture
def named(start_session, project: Path):
    path = project / "named.py"
    path.write_text(NAMED_NOTEBOOK)
    return start_session(path)


def test_first_exec_starts_a_kernel_and_runs_the_notebook(session, local_backend, store, tmp_path: Path) -> None:
    client = KernelClient(session.url, session.token)
    assert client.stable_session_id() is None  # headless marimo: no kernel yet

    first = ex(session, local_backend, store, tmp_path, code="x + y")
    assert first.ok, first
    assert first.value == "3"
    assert first.kernel_created and first.ran_notebook
    assert "hello-from-notebook" in first.output

    second = ex(session, local_backend, store, tmp_path, code="x * 10")
    assert second.ok and second.value == "10"
    assert not second.kernel_created and not second.ran_notebook
    assert "hello-from-notebook" not in second.output  # nothing re-ran


def test_names_are_scratch_but_objects_are_live(named, local_backend, store, tmp_path: Path) -> None:
    assert ex(named, local_backend, store, tmp_path, code="state").value == "{'loads': 1, 'steps': 1}"
    assert ex(named, local_backend, store, tmp_path, code="scratch = 5").ok
    gone = ex(named, local_backend, store, tmp_path, code="scratch")
    assert not gone.ok and gone.error == "NameError: name 'scratch' is not defined"
    # In-place changes stay: the same object lives in the kernel.
    assert ex(named, local_backend, store, tmp_path, code="state['poked'] = True").ok
    assert ex(named, local_backend, store, tmp_path, code="state.get('poked')").value == "True"


def test_errors_and_output(session, local_backend, store, tmp_path: Path) -> None:
    failed = ex(session, local_backend, store, tmp_path, code="print('before')\n1 / 0")
    assert not failed.ok and failed.error == "ZeroDivisionError: division by zero"
    assert "before" in failed.output
    assert 'File "<nbgpu exec>", line 2' in failed.output


def test_long_output_is_bounded(session, local_backend, store, tmp_path: Path) -> None:
    result = ex(session, local_backend, store, tmp_path, code="for i in range(500):\n    print(f'row-{i}')",
                max_output_lines=20)
    assert result.ok and result.truncated
    assert len(result.output.splitlines()) == 20
    assert result.full_output_path is not None and "row-499" in result.full_output_path.read_text()


def test_list_and_run_cells_by_name(named, local_backend, store, tmp_path: Path) -> None:
    listed = ex(named, local_backend, store, tmp_path, list_cells=True)
    assert [c["name"] for c in listed.cells] == ["load", "step"]
    assert not listed.ran_notebook  # listing never runs anything

    ran = ex(named, local_backend, store, tmp_path, cells=["step"])
    assert ran.ok and ran.ran_notebook  # first time: the whole notebook
    again = ex(named, local_backend, store, tmp_path, cells=["step"])
    assert again.ok and not again.ran_notebook
    assert "loading" not in again.output  # only `step` ran, not the cell above it
    assert ex(named, local_backend, store, tmp_path, code="state").value == "{'loads': 1, 'steps': 2}"

    missing = ex(named, local_backend, store, tmp_path, cells=["nope"])
    assert not missing.ok and "no cell named 'nope'" in missing.error and "load, step" in missing.error


def test_an_edited_cell_runs_with_the_new_code(named, local_backend, store, project: Path, tmp_path: Path) -> None:
    assert ex(named, local_backend, store, tmp_path, cells=["step"]).ok
    path = project / "named.py"
    path.write_text(path.read_text().replace("factor = 2", "factor = 7  # edited"))

    ran = ex(named, local_backend, store, tmp_path, cells=["step"])  # syncs first, waits for the reload
    assert ran.ok, ran
    assert "named.py" in ran.pushed
    assert ex(named, local_backend, store, tmp_path, code="factor").value == "7"
    assert ex(named, local_backend, store, tmp_path, code="state['loads']").value == "1"  # not reloaded


def test_a_failing_cell_fails_the_exec(named, local_backend, store, project: Path, tmp_path: Path) -> None:
    assert ex(named, local_backend, store, tmp_path, cells=["step"]).ok
    path = project / "named.py"
    path.write_text(path.read_text().replace("factor = 2", "factor = 2 / 0"))
    ran = ex(named, local_backend, store, tmp_path, cells=["step"])
    assert not ran.ok
    assert ran.error == "cell step: ZeroDivisionError: division by zero"
    assert [c["name"] for c in ran.cells] == ["step"]
    assert "ZeroDivisionError" in ran.output  # marimo's traceback


def test_exec_marks_the_session_active(session, local_backend, store, tmp_path: Path) -> None:
    handle = local_backend.get(session.runtime_id)
    activity = lifecycle.runtime_path(handle, lifecycle.ACTIVITY_NAME)
    before = time.time()
    assert ex(session, local_backend, store, tmp_path, code="1").ok
    assert os.stat(activity).st_mtime >= before - 1


async def _browser_kernel_ready(url: str, token: str) -> dict:
    import json as _json
    import urllib.parse

    import aiohttp

    parts = urllib.parse.urlsplit(url)
    ws_url = f"ws://{parts.netloc}/ws?session_id=browser-tab&access_token={urllib.parse.quote(token)}"
    async with aiohttp.ClientSession() as http, http.ws_connect(ws_url, max_msg_size=0) as ws:
        async for msg in ws:
            data = _json.loads(msg.data)
            if data.get("op") == "kernel-ready":
                return data["data"]
    raise AssertionError("no kernel-ready")


def test_a_browser_opened_later_is_the_editor(session, local_backend, store, tmp_path: Path) -> None:
    import asyncio

    # exec leaves no client connected, so a browser that connects later
    # resumes the kernel as its editor rather than as a read-only viewer.
    assert ex(session, local_backend, store, tmp_path, code="x").value == "1"
    ready = asyncio.run(_browser_kernel_ready(session.url, session.token))
    assert ready["kiosk"] is False and ready["resumed"] is True
    assert ex(session, local_backend, store, tmp_path, code="y").value == "2"


def test_cli_exec(project: Path, cli_cleanup) -> None:
    json_out(nbgpu("start", "notebook.py", "--backend", "local", "--detach", "--json", cwd=project))

    data = json_out(nbgpu("exec", "notebook.py", "x + y", "--json", cwd=project))
    assert data["ok"] is True and data["value"] == "3" and data["ran_notebook"] is True
    assert {"output", "truncated", "full_output_path", "cells", "kernel_created", "pushed", "warnings",
            "env", "error"} <= set(data)

    proc = nbgpu("exec", "notebook.py", "1/0", cwd=project)
    assert proc.returncode == 1
    assert "ZeroDivisionError" in proc.stdout  # the traceback is output
    assert proc.stderr.strip().splitlines()[-1].endswith("ZeroDivisionError: division by zero")

    proc = nbgpu("exec", "notebook.py", "--list", cwd=project)
    assert proc.returncode == 0 and "x = 1" in proc.stdout

    (project / "probe.py").write_text("z = x + 41\nz\n")
    proc = nbgpu("exec", "notebook.py", "-f", "probe.py", cwd=project)
    assert proc.returncode == 0 and proc.stdout.strip() == "42"

    proc = nbgpu("exec", "notebook.py", cwd=project)
    assert proc.returncode != 0 and "exactly one of" in proc.stderr


def test_cli_exec_without_a_session(project: Path, cli_cleanup) -> None:
    proc = nbgpu("exec", "notebook.py", "1", cwd=project)
    assert proc.returncode != 0 and "no session" in proc.stderr and "Traceback" not in proc.stderr


def test_kernel_client_reports_a_missing_server() -> None:
    client = KernelClient("http://127.0.0.1:9", "token")
    with pytest.raises(kernel.KernelError, match="could not reach marimo"):
        client.sessions()
