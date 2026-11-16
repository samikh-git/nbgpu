"""nbgpu.kernel's parsing and nbgpu.kernel_runner's code evaluation, without marimo running."""

from __future__ import annotations

import ast
import asyncio
import io
import json
from contextlib import redirect_stderr, redirect_stdout

import pytest

from nbgpu import kernel, kernel_runner
from nbgpu.kernel_runner import RESULT_MARK


def result_line(**fields: object) -> str:
    base = {"ok": True, "value": None, "error": None, "cells": []}
    return RESULT_MARK + json.dumps({**base, **fields}) + "\n"


def test_parse_result_splits_the_result_from_the_output() -> None:
    out = kernel.parse_result("hello\n" + result_line(value="3"), "")
    assert out.ok and out.value == "3" and out.stdout == "hello\n" and not out.pending


def test_result_after_output_without_a_newline_is_still_found() -> None:
    # print("x", end="") then the runner's result line: same line.
    out = kernel.parse_result("partial" + result_line(value="1"), "")
    assert out.value == "1" and out.stdout == "partial"


def test_the_last_result_wins_and_marks_never_leak() -> None:
    text = result_line(value="first") + "between\n" + result_line(value="second")
    stripped, result = kernel.strip_results(text)
    assert stripped == "between\n" and result is not None and result["value"] == "second"
    # The record separator in the mark is a line break for str.splitlines;
    # parsing must not depend on that.
    assert "\x1e" in RESULT_MARK


def test_no_result_line_is_a_failure_with_the_last_stderr_line() -> None:
    out = kernel.parse_result("", "Traceback...\nModuleNotFoundError: No module named 'marimo._code_mode'\n")
    assert not out.ok and out.error == "ModuleNotFoundError: No module named 'marimo._code_mode'"


def test_pending_and_first_run_fields() -> None:
    out = kernel.parse_result(result_line(ok=False, pending=True, ran_notebook=True, notebook_error="cell a: boom"), "")
    assert out.pending and out.ran_notebook and out.notebook_error == "cell a: boom"


def test_iter_sse_reads_named_events() -> None:
    stream = [
        b"event: stdout\n", b'data: {"data": "hi\\n"}\n', b"\n",
        b": a comment\n",
        b"event: done\n", b'data: {"success": true}\n', b"\n",
    ]
    assert list(kernel.iter_sse(stream)) == [("stdout", '{"data": "hi\\n"}'), ("done", '{"success": true}')]


def test_build_code_is_the_runner_plus_one_call() -> None:
    code = kernel.build_code({"mode": "code", "code": "print('a \"quoted\" value')"})
    tree = ast.parse(code)  # parses as a module: top-level await is allowed in the scratchpad
    assert code.startswith(kernel.runner_source())
    last = code.strip().splitlines()[-1]
    assert last.startswith("await _nbgpu_main(")
    params = json.loads(ast.literal_eval(last[len("await _nbgpu_main("):-1]))
    assert params == {"mode": "code", "code": "print('a \"quoted\" value')"}
    assert tree.body


def test_runner_imports_nothing_from_nbgpu() -> None:
    tree = ast.parse(kernel.runner_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("nbgpu"), node.module
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("nbgpu") for a in node.names)


def run_code(code: str) -> tuple[dict[str, object], str, str]:
    result = {"ok": True, "value": None, "error": None, "cells": []}
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        asyncio.run(kernel_runner._nbgpu_run_code(code, result))
    return result, out.getvalue(), err.getvalue()


def test_the_last_expression_is_the_value() -> None:
    result, out, _ = run_code("a = 6\nprint('side')\na * 7")
    assert result == {"ok": True, "value": "42", "error": None, "cells": []}
    assert out == "side\n"


def test_no_value_for_a_statement_or_none() -> None:
    assert run_code("b = 1")[0]["value"] is None
    assert run_code("None")[0]["value"] is None


def test_top_level_await() -> None:
    result, _, _ = run_code("import asyncio\nawait asyncio.sleep(0)\nawait asyncio.sleep(0, result='slept')")
    assert result["value"] == "'slept'"


def test_errors_keep_the_code_s_own_line_numbers() -> None:
    result, _, err = run_code("x = 1\ny = 2\nraise ValueError('third line')")
    assert result["ok"] is False and result["error"] == "ValueError: third line"
    assert 'File "<nbgpu exec>", line 3' in err
    assert "raise ValueError('third line')" in err  # the source line is shown
    assert "kernel_runner" not in err  # the runner's own frames are left out


def test_syntax_errors_are_reported_without_running() -> None:
    result, _, err = run_code("print('ran')\ndef broken(:\n")
    assert result["ok"] is False and result["error"].startswith("SyntaxError")
    assert "line 2" in result["error"] and "ran" not in err


def test_long_values_are_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernel_runner, "_NBGPU_MAX_VALUE_CHARS", 10)
    result, _, _ = run_code("'x' * 100")
    assert result["value"].startswith("'xxxxxxxxx") and "more characters" in result["value"]


def test_main_prints_one_result_line_and_touches_the_activity_file(tmp_path) -> None:
    activity = tmp_path / "nbgpu-activity"
    out = io.StringIO()
    with redirect_stdout(out):
        asyncio.run(kernel_runner._nbgpu_main(json.dumps({"mode": "code", "code": "1 + 1", "activity": str(activity)})))
    parsed = kernel.parse_result(out.getvalue(), "")
    assert parsed.ok and parsed.value == "2"
    assert activity.exists()
