"""Runs inside a live marimo kernel, sent as code by `nbgpu exec`.

nbgpu.kernel sends this file's source to marimo's scratchpad
(`POST /api/kernel/execute`), followed by one line,
`await _nbgpu_main(<params as a JSON string>)`. It must import nothing from
nbgpu: only the standard library, and marimo itself (`marimo._code_mode`,
`marimo._ast.parse`), both private APIs checked against marimo 0.25.

The scratchpad runs with a shallow copy of the notebook's globals: names
bound here are discarded after the call, objects mutated in place stay
mutated. Every name this file defines starts with `_nbgpu` so it never
shadows a notebook variable the code reads.

Modes (params["mode"]):
- "code": run params["code"] as a module named `<nbgpu exec>` in the
  scratchpad's globals (top-level `await` allowed). If the last statement is
  an expression, its value's `repr()` is the result. Line numbers in
  tracebacks are the code's own; frames of this file are left out.
- "cells": run the cells in params["cells"] (names or ids), or every cell
  when it is "all", through code mode (`ctx.run_cell`; marimo then runs
  their stale dependents too). With params["wait_for"] (the notebook's path
  on the runtime), run nothing unless the kernel's cells match that file,
  and answer {"pending": true} instead: nbgpu just pushed it, and marimo's
  file watcher loads it a fraction of a second later. The kernel applies
  that reload only between requests, so the caller retries; waiting here
  would block the reload.
- "list": the notebook's cells, in order.

With params["prime"], "code" and "cells" first run every cell if no cell
has run in this kernel yet (marimo starts a kernel with every cell stale),
so the notebook's variables exist; the result then has "ran_notebook" and,
if cells failed, "notebook_error".

Results go to stdout as one line, `RESULT_MARK + json` (the last such line
wins): {"ok", "value", "error", "cells", "ran_notebook", "notebook_error"},
plus "pending" (see "cells"). "cells" is a list of {"name",
"id", "status", "errors", "first_line"}; for "cells" mode it holds the
cells left in an error state. Everything else the code or the cells print
is streamed as usual.

While it runs it touches params["activity"] (when set) every
ACTIVITY_EVERY_S, so the idle watchdog counts an exec, even one that
waits without using CPU, as activity.
"""

RESULT_MARK = "\x1enbgpu-result:"
ACTIVITY_EVERY_S = 20.0
_NBGPU_FILENAME = "<nbgpu exec>"
_NBGPU_ERROR_STATES = ("exception", "marimo-error", "cancelled", "interrupted")
_NBGPU_MAX_VALUE_CHARS = 200_000


async def _nbgpu_main(params_json):  # noqa: C901 - one entry point, kept flat on purpose
    import json
    import threading

    params = json.loads(params_json)
    done = threading.Event()
    toucher = None
    if params.get("activity"):
        toucher = threading.Thread(target=_nbgpu_touch_loop, args=(params["activity"], done), daemon=True)
        toucher.start()
    result = {"ok": True, "value": None, "error": None, "cells": [], "ran_notebook": False, "notebook_error": None}
    try:
        mode = params["mode"]
        if mode == "code":
            if params.get("prime"):
                await _nbgpu_prime(result)
            await _nbgpu_run_code(params["code"], result)
        elif mode == "list":
            result["cells"] = await _nbgpu_cells_info(None)
        elif mode == "cells":
            await _nbgpu_run_cells(params, result)
        else:
            raise ValueError(f"unknown mode {mode!r}")
    except _NbgpuError as exc:
        result["ok"] = False
        result["error"] = str(exc)
    finally:
        done.set()
        if toucher is not None:
            toucher.join(1)
        _nbgpu_touch(params.get("activity"))
    print(RESULT_MARK + json.dumps(result), flush=True)


class _NbgpuError(Exception):
    """A failure reported as the result's error, without a traceback."""


def _nbgpu_touch(path):
    if not path:
        return
    import os

    try:
        with open(path, "a", encoding="utf-8"):
            pass
        os.utime(path, None)
    except OSError:
        pass


def _nbgpu_touch_loop(path, done):
    _nbgpu_touch(path)
    while not done.wait(ACTIVITY_EVERY_S):
        _nbgpu_touch(path)


async def _nbgpu_run_code(code, result):
    import ast
    import inspect
    import linecache
    import sys
    import traceback

    linecache.cache[_NBGPU_FILENAME] = (len(code), None, code.splitlines(True), _NBGPU_FILENAME)
    flags = ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
    namespace = globals()
    try:
        tree = ast.parse(code, _NBGPU_FILENAME, "exec")
    except SyntaxError as exc:
        traceback.print_exception(type(exc), exc, None, file=sys.stderr)
        result["ok"] = False
        result["error"] = f"SyntaxError: {exc.msg} (line {exc.lineno})"
        return
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    try:
        outcome = eval(compile(tree, _NBGPU_FILENAME, "exec", flags=flags), namespace)
        if inspect.iscoroutine(outcome):
            await outcome
        if last is not None:
            expr = ast.Expression(last.value)
            ast.copy_location(expr, last)
            value = eval(compile(expr, _NBGPU_FILENAME, "eval", flags=flags), namespace)
            if inspect.iscoroutine(value):
                value = await value
            if value is not None:
                text = repr(value)
                if len(text) > _NBGPU_MAX_VALUE_CHARS:
                    text = text[:_NBGPU_MAX_VALUE_CHARS] + f"... ({len(text) - _NBGPU_MAX_VALUE_CHARS} more characters)"
                result["value"] = text
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        tb = exc.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename != _NBGPU_FILENAME:
            tb = tb.tb_next
        traceback.print_exception(type(exc), exc, tb, file=sys.stderr)
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"


def _nbgpu_cell_info(cell):
    code = cell.code or ""
    first = next((line for line in code.splitlines() if line.strip()), "")
    status = cell.status
    return {
        "name": cell.name,
        "id": str(cell.id),
        "status": None if status is None else str(status),
        "errors": [err.msg for err in cell.errors],
        "first_line": first[:120],
    }


async def _nbgpu_cells_info(only_errors):
    import marimo._code_mode as cm

    async with cm.get_context() as ctx:
        cells = [_nbgpu_cell_info(c) for c in ctx.cells]
    if only_errors:
        cells = [c for c in cells if c["status"] in _NBGPU_ERROR_STATES]
    return cells


async def _nbgpu_kernel_codes():
    import marimo._code_mode as cm

    async with cm.get_context() as ctx:
        return sorted(c.code or "" for c in ctx.cells)


def _nbgpu_file_codes(path):
    from marimo._ast.parse import parse_notebook

    with open(path, encoding="utf-8") as f:
        notebook = parse_notebook(f.read(), filepath=path)
    if notebook is None:
        return None
    return sorted(c.code or "" for c in notebook.cells)


async def _nbgpu_matches_file(path):
    try:
        expected = _nbgpu_file_codes(path)
    except (OSError, SyntaxError):
        return False
    return expected is not None and expected == await _nbgpu_kernel_codes()


def _nbgpu_has_run(cell):
    """Whether the cell has run in this kernel (successfully or not)."""
    impl = getattr(cell, "_impl", None)  # private; fall back to the public status
    if impl is not None and hasattr(impl, "run_result_status"):
        return impl.run_result_status is not None
    return cell.status not in (None, "stale")


async def _nbgpu_prime(result):
    """Run every cell if none has run yet (a kernel nbgpu just started, or
    one only listed so far), so the notebook's variables exist."""
    import marimo._code_mode as cm

    async with cm.get_context() as ctx:
        code_cells = [c for c in ctx.cells if (c.code or "").strip()]
        if not code_cells or any(_nbgpu_has_run(c) for c in code_cells):
            return
        for c in code_cells:
            ctx.run_cell(str(c.id))
    result["ran_notebook"] = True
    # The scratchpad's namespace was copied before those cells ran.
    async with cm.get_context() as ctx:
        fresh = {k: v for k, v in ctx.globals.items() if not k.startswith("_nbgpu")}
    globals().update(fresh)
    failed = await _nbgpu_cells_info(True)
    if failed:
        result["notebook_error"] = "; ".join(
            f"cell {c['name'] or c['id']}: {c['errors'][0] if c['errors'] else c['status']}" for c in failed
        )


def _nbgpu_targets(ctx, wanted):
    """Cell ids for `wanted` (names or ids, or "all")."""
    ids = [str(c.id) for c in ctx.cells]
    if wanted == "all":
        return [str(c.id) for c in ctx.cells if (c.code or "").strip()]
    by_name = {}
    for c in ctx.cells:
        by_name.setdefault(c.name, []).append(str(c.id))
    targets = []
    for want in wanted:
        if want in ids:
            targets.append(want)
        elif len(by_name.get(want, [])) == 1:
            targets.append(by_name[want][0])
        elif want in by_name:
            raise _NbgpuError(f"{len(by_name[want])} cells are named {want!r}; use a cell id")
        else:
            names = sorted(n for n in by_name if n and n != "_")
            hint = f"; named cells: {', '.join(names)}" if names else "; the notebook has no named cells"
            raise _NbgpuError(f"no cell named {want!r}{hint} (`nbgpu exec NOTEBOOK --list` shows ids)")
    return targets


async def _nbgpu_run_cells(params, result):
    import marimo._code_mode as cm

    # The reload is applied by the kernel itself, after this call returns,
    # so waiting here would wait forever: report and let the caller retry.
    if params.get("wait_for") and not await _nbgpu_matches_file(params["wait_for"]):
        result["ok"] = False
        result["pending"] = True
        result["error"] = "the kernel has not loaded the edited notebook yet"
        return
    async with cm.get_context() as ctx:
        targets = _nbgpu_targets(ctx, params["cells"])  # before priming: a bad name runs nothing
    if params.get("prime"):
        await _nbgpu_prime(result)
    async with cm.get_context() as ctx:
        before = {str(c.id): str(c.status) for c in ctx.cells}
        if not result["ran_notebook"]:  # the first run just ran them with the current code
            for target in targets:
                ctx.run_cell(target)
    # Cells that were already failing before this run and were not asked
    # for are still listed, but do not make the run fail.
    fresh = {i for i, status in before.items() if i in targets or status not in _NBGPU_ERROR_STATES}
    if result["ran_notebook"]:
        fresh = set(before)
    cells = await _nbgpu_cells_info(True)  # marimo streams their tracebacks to stderr itself
    new = [c for c in cells if c["id"] in fresh]
    result["cells"] = cells
    if new:
        result["ok"] = False
        result["error"] = "; ".join(
            f"cell {c['name'] or c['id']}: {c['errors'][0] if c['errors'] else c['status']}" for c in new
        )
