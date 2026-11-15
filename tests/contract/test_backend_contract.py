"""The Backend contract (nbgpu/backends/base.py), run against every backend.

Each test runs for `local`, and for `modal` when NBGPU_MODAL_TESTS=1.
Modal runtimes here are CPU-only (gpu=None); the GPU cases only check
validation, which must fail before anything is provisioned.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

import pytest

from nbgpu.backends.base import (
    ABSENT,
    CAP_GPU,
    CAP_IDLE_TIMEOUT,
    CAP_MAX_RUNTIME,
    CAP_VOLUMES,
    Backend,
    BackendError,
    ExecResult,
    RemoteProcess,
    RuntimeHandle,
    RuntimeRequest,
    TEMP_PREFIX,
    UnsupportedError,
)
from tests.contract.conftest import EXPOSED_PORT, RuntimeFactory, make_spec, wait_for

PY = "python"


def collect_lines(proc: RemoteProcess, timeout_s: float) -> list[str]:
    """Drain proc.lines() in a thread so a hanging iterator fails the test
    instead of hanging it."""
    lines: list[str] = []
    errors: list[BaseException] = []

    def drain() -> None:
        try:
            for line in proc.lines():
                lines.append(line)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    thread.join(timeout_s)
    assert not thread.is_alive(), f"lines() did not end within {timeout_s}s; got {lines!r}"
    if errors:
        raise errors[0]
    return lines


def wait_exit(proc: RemoteProcess, timeout_s: float) -> int:
    code = wait_for(lambda: proc.poll() is not None, timeout_s, what="process exit")
    assert code
    result = proc.poll()
    assert result is not None
    return result


def py(code: str) -> list[str]:
    return [PY, "-c", code]


def remote_path(handle: RuntimeHandle, rel: str) -> str:
    return str(PurePosixPath(handle.remote_root) / rel)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# registry and capabilities


def test_get_backend_registry() -> None:
    from nbgpu.backends import get_backend
    from nbgpu.backends.local import LocalBackend

    assert isinstance(get_backend("local"), LocalBackend)
    with pytest.raises(BackendError) as info:
        get_backend("no-such-backend")
    message = str(info.value)
    assert "local" in message and "modal" in message


def test_name_and_capabilities(backend: Backend) -> None:
    assert isinstance(backend.capabilities, frozenset)
    if backend.name == "local":
        assert backend.capabilities == frozenset()
    else:
        assert backend.name == "modal"
        assert {CAP_GPU, CAP_VOLUMES, CAP_MAX_RUNTIME} <= backend.capabilities
        # Idle shutdown is the in-runtime watchdog's job, not Modal's.
        assert CAP_IDLE_TIMEOUT not in backend.capabilities


# --------------------------------------------------------------------------
# create / get / find


def test_create_gives_empty_remote_root_and_tags(backend: Backend, runtimes: RuntimeFactory) -> None:
    before = time.time()
    handle = runtimes(tags={"role": "create-test"})
    after = time.time()

    assert handle.backend == backend.name
    assert isinstance(handle.runtime_id, str) and handle.runtime_id
    assert PurePosixPath(handle.remote_root).is_absolute()
    assert handle.tags.get("role") == "create-test"
    assert handle.tags.get("nbgpu.test") == runtimes.run_tag
    assert before - 5 <= handle.created_at <= after + 5
    port = handle.port_for(EXPOSED_PORT)
    assert isinstance(port, int) and 0 < port < 65536

    # remote_root exists, is a directory and is empty.
    res = backend.exec(
        handle,
        py("import os, sys; print(os.path.isdir(sys.argv[1])); print(sorted(os.listdir(sys.argv[1])))")
        + [handle.remote_root],
    )
    assert res.exit_code == 0, res.stderr
    assert res.stdout.splitlines() == ["True", "[]"]
    assert backend.hash_files(handle) == {}


def test_get_returns_running_runtime(backend: Backend, handle: RuntimeHandle) -> None:
    got = backend.get(handle.runtime_id)
    assert got is not None
    assert got.runtime_id == handle.runtime_id
    assert got.backend == backend.name
    assert got.remote_root == handle.remote_root
    assert dict(got.tags).items() >= dict(handle.tags).items()
    assert got.port_for(EXPOSED_PORT) == handle.port_for(EXPOSED_PORT)


def test_get_unknown_runtime_is_none(backend: Backend) -> None:
    assert backend.get(f"nbgpu-no-such-runtime-{uuid.uuid4().hex[:8]}") is None


def test_find_matches_tag_subsets_and_drops_terminated(backend: Backend, runtimes: RuntimeFactory) -> None:
    group = uuid.uuid4().hex[:8]
    a = runtimes(tags={"group": group, "which": "a"})
    b = runtimes(tags={"group": group, "which": "b"})

    ids = lambda hs: sorted(h.runtime_id for h in hs)  # noqa: E731

    assert ids(backend.find({"group": group})) == sorted([a.runtime_id, b.runtime_id])
    assert ids(backend.find({"group": group, "which": "a"})) == [a.runtime_id]
    assert ids(backend.find({"group": group, "which": "c"})) == []
    assert ids(backend.find({"group": "other-" + group})) == []
    # A key the runtimes do not carry at all never matches.
    assert ids(backend.find({"group": group, "absent-key": "x"})) == []
    # Handles from find carry the full tags.
    found_a = backend.find({"group": group, "which": "a"})[0]
    assert found_a.tags.get("which") == "a"
    assert found_a.remote_root == a.remote_root

    backend.terminate(a)
    assert backend.get(a.runtime_id) is None
    assert ids(backend.find({"group": group})) == [b.runtime_id]


# --------------------------------------------------------------------------
# exec


def test_exec_returns_exit_code_stdout_and_stderr(backend: Backend, handle: RuntimeHandle) -> None:
    res = backend.exec(
        handle,
        py("import sys; print('to-stdout'); print('to-stderr', file=sys.stderr); sys.exit(3)"),
    )
    assert isinstance(res, ExecResult)
    assert res.exit_code == 3  # non-zero is returned, not raised
    assert res.stdout == "to-stdout\n"
    assert res.stderr.strip() == "to-stderr"


def test_exec_success_and_argv_is_not_shell_parsed(backend: Backend, handle: RuntimeHandle) -> None:
    tricky = "a b; echo $HOME 'quoted'"
    res = backend.exec(handle, py("import sys; print(sys.argv[1])") + [tricky])
    assert res.exit_code == 0
    assert res.stdout == tricky + "\n"


def test_exec_cwd_defaults_to_remote_root(backend: Backend, handle: RuntimeHandle) -> None:
    res = backend.exec(
        handle, py("import os, sys; print(os.path.samefile(os.getcwd(), sys.argv[1]))") + [handle.remote_root]
    )
    assert res.exit_code == 0, res.stderr
    assert res.stdout.strip() == "True"


def test_exec_explicit_cwd(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    backend.write_file(handle, f"{workdir}/sub/marker.txt", b"here")
    sub = remote_path(handle, f"{workdir}/sub")
    res = backend.exec(handle, py("print(open('marker.txt').read())"), cwd=sub)
    assert res.exit_code == 0, res.stderr
    assert res.stdout.strip() == "here"


def test_exec_env_is_added_to_the_environment(backend: Backend, handle: RuntimeHandle) -> None:
    # `python` must still resolve, i.e. env extends the default environment.
    res = backend.exec(
        handle,
        py("import os; print(os.environ['NBGPU_CONTRACT_VAR']); print(bool(os.environ.get('PATH')))"),
        env={"NBGPU_CONTRACT_VAR": "hello world"},
    )
    assert res.exit_code == 0, res.stderr
    assert res.stdout.splitlines() == ["hello world", "True"]


def test_exec_timeout_raises_backend_error(backend: Backend, handle: RuntimeHandle) -> None:
    start = time.monotonic()
    with pytest.raises(BackendError):
        backend.exec(handle, py("import time; time.sleep(60)"), timeout_s=1.0)
    assert time.monotonic() - start < 30


# --------------------------------------------------------------------------
# spawn


def test_spawn_lines_and_exit_code(backend: Backend, handle: RuntimeHandle) -> None:
    proc = backend.spawn(
        handle,
        [PY, "-u", "-c", "import sys; print('out-1'); print('err-1', file=sys.stderr); print('out-2'); sys.exit(5)"],
    )
    lines = collect_lines(proc, timeout_s=60)
    # Merged stdout and stderr, no trailing newlines.
    assert set(lines) >= {"out-1", "err-1", "out-2"}
    assert lines.index("out-1") < lines.index("out-2")
    assert all(not line.endswith("\n") for line in lines)
    assert wait_exit(proc, timeout_s=30) == 5


def test_spawn_returns_immediately_and_polls_none_while_running(backend: Backend, handle: RuntimeHandle) -> None:
    start = time.monotonic()
    proc = backend.spawn(handle, py("import time; time.sleep(120)"))
    try:
        assert time.monotonic() - start < 20
        assert proc.poll() is None
        time.sleep(1)
        assert proc.poll() is None
    finally:
        proc.terminate()
    assert wait_exit(proc, timeout_s=30) is not None
    proc.terminate()  # already exited: no error


def test_spawn_cwd_and_env(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    backend.write_file(handle, f"{workdir}/placeholder", b"")
    cwd = remote_path(handle, workdir)
    proc = backend.spawn(
        handle,
        [PY, "-u", "-c", "import os, sys; print(os.path.samefile(os.getcwd(), sys.argv[1])); print(os.environ['NBGPU_SPAWN_VAR'])", cwd],
        cwd=cwd,
        env={"NBGPU_SPAWN_VAR": "spawned"},
    )
    lines = collect_lines(proc, timeout_s=60)
    assert lines == ["True", "spawned"]
    assert wait_exit(proc, timeout_s=30) == 0


def test_spawn_default_cwd_is_remote_root(backend: Backend, handle: RuntimeHandle) -> None:
    proc = backend.spawn(
        handle, [PY, "-u", "-c", "import os, sys; print(os.path.samefile(os.getcwd(), sys.argv[1]))", handle.remote_root]
    )
    assert collect_lines(proc, timeout_s=60) == ["True"]


def test_spawned_process_terminate(backend: Backend, handle: RuntimeHandle) -> None:
    proc = backend.spawn(handle, [PY, "-u", "-c", "import time; print('started', flush=True); time.sleep(300)"])
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.extend(proc.lines()), daemon=True)
    reader.start()
    wait_for(lambda: "started" in lines, 60, what="spawned process output")
    proc.terminate()
    wait_exit(proc, timeout_s=30)
    reader.join(30)
    assert not reader.is_alive(), "lines() must end once the process is terminated"
    proc.terminate()  # idempotent


# --------------------------------------------------------------------------
# expose


def _http_get(url: str, timeout_s: float = 5) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_expose_reaches_a_server_on_the_mapped_port(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    url = backend.expose(handle, EXPOSED_PORT)
    assert url.startswith(("http://", "https://"))
    assert not url.endswith("/")
    assert backend.expose(handle, EXPOSED_PORT) == url  # stable

    token = uuid.uuid4().hex
    backend.write_file(handle, f"{workdir}/www/marker.txt", token.encode())
    proc = backend.spawn(
        handle,
        [PY, "-m", "http.server", str(handle.port_for(EXPOSED_PORT)), "--bind", "0.0.0.0"],
        cwd=remote_path(handle, f"{workdir}/www"),
    )
    try:
        def fetch() -> bool:
            try:
                status, body = _http_get(url + "/marker.txt")
            except (urllib.error.URLError, OSError, TimeoutError):
                return False
            return status == 200 and body == token.encode()

        wait_for(fetch, timeout_s=90, interval_s=0.5, what=f"{url}/marker.txt to serve the marker")
    finally:
        proc.terminate()


# --------------------------------------------------------------------------
# files


def test_read_file_missing_is_none(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    assert backend.read_file(handle, f"{workdir}/does-not-exist.txt") is None
    assert backend.read_file(handle, f"{workdir}/no/such/dir/file.txt") is None


def test_write_file_creates_parents_and_is_visible_to_exec(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    rel = f"{workdir}/a/b/c/hello.txt"
    backend.write_file(handle, rel, b"hello from write_file\n")
    assert backend.read_file(handle, rel) == b"hello from write_file\n"
    res = backend.exec(handle, ["cat", rel])
    assert res.exit_code == 0, res.stderr
    assert res.stdout == "hello from write_file\n"


def test_absolute_paths_are_used_as_is(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    absolute = remote_path(handle, f"{workdir}/abs/file.txt")
    backend.write_file(handle, absolute, b"absolute")
    assert backend.read_file(handle, f"{workdir}/abs/file.txt") == b"absolute"
    assert backend.read_file(handle, absolute) == b"absolute"
    res = backend.exec(handle, ["cat", absolute])
    assert res.stdout == "absolute"


def test_write_file_roundtrips_binary_and_replaces(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    rel = f"{workdir}/bin/blob.bin"
    data = bytes(range(256)) * 64 + b"\x00\xff\r\n"
    backend.write_file(handle, rel, data)
    assert backend.read_file(handle, rel) == data

    backend.write_file(handle, rel, b"short")
    assert backend.read_file(handle, rel) == b"short"

    backend.write_file(handle, rel, b"")
    assert backend.read_file(handle, rel) == b""

    # Atomic replacement must not leave temp files behind.
    assert backend.hash_files(handle, f"{workdir}/bin") == {"blob.bin": sha256(b"")}


def test_write_file_large(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    data = os.urandom(3 * 1024 * 1024)
    backend.write_file(handle, f"{workdir}/large.bin", data)
    assert backend.read_file(handle, f"{workdir}/large.bin") == data


def test_written_file_visible_to_remote_python(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    backend.write_file(handle, f"{workdir}/pkg/mod.py", b"VALUE = 'imported'\n")
    res = backend.exec(handle, py("import mod; print(mod.VALUE)"), cwd=remote_path(handle, f"{workdir}/pkg"))
    assert res.exit_code == 0, res.stderr
    assert res.stdout.strip() == "imported"


def test_files_written_by_remote_processes_are_readable(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    res = backend.exec(
        handle,
        ["sh", "-c", f"mkdir -p {shlex.quote(workdir)}/out && printf 'from-runtime' > {shlex.quote(workdir)}/out/r.txt"],
    )
    assert res.exit_code == 0, res.stderr
    assert backend.read_file(handle, f"{workdir}/out/r.txt") == b"from-runtime"


# --------------------------------------------------------------------------
# write_file compare-and-swap


def no_temp_files(backend: Backend, handle: RuntimeHandle, directory: str) -> list[str]:
    """Names in `directory` (as listed inside the runtime) starting with TEMP_PREFIX."""
    res = backend.exec(handle, py("import os, sys; print('\\n'.join(os.listdir(sys.argv[1])))") + [directory])
    assert res.exit_code == 0, res.stderr
    return [name for name in res.stdout.split() if name.startswith(TEMP_PREFIX)]


def test_write_file_returns_true_when_unconditional(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    rel = f"{workdir}/cas/any.txt"
    assert backend.write_file(handle, rel, b"one") is True
    assert backend.write_file(handle, rel, b"two", expect=None) is True
    assert backend.read_file(handle, rel) == b"two"
    assert no_temp_files(backend, handle, f"{workdir}/cas") == []


def test_write_file_expect_absent(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    rel = f"{workdir}/cas/new/absent.txt"
    assert backend.write_file(handle, rel, b"first", expect=ABSENT) is True  # creates parents too
    assert backend.read_file(handle, rel) == b"first"
    # The file exists now: ABSENT fails and leaves it untouched.
    assert backend.write_file(handle, rel, b"second", expect=ABSENT) is False
    assert backend.read_file(handle, rel) == b"first"
    assert no_temp_files(backend, handle, f"{workdir}/cas/new") == []


def test_write_file_expect_absent_on_file_made_in_runtime(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    res = backend.exec(handle, ["sh", "-c", f"mkdir -p {shlex.quote(workdir)} && printf runtime > {shlex.quote(workdir)}/r.txt"])
    assert res.exit_code == 0, res.stderr
    assert backend.write_file(handle, f"{workdir}/r.txt", b"laptop", expect=ABSENT) is False
    assert backend.read_file(handle, f"{workdir}/r.txt") == b"runtime"
    assert no_temp_files(backend, handle, workdir) == []


def test_write_file_expect_hash_match_replaces(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    rel = f"{workdir}/cas/h.txt"
    assert backend.write_file(handle, rel, b"v1") is True
    assert backend.write_file(handle, rel, b"v2", expect=sha256(b"v1")) is True
    assert backend.read_file(handle, rel) == b"v2"
    # Upper-case hex is the same digest.
    assert backend.write_file(handle, rel, b"v3", expect=sha256(b"v2").upper()) is True
    assert backend.read_file(handle, rel) == b"v3"
    # Expecting the hash of empty content works for an empty file.
    assert backend.write_file(handle, rel, b"") is True
    assert backend.write_file(handle, rel, b"v4", expect=sha256(b"")) is True
    assert backend.read_file(handle, rel) == b"v4"
    assert no_temp_files(backend, handle, f"{workdir}/cas") == []


def test_write_file_expect_hash_mismatch_leaves_file_untouched(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    rel = f"{workdir}/cas/m.txt"
    assert backend.write_file(handle, rel, b"original") is True
    # Someone in the runtime changed it since the caller hashed it.
    res = backend.exec(handle, ["sh", "-c", f"printf changed > {shlex.quote(rel)}"])
    assert res.exit_code == 0, res.stderr
    assert backend.write_file(handle, rel, b"stale overwrite", expect=sha256(b"original")) is False
    assert backend.read_file(handle, rel) == b"changed"
    assert backend.hash_files(handle, f"{workdir}/cas") == {"m.txt": sha256(b"changed")}
    assert no_temp_files(backend, handle, f"{workdir}/cas") == []


def test_write_file_expect_hash_on_missing_file_is_false(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    rel = f"{workdir}/cas/gone.txt"
    assert backend.write_file(handle, rel, b"x", expect=sha256(b"something")) is False
    assert backend.read_file(handle, rel) is None
    assert no_temp_files(backend, handle, f"{workdir}/cas") == []


def test_write_file_rejects_malformed_expect(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    rel = f"{workdir}/cas/bad.txt"
    for bad in ("not-a-hash", "ab" * 31, "zz" * 32):
        with pytest.raises(BackendError):
            backend.write_file(handle, rel, b"x", expect=bad)
    assert backend.read_file(handle, rel) is None


# --------------------------------------------------------------------------
# hash_files


def test_hash_files_matches_hashlib_and_applies_exclusions(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    included = {
        "x.txt": b"x content\n",
        "empty.txt": b"",
        "d/y.bin": bytes(range(256)),
        "d/e/deep.py": b"print('deep')\n",
        ".hidden/kept.toml": b"a = 1\n",
        "gitignore-like.git.txt": b"not excluded: .git is only a whole path part",
    }
    excluded = {
        ".git/HEAD": b"ref: refs/heads/main\n",
        ".venv/lib/site.py": b"",
        "__pycache__/m.cpython-312.pyc": b"\x00",
        "d/__pycache__/q.pyc": b"\x00",
        "node_modules/pkg/index.js": b"module.exports = 1\n",
        "d/node_modules/x.js": b"",
    }
    for rel, data in {**included, **excluded}.items():
        backend.write_file(handle, f"{workdir}/hf/{rel}", data)

    got = backend.hash_files(handle, f"{workdir}/hf")
    assert got == {rel: sha256(data) for rel, data in included.items()}

    # Absolute root works too and gives the same relative keys.
    assert backend.hash_files(handle, remote_path(handle, f"{workdir}/hf")) == got

    # The default root is remote_root: keys are relative to it.
    everything = backend.hash_files(handle)
    assert everything[f"{workdir}/hf/x.txt"] == sha256(b"x content\n")
    assert not any(k.startswith(f"{workdir}/hf/.git/") for k in everything)


def test_hash_files_skips_temp_files_and_gives_plain_relative_paths(
    backend: Backend, handle: RuntimeHandle, workdir: str
) -> None:
    backend.write_file(handle, f"{workdir}/ht/a.txt", b"a")
    backend.write_file(handle, f"{workdir}/ht/d/{TEMP_PREFIX}leftover", b"partial")
    backend.write_file(handle, f"{workdir}/ht/{TEMP_PREFIX}x.py", b"partial")
    got = backend.hash_files(handle, f"{workdir}/ht")
    assert got == {"a.txt": sha256(b"a")}
    everything = backend.hash_files(handle)
    for key in everything:
        assert not key.startswith("/")
        assert ".." not in key.split("/")
        assert not key.rsplit("/", 1)[-1].startswith(TEMP_PREFIX)


def test_hash_files_missing_root_is_empty(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    assert backend.hash_files(handle, f"{workdir}/nothing-here") == {}


def hash_cache_path(handle: RuntimeHandle) -> str:
    """Where every backend keeps hash_files' stat cache: next to remote_root."""
    return str(PurePosixPath(handle.remote_root).parent / "nbgpu-hash-cache.json")


def test_hash_files_detects_change_with_same_size_and_mtime(
    backend: Backend, handle: RuntimeHandle, workdir: str
) -> None:
    rel = f"{workdir}/hc/same.bin"
    backend.write_file(handle, rel, b"A" * 512)
    time.sleep(2.2)  # past the cache's recently-modified window
    assert backend.hash_files(handle, f"{workdir}/hc") == {"same.bin": sha256(b"A" * 512)}
    # Rewrite in place with the same size and restore the mtime: only ctime changes.
    res = backend.exec(
        handle,
        py(
            "import os, sys; p = sys.argv[1]; st = os.stat(p)\n"
            "with open(p, 'r+b') as f: f.write(b'B' * 512)\n"
            "os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))\n"
            "st2 = os.stat(p); print(st2.st_mtime_ns == st.st_mtime_ns, st2.st_size == st.st_size)"
        )
        + [rel],
    )
    assert res.exit_code == 0, res.stderr
    assert res.stdout.split() == ["True", "True"]
    time.sleep(2.2)
    assert backend.hash_files(handle, f"{workdir}/hc") == {"same.bin": sha256(b"B" * 512)}


def test_hash_files_tolerates_corrupt_cache(backend: Backend, handle: RuntimeHandle, workdir: str) -> None:
    backend.write_file(handle, f"{workdir}/hk/k.txt", b"k")
    cache = hash_cache_path(handle)
    for garbage in (b"{definitely not json", b"[1, 2]", b'{"root": null, "files": "x"}'):
        backend.write_file(handle, cache, garbage)
        assert backend.hash_files(handle, f"{workdir}/hk") == {"k.txt": sha256(b"k")}
    # The cache lives outside remote_root, so it never shows up as a project file.
    assert backend.read_file(handle, cache) is not None
    assert "nbgpu-hash-cache.json" not in {k.rsplit("/", 1)[-1] for k in backend.hash_files(handle)}


def test_user_commands_do_not_get_the_internal_marker(backend: Backend, handle: RuntimeHandle) -> None:
    code = "import os; print(os.environ.get('NBGPU_INTERNAL', '-'))"
    res = backend.exec(handle, py(code))
    assert res.exit_code == 0, res.stderr
    assert res.stdout.strip() == "-"
    assert collect_lines(backend.spawn(handle, [PY, "-u", "-c", code]), timeout_s=60) == ["-"]
    # Passed explicitly, it is kept.
    res = backend.exec(handle, py(code), env={"NBGPU_INTERNAL": "1"})
    assert res.stdout.strip() == "1"


# --------------------------------------------------------------------------
# terminate


def test_terminate_kills_processes_and_is_idempotent(backend: Backend, runtimes: RuntimeFactory) -> None:
    handle = runtimes(tags={"role": "terminate-test"})
    proc = backend.spawn(handle, [PY, "-u", "-c", "import time; print('up', flush=True); time.sleep(600)"])
    lines: list[str] = []
    threading.Thread(target=lambda: lines.extend(proc.lines()), daemon=True).start()
    wait_for(lambda: "up" in lines, 60, what="spawned process to start")

    backend.terminate(handle)
    assert backend.get(handle.runtime_id) is None
    assert handle.runtime_id not in {h.runtime_id for h in backend.find({"nbgpu.test": runtimes.run_tag})}
    wait_exit(proc, timeout_s=60)

    backend.terminate(handle)  # already gone: no error


# --------------------------------------------------------------------------
# validation before provisioning


def test_unknown_gpu_is_unsupported_before_creating(backend: Backend, spec_dir: Path, run_tag: str) -> None:
    tag = uuid.uuid4().hex[:8]
    request = RuntimeRequest(
        spec=make_spec(spec_dir, backend.name, gpu="NOT-A-REAL-GPU-9000"),
        tags={"nbgpu.test": run_tag, "gpu-check": tag},
        ports=(EXPOSED_PORT,),
    )
    with pytest.raises(UnsupportedError) as info:
        handle = backend.create(request)
        backend.terminate(handle)  # only reached if the backend wrongly accepted it
    if backend.name == "modal":
        assert "A10G" in str(info.value) or "H100" in str(info.value), "should list known GPUs"
    assert backend.find({"gpu-check": tag}) == []


# --------------------------------------------------------------------------
# local-only details


@pytest.fixture
def local_backend_dir(tmp_path: Path) -> Iterator[Path]:
    base = tmp_path / "local-runtimes"
    yield base
    try:
        from nbgpu.backends.local import LocalBackend

        b = LocalBackend(base_dir=base)
        for h in b.find({}):
            b.terminate(h)
    except NotImplementedError:
        pass


def test_local_any_gpu_is_unsupported_and_nothing_is_created(local_backend_dir: Path, tmp_path: Path) -> None:
    from nbgpu.backends.local import LocalBackend

    backend = LocalBackend(base_dir=local_backend_dir)
    for gpu in ("A10G", "H100:2", "T4"):
        request = RuntimeRequest(spec=make_spec(tmp_path / "spec", "local", gpu=gpu), tags={"k": "v"}, ports=(EXPOSED_PORT,))
        with pytest.raises(UnsupportedError):
            backend.create(request)
    assert backend.find({}) == []
    # No runtime directory was made either.
    leftovers = [p for p in local_backend_dir.rglob("project")] if local_backend_dir.exists() else []
    assert leftovers == []


def test_local_state_dirs_are_private(tmp_path: Path) -> None:
    from nbgpu.backends.local import LocalBackend

    existing = tmp_path / "state"
    existing.mkdir(mode=0o755)
    os.chmod(existing, 0o755)
    base = existing / "nested" / "local"
    old_umask = os.umask(0o000)  # chmod must be exact, whatever the umask
    try:
        b = LocalBackend(base_dir=base)
        h = b.create(RuntimeRequest(spec=make_spec(tmp_path / "spec", "local"), tags={}))
    finally:
        os.umask(old_umask)
    try:
        assert (base.stat().st_mode & 0o777) == 0o700
        assert ((existing / "nested").stat().st_mode & 0o777) == 0o700
        assert (existing.stat().st_mode & 0o777) == 0o755  # pre-existing: untouched
    finally:
        b.terminate(h)

    # A pre-existing base_dir keeps its permissions.
    other = tmp_path / "other"
    other.mkdir()
    os.chmod(other, 0o750)
    b2 = LocalBackend(base_dir=other)
    assert b2.find({}) == []
    h2 = b2.create(RuntimeRequest(spec=make_spec(tmp_path / "spec", "local"), tags={}))
    try:
        assert (other.stat().st_mode & 0o777) == 0o750
    finally:
        b2.terminate(h2)


def test_local_registry_is_shared_across_instances(local_backend_dir: Path, tmp_path: Path) -> None:
    from nbgpu.backends.local import LocalBackend

    first = LocalBackend(base_dir=local_backend_dir)
    handle = first.create(
        RuntimeRequest(spec=make_spec(tmp_path / "spec", "local"), tags={"shared": "yes"}, ports=(EXPOSED_PORT,))
    )
    try:
        second = LocalBackend(base_dir=local_backend_dir)
        got = second.get(handle.runtime_id)
        assert got is not None and got.remote_root == handle.remote_root
        assert got.port_for(EXPOSED_PORT) == handle.port_for(EXPOSED_PORT)
        assert [h.runtime_id for h in second.find({"shared": "yes"})] == [handle.runtime_id]
        second.terminate(handle)
        assert first.get(handle.runtime_id) is None
    finally:
        first.terminate(handle)


def test_local_layout_ports_and_cleanup(local_backend_dir: Path, tmp_path: Path) -> None:
    from nbgpu.backends.local import LocalBackend

    backend = LocalBackend(base_dir=local_backend_dir)
    handle = backend.create(
        RuntimeRequest(spec=make_spec(tmp_path / "spec", "local"), tags={}, ports=(EXPOSED_PORT, 2718))
    )
    try:
        root = Path(handle.remote_root)
        assert root.name == "project"
        assert root.is_dir()
        assert root.resolve().is_relative_to(local_backend_dir.resolve())
        assert set(handle.port_map) == {EXPOSED_PORT, 2718}
        assert handle.port_for(EXPOSED_PORT) != handle.port_for(2718)
        assert backend.expose(handle, 2718) == f"http://127.0.0.1:{handle.port_for(2718)}"
    finally:
        backend.terminate(handle)
    assert not root.exists(), "terminate deletes the runtime directory"


def test_local_terminate_kills_the_whole_process_group(local_backend_dir: Path, tmp_path: Path) -> None:
    from nbgpu.backends.local import LocalBackend

    backend = LocalBackend(base_dir=local_backend_dir)
    handle = backend.create(RuntimeRequest(spec=make_spec(tmp_path / "spec", "local"), tags={}, ports=()))
    pid_file = tmp_path / "grandchild.pid"
    try:
        # A grandchild that would outlive its parent unless the group is killed.
        backend.spawn(handle, ["sh", "-c", f"sleep 600 & echo $! > {shlex.quote(str(pid_file))}; wait"])
        wait_for(lambda: pid_file.exists() and pid_file.read_text().strip(), 30, what="grandchild pid")
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)  # alive
    finally:
        backend.terminate(handle)

    def gone() -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        # A zombie still answers kill(0); treat a reaped or zombie child as gone.
        try:
            import psutil

            return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        except Exception:
            return True

    wait_for(gone, 15, what=f"grandchild {pid} to die")


def test_local_spawned_process_survives_the_backend_object(local_backend_dir: Path, tmp_path: Path) -> None:
    """`nbgpu start --detach` relies on runtime processes outliving the CLI
    process's backend object; termination is explicit."""
    from nbgpu.backends.local import LocalBackend

    backend = LocalBackend(base_dir=local_backend_dir)
    handle = backend.create(RuntimeRequest(spec=make_spec(tmp_path / "spec", "local"), tags={}, ports=()))
    try:
        proc = backend.spawn(handle, [PY, "-c", "import os, time; open('pid', 'w').write(str(os.getpid())); time.sleep(600)"])
        wait_for(lambda: backend.read_file(handle, "pid"), 30, what="pid file")
        pid = int(backend.read_file(handle, "pid") or b"0")
        del proc, backend
        import gc

        gc.collect()
        os.kill(pid, 0)  # still alive
    finally:
        LocalBackend(base_dir=local_backend_dir).terminate(handle)


def test_local_hash_files_does_not_reread_unchanged_files(
    local_backend_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import nbgpu.backends.local as local_mod
    from nbgpu.backends.local import LocalBackend

    backend = LocalBackend(base_dir=local_backend_dir)
    handle = backend.create(RuntimeRequest(spec=make_spec(tmp_path / "spec", "local"), tags={}, ports=()))
    try:
        for i in range(5):
            backend.write_file(handle, f"d/f{i}.txt", f"content {i}".encode())
        reads: list[str] = []
        real = local_mod._hash_file
        monkeypatch.setattr(local_mod, "_hash_file", lambda p: reads.append(p) or real(p))
        expected = {f"d/f{i}.txt": sha256(f"content {i}".encode()) for i in range(5)}

        # Within the recently-modified window every file is re-read.
        assert backend.hash_files(handle) == expected
        assert backend.hash_files(handle) == expected
        assert len(reads) == 10

        monkeypatch.setattr(local_mod, "HASH_RECENT_NS", 0)  # as if 2 s had passed
        reads.clear()
        assert backend.hash_files(handle) == expected
        assert reads == []  # all served from the stat cache

        backend.write_file(handle, "d/f2.txt", b"changed")
        assert backend.hash_files(handle) == {**expected, "d/f2.txt": sha256(b"changed")}
        assert [Path(p).name for p in reads] == ["f2.txt"]

        cache = Path(handle.remote_root).parent / "nbgpu-hash-cache.json"
        assert cache.is_file()
        assert not [p for p in cache.parent.iterdir() if p.name.startswith(TEMP_PREFIX)]
        # Without a usable cache, hashing still works (and re-reads).
        cache.write_text("{oops")
        reads.clear()
        assert backend.hash_files(handle) == {**expected, "d/f2.txt": sha256(b"changed")}
        assert len(reads) == 5
    finally:
        backend.terminate(handle)


def test_local_user_commands_never_inherit_internal_marker(
    local_backend_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nbgpu.backends.local import LocalBackend

    monkeypatch.setenv("NBGPU_INTERNAL", "1")  # e.g. the CLI itself was started by a helper
    backend = LocalBackend(base_dir=local_backend_dir)
    handle = backend.create(RuntimeRequest(spec=make_spec(tmp_path / "spec", "local"), tags={}, ports=()))
    try:
        res = backend.exec(handle, py("import os; print(os.environ.get('NBGPU_INTERNAL', '-'))"))
        assert res.stdout.strip() == "-"
    finally:
        backend.terminate(handle)
