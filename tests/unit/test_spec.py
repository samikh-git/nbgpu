"""Tests for nbgpu.spec: PEP 723 header parsing and compute spec validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from nbgpu.spec import (
    DEFAULT_IDLE_TIMEOUT_S,
    DEFAULT_MAX_RUNTIME_S,
    DEFAULT_PROVIDER,
    ComputeSpec,
    NotebookSpec,
    SpecError,
    load_spec,
    parse_duration,
    parse_memory,
    read_script_metadata,
)

MARIMO_BODY = textwrap.dedent(
    """\
    import marimo

    __generated_with = "0.25.2"
    app = marimo.App()


    @app.cell
    def _():
        import marimo as mo
        return (mo,)


    if __name__ == "__main__":
        app.run()
    """
)


def header(*lines: str) -> str:
    """Build a `# /// script` block from TOML lines (each gets a `# ` prefix,
    empty lines become a bare `#`)."""
    body = "\n".join(f"# {line}" if line else "#" for line in lines)
    return f"# /// script\n{body}\n# ///\n"


def write_notebook(tmp_path: Path, text: str, name: str = "nb.py") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# --------------------------------------------------------------------------
# read_script_metadata
# --------------------------------------------------------------------------


class TestReadScriptMetadata:
    def test_no_block_returns_none(self) -> None:
        assert read_script_metadata(MARIMO_BODY) is None

    def test_empty_text_returns_none(self) -> None:
        assert read_script_metadata("") is None

    def test_basic_block(self) -> None:
        text = header('requires-python = ">=3.12"', 'dependencies = ["marimo", "torch"]')
        assert read_script_metadata(text) == {
            "requires-python": ">=3.12",
            "dependencies": ["marimo", "torch"],
        }

    def test_block_with_no_tool_table(self) -> None:
        text = header('dependencies = ["marimo"]') + "\n" + MARIMO_BODY
        meta = read_script_metadata(text)
        assert meta == {"dependencies": ["marimo"]}
        assert "tool" not in meta

    def test_bare_hash_lines_are_blank_lines(self) -> None:
        text = header(
            'requires-python = ">=3.12"',
            "",
            "[tool.nbgpu]",
            'gpu = "A10G"',
            "",
            "cpu = 8",
        )
        assert read_script_metadata(text) == {
            "requires-python": ">=3.12",
            "tool": {"nbgpu": {"gpu": "A10G", "cpu": 8}},
        }

    def test_full_example_header_from_module_docstring(self) -> None:
        text = header(
            'requires-python = ">=3.12"',
            'dependencies = ["marimo", "torch"]',
            "",
            "[tool.nbgpu]",
            'provider = "modal"',
            'gpu = "A10G"',
            "cpu = 8",
            'memory = "32GiB"',
            'idle-timeout = "20m"',
            'max-runtime = "8h"',
            'volumes = { "/data" = "my-datasets" }',
        ) + "\n" + MARIMO_BODY
        meta = read_script_metadata(text)
        assert meta is not None
        assert meta["tool"]["nbgpu"] == {
            "provider": "modal",
            "gpu": "A10G",
            "cpu": 8,
            "memory": "32GiB",
            "idle-timeout": "20m",
            "max-runtime": "8h",
            "volumes": {"/data": "my-datasets"},
        }

    def test_block_of_only_bare_hash_is_empty_dict(self) -> None:
        assert read_script_metadata("# /// script\n#\n# ///\n") == {}

    def test_block_after_leading_docstring_and_code(self) -> None:
        text = '"""A notebook."""\n\n' + header('dependencies = ["numpy"]') + MARIMO_BODY
        assert read_script_metadata(text) == {"dependencies": ["numpy"]}

    def test_block_without_trailing_newline_at_eof(self) -> None:
        text = "# /// script\n# dependencies = []\n# ///"
        assert read_script_metadata(text) == {"dependencies": []}

    def test_two_script_blocks_raise(self) -> None:
        text = header('dependencies = ["a"]') + "\n" + header('dependencies = ["b"]')
        with pytest.raises(SpecError):
            read_script_metadata(text)

    def test_other_block_types_are_ignored(self) -> None:
        text = (
            "# /// pyproject\n# [project]\n# name = 'x'\n# ///\n\n"
            + header('dependencies = ["marimo"]')
        )
        assert read_script_metadata(text) == {"dependencies": ["marimo"]}

    def test_only_other_block_type_returns_none(self) -> None:
        text = "# /// pyproject\n# [project]\n# name = 'x'\n# ///\n"
        assert read_script_metadata(text) is None

    def test_opening_line_must_be_exact(self) -> None:
        # Indented or decorated markers are not a block start.
        text = "    # /// script\n    # dependencies = []\n    # ///\n"
        assert read_script_metadata(text) is None

    def test_invalid_toml_raises(self) -> None:
        text = header("dependencies = [", "this is not toml")
        with pytest.raises(SpecError):
            read_script_metadata(text)

    def test_spec_error_is_value_error(self) -> None:
        assert issubclass(SpecError, ValueError)


# --------------------------------------------------------------------------
# parse_duration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (90, 90),
        (3600, 3600),
        ("90s", 90),
        ("1s", 1),
        ("20m", 20 * 60),
        ("8h", 8 * 3600),
        ("1h30m", 3600 + 30 * 60),
        ("2h 15m", 2 * 3600 + 15 * 60),
        ("1h30m15s", 3600 + 30 * 60 + 15),
        ("45m30s", 45 * 60 + 30),
    ],
)
def test_parse_duration_valid(value: str | int, expected: int) -> None:
    result = parse_duration(value)
    assert result == expected
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -60,
        "",
        "soon",
        "5d",
        1.5,
        "0s",
        "0m",
        "-5m",
        "h",
        "m30",
        "20 minutes",
    ],
)
def test_parse_duration_invalid(value: object) -> None:
    with pytest.raises(SpecError):
        parse_duration(value)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# parse_memory
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (512, 512),
        (1, 1),
        ("512MiB", 512),
        ("1GiB", 1024),
        ("32GiB", 32 * 1024),
        # Decimal units: bytes / 2**20, rounded up.
        ("500MB", 477),  # 500e6 / 1048576 = 476.84
        ("32GB", 30518),  # 32e9 / 1048576 = 30517.58
        ("1GB", 954),  # 1e9 / 1048576 = 953.67
        ("1MB", 1),  # 0.95 MiB rounds up to 1
        # Case-insensitive units.
        ("32gib", 32 * 1024),
        ("32GIB", 32 * 1024),
        ("512mib", 512),
        ("32gb", 30518),
        ("500mb", 477),
    ],
)
def test_parse_memory_valid(value: str | int, expected: int) -> None:
    result = parse_memory(value)
    assert result == expected
    assert isinstance(result, int)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -512,
        "512",  # string without a unit
        "32",
        "",
        "0GiB",
        "-1GiB",
        "32XB",  # unknown unit
        "32 furlongs",
        "GiB",
        "lots",
    ],
)
def test_parse_memory_invalid(value: object) -> None:
    with pytest.raises(SpecError):
        parse_memory(value)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# load_spec
# --------------------------------------------------------------------------


FULL_HEADER = header(
    'requires-python = ">=3.12"',
    'dependencies = ["marimo", "torch"]',
    "",
    "[tool.nbgpu]",
    'provider = "modal"',
    'gpu = "A10G"',
    "cpu = 8",
    'memory = "32GiB"',
    'idle-timeout = "30m"',
    'max-runtime = "4h"',
    'volumes = { "/data" = "my-datasets" }',
)


def nbgpu_header(*tool_lines: str, deps: str = '["marimo"]') -> str:
    return header(f"dependencies = {deps}", "", "[tool.nbgpu]", *tool_lines)


class TestLoadSpecDefaults:
    def test_no_header_gets_all_defaults(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, MARIMO_BODY)
        spec = load_spec(path)
        assert isinstance(spec, NotebookSpec)
        assert spec.path == path.resolve()
        assert spec.requires_python is None
        assert spec.dependencies == ()
        assert spec.compute == ComputeSpec()
        assert spec.compute.provider == DEFAULT_PROVIDER == "modal"
        assert spec.compute.gpu is None
        assert spec.compute.cpu is None
        assert spec.compute.memory_mib is None
        assert spec.compute.idle_timeout_s == DEFAULT_IDLE_TIMEOUT_S == 20 * 60
        assert spec.compute.max_runtime_s == DEFAULT_MAX_RUNTIME_S == 8 * 3600
        assert dict(spec.compute.volumes) == {}

    def test_header_without_tool_table(self, tmp_path: Path) -> None:
        text = header('requires-python = ">=3.11"', 'dependencies = ["marimo", "numpy"]')
        path = write_notebook(tmp_path, text + "\n" + MARIMO_BODY)
        spec = load_spec(path)
        assert spec.requires_python == ">=3.11"
        assert spec.dependencies == ("marimo", "numpy")
        assert isinstance(spec.dependencies, tuple)
        assert spec.compute == ComputeSpec()

    def test_header_with_other_tool_table_only(self, tmp_path: Path) -> None:
        text = header('dependencies = ["marimo"]', "", "[tool.ruff]", "line-length = 100")
        path = write_notebook(tmp_path, text + MARIMO_BODY)
        spec = load_spec(path)
        assert spec.compute == ComputeSpec()
        assert spec.dependencies == ("marimo",)

    def test_empty_nbgpu_table(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, nbgpu_header() + MARIMO_BODY)
        assert load_spec(path).compute == ComputeSpec()

    def test_header_without_dependencies(self, tmp_path: Path) -> None:
        text = header('requires-python = ">=3.12"', "", "[tool.nbgpu]", 'gpu = "T4"')
        path = write_notebook(tmp_path, text + MARIMO_BODY)
        spec = load_spec(path)
        assert spec.dependencies == ()
        assert spec.requires_python == ">=3.12"
        assert spec.compute.gpu == "T4"


class TestLoadSpecHeader:
    def test_full_header(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, FULL_HEADER + "\n" + MARIMO_BODY)
        spec = load_spec(path)
        assert spec.requires_python == ">=3.12"
        assert spec.dependencies == ("marimo", "torch")
        c = spec.compute
        assert c.provider == "modal"
        assert c.gpu == "A10G"
        assert c.cpu == 8
        assert c.memory_mib == 32 * 1024
        assert c.idle_timeout_s == 30 * 60
        assert c.max_runtime_s == 4 * 3600
        assert dict(c.volumes) == {"/data": "my-datasets"}

    def test_gpu_kept_as_written(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, nbgpu_header('gpu = "H100:2"') + MARIMO_BODY)
        assert load_spec(path).compute.gpu == "H100:2"

    def test_lowercase_gpu_kept_as_written(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, nbgpu_header('gpu = "a10g"') + MARIMO_BODY)
        assert load_spec(path).compute.gpu == "a10g"

    def test_float_cpu(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, nbgpu_header("cpu = 0.5") + MARIMO_BODY)
        assert load_spec(path).compute.cpu == 0.5

    def test_integer_memory_is_mib(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, nbgpu_header("memory = 2048") + MARIMO_BODY)
        assert load_spec(path).compute.memory_mib == 2048

    def test_integer_durations_are_seconds(self, tmp_path: Path) -> None:
        path = write_notebook(
            tmp_path, nbgpu_header("idle-timeout = 300", "max-runtime = 7200") + MARIMO_BODY
        )
        c = load_spec(path).compute
        assert c.idle_timeout_s == 300
        assert c.max_runtime_s == 7200

    def test_multiple_volumes(self, tmp_path: Path) -> None:
        path = write_notebook(
            tmp_path,
            nbgpu_header('volumes = { "/data" = "datasets", "/models" = "weights" }')
            + MARIMO_BODY,
        )
        assert dict(load_spec(path).compute.volumes) == {
            "/data": "datasets",
            "/models": "weights",
        }

    def test_other_provider(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, nbgpu_header('provider = "local"') + MARIMO_BODY)
        assert load_spec(path).compute.provider == "local"

    def test_header_with_plain_python_body(self, tmp_path: Path) -> None:
        text = nbgpu_header('gpu = "T4"') + "\nprint('hello')\n"
        assert load_spec(write_notebook(tmp_path, text)).compute.gpu == "T4"


class TestLoadSpecPath:
    def test_relative_path_resolved_to_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sub = tmp_path / "proj"
        sub.mkdir()
        write_notebook(sub, MARIMO_BODY, name="analysis.py")
        monkeypatch.chdir(tmp_path)
        spec = load_spec(Path("proj/analysis.py"))
        assert spec.path.is_absolute()
        assert spec.path == (tmp_path / "proj" / "analysis.py").resolve()

    def test_dotted_relative_path_normalised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sub = tmp_path / "proj"
        sub.mkdir()
        write_notebook(sub, MARIMO_BODY, name="nb.py")
        monkeypatch.chdir(sub)
        spec = load_spec(Path("../proj/./nb.py"))
        assert spec.path == (sub / "nb.py").resolve()
        assert ".." not in spec.path.parts

    def test_missing_file_raises_spec_error(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError):
            load_spec(tmp_path / "does-not-exist.py")

    def test_two_blocks_in_file_raise(self, tmp_path: Path) -> None:
        text = header('dependencies = ["a"]') + header('dependencies = ["b"]') + MARIMO_BODY
        with pytest.raises(SpecError):
            load_spec(write_notebook(tmp_path, text))

    def test_invalid_toml_in_file_raises(self, tmp_path: Path) -> None:
        text = header("[tool.nbgpu", 'gpu = "A10G"') + MARIMO_BODY
        with pytest.raises(SpecError):
            load_spec(write_notebook(tmp_path, text))


class TestLoadSpecValidation:
    @pytest.mark.parametrize("key", ["gpus", "memory_mib", "idle_timeout", "region"])
    def test_unknown_header_key_names_the_key(self, tmp_path: Path, key: str) -> None:
        path = write_notebook(tmp_path, nbgpu_header(f'{key} = "x"') + MARIMO_BODY)
        with pytest.raises(SpecError, match=key):
            load_spec(path)

    @pytest.mark.parametrize(
        ("line", "key"),
        [
            ('provider = ""', "provider"),
            ("provider = 3", "provider"),
            ('gpu = ""', "gpu"),
            ("gpu = 1", "gpu"),
            ("gpu = true", "gpu"),
            ("cpu = 0", "cpu"),
            ("cpu = -2", "cpu"),
            ('cpu = "8"', "cpu"),
            ('memory = "32"', "memory"),
            ("memory = 0", "memory"),
            ('memory = "32 lightyears"', "memory"),
            ('idle-timeout = "soon"', "idle-timeout"),
            ("idle-timeout = 0", "idle-timeout"),
            ('max-runtime = "5d"', "max-runtime"),
            ("max-runtime = -1", "max-runtime"),
            ('volumes = { "data" = "rel" }', "volumes"),
            ('volumes = { "/data" = "" }', "volumes"),
            ('volumes = { "/data" = 3 }', "volumes"),
            ('volumes = "/data"', "volumes"),
            ('volumes = ["/data"]', "volumes"),
        ],
    )
    def test_invalid_header_values(self, tmp_path: Path, line: str, key: str) -> None:
        path = write_notebook(tmp_path, nbgpu_header(line) + MARIMO_BODY)
        with pytest.raises(SpecError, match=key):
            load_spec(path)


class TestLoadSpecOverrides:
    def test_overrides_win_over_header(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, FULL_HEADER + MARIMO_BODY)
        spec = load_spec(
            path,
            {
                "gpu": "H100:2",
                "cpu": 16,
                "memory": "64GiB",
                "idle-timeout": "5m",
                "max-runtime": "1h",
                "provider": "local",
                "volumes": {"/scratch": "tmp-vol"},
            },
        )
        c = spec.compute
        assert c.gpu == "H100:2"
        assert c.cpu == 16
        assert c.memory_mib == 64 * 1024
        assert c.idle_timeout_s == 5 * 60
        assert c.max_runtime_s == 3600
        assert c.provider == "local"
        assert dict(c.volumes) == {"/scratch": "tmp-vol"}
        # Non-compute fields still come from the header.
        assert spec.dependencies == ("marimo", "torch")
        assert spec.requires_python == ">=3.12"

    def test_partial_override_keeps_other_header_values(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, FULL_HEADER + MARIMO_BODY)
        c = load_spec(path, {"gpu": "T4"}).compute
        assert c.gpu == "T4"
        assert c.cpu == 8
        assert c.memory_mib == 32 * 1024
        assert c.idle_timeout_s == 30 * 60

    def test_overrides_apply_without_header(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, MARIMO_BODY)
        c = load_spec(path, {"gpu": "A100", "idle-timeout": 120}).compute
        assert c.gpu == "A100"
        assert c.idle_timeout_s == 120
        assert c.provider == DEFAULT_PROVIDER

    def test_empty_overrides_same_as_none(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, FULL_HEADER + MARIMO_BODY)
        assert load_spec(path, {}) == load_spec(path)
        assert load_spec(path, None) == load_spec(path)

    def test_unknown_override_key_names_the_key(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, MARIMO_BODY)
        with pytest.raises(SpecError, match="gpus"):
            load_spec(path, {"gpus": "A10G"})

    @pytest.mark.parametrize(
        ("overrides", "key"),
        [
            ({"gpu": ""}, "gpu"),
            ({"provider": ""}, "provider"),
            ({"cpu": 0}, "cpu"),
            ({"cpu": -1.5}, "cpu"),
            ({"memory": "lots"}, "memory"),
            ({"idle-timeout": "5d"}, "idle-timeout"),
            ({"max-runtime": 0}, "max-runtime"),
            ({"volumes": {"relative/path": "vol"}}, "volumes"),
        ],
    )
    def test_overrides_are_validated(
        self, tmp_path: Path, overrides: dict[str, object], key: str
    ) -> None:
        path = write_notebook(tmp_path, FULL_HEADER + MARIMO_BODY)
        with pytest.raises(SpecError, match=key):
            load_spec(path, overrides)

    def test_override_does_not_mask_invalid_header_for_other_key(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, nbgpu_header("cpu = -1") + MARIMO_BODY)
        with pytest.raises(SpecError, match="cpu"):
            load_spec(path, {"gpu": "T4"})

    def test_override_replaces_header_gpu(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, nbgpu_header('gpu = "A10G"') + MARIMO_BODY)
        assert load_spec(path, {"gpu": "L4"}).compute.gpu == "L4"

    def test_none_override_means_not_given(self, tmp_path: Path) -> None:
        path = write_notebook(tmp_path, FULL_HEADER + MARIMO_BODY)
        overrides = {"gpu": None, "cpu": None, "memory": None, "idle-timeout": None,
                     "max-runtime": None, "provider": None, "volumes": None}
        assert load_spec(path, overrides) == load_spec(path)


# --------------------------------------------------------------------------
# [tool.nbgpu] requirements


def _nb_with_requirements(tmp_path, value: str):
    nb = tmp_path / "nb.py"
    nb.write_text(f'# /// script\n# dependencies = ["marimo"]\n# [tool.nbgpu]\n# requirements = {value}\n# ///\n')
    return nb


def test_requirements_files_are_read_into_dependencies(tmp_path) -> None:
    (tmp_path / "base.txt").write_text("numpy>=2  # comment\n")
    (tmp_path / "requirements.txt").write_text("# header\n-r base.txt\n\npolars\n")
    (tmp_path / "dev.txt").write_text("pytest\n")
    spec = load_spec(_nb_with_requirements(tmp_path, '["requirements.txt", "dev.txt"]'))
    assert spec.dependencies == ("marimo", "numpy>=2", "polars", "pytest")
    assert [p.name for p in spec.requirement_files] == ["requirements.txt", "dev.txt"]


@pytest.mark.parametrize(
    "content, match",
    [("-e .\n", "unsupported line 1"), ("--index-url https://x\n", "unsupported"), ("-r requirements.txt\n", "itself")],
)
def test_requirements_files_reject_what_cannot_be_installed(tmp_path, content: str, match: str) -> None:
    (tmp_path / "requirements.txt").write_text(content)
    with pytest.raises(SpecError, match=match):
        load_spec(_nb_with_requirements(tmp_path, '"requirements.txt"'))


def test_missing_requirements_file_is_an_error(tmp_path) -> None:
    with pytest.raises(SpecError, match="cannot read"):
        load_spec(_nb_with_requirements(tmp_path, '"nope.txt"'))


@pytest.mark.parametrize(
    ("line", "expected"),
    [(None, "nbgpu-cache"), ("cache = true", "nbgpu-cache"), ("cache = false", None), ('cache = "team-cache"', "team-cache")],
)
def test_cache_key(tmp_path: Path, line: str | None, expected: str | None) -> None:
    path = write_notebook(tmp_path, nbgpu_header(*([line] if line else [])))
    assert load_spec(path).compute.cache_volume == expected


@pytest.mark.parametrize("line", ['cache = ""', "cache = 3", 'snapshot = "yes"'])
def test_cache_and_snapshot_reject_bad_values(tmp_path: Path, line: str) -> None:
    with pytest.raises(SpecError, match=line.split(" ")[0]):
        load_spec(write_notebook(tmp_path, nbgpu_header(line)))


def test_snapshot_key(tmp_path: Path) -> None:
    assert load_spec(write_notebook(tmp_path, nbgpu_header())).snapshot is False
    assert load_spec(write_notebook(tmp_path, nbgpu_header("snapshot = true"))).snapshot is True


def test_cell_cache_is_opt_in(tmp_path: Path) -> None:
    assert load_spec(write_notebook(tmp_path, nbgpu_header())).cell_cache is False
    assert load_spec(write_notebook(tmp_path, nbgpu_header("cell-cache = true"))).cell_cache is True
    with pytest.raises(SpecError, match="cell-cache"):
        load_spec(write_notebook(tmp_path, nbgpu_header('cell-cache = "yes"')))


def test_cell_cache_needs_the_cache_volume(tmp_path: Path) -> None:
    with pytest.raises(SpecError, match="needs `cache`"):
        load_spec(write_notebook(tmp_path, nbgpu_header("cell-cache = true", "cache = false")))
