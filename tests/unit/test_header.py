"""nbgpu.header: editing a notebook's dependencies in place."""

from __future__ import annotations

import pytest

from nbgpu.header import requirement_name, update_dependencies
from nbgpu.spec import read_script_metadata

NB = '''# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "torch"]
#
# [tool.nbgpu]
# gpu = "A10"  # keep this comment
# ///

import marimo

app = marimo.App()
'''


def test_add_keeps_everything_else_byte_for_byte() -> None:
    text, deps = update_dependencies(NB, add=["emoji"])
    assert deps == ["marimo", "torch", "emoji"]
    assert text == NB.replace('["marimo", "torch"]', '["marimo", "torch", "emoji"]')


def test_existing_names_are_not_duplicated_but_explicit_versions_replace() -> None:
    assert update_dependencies(NB, add=["Torch"])[0] == NB
    text, deps = update_dependencies(NB, add=["torch==2.5.0"])
    assert deps == ["marimo", "torch==2.5.0"]


def test_remove_by_name_and_multiline_round_trip() -> None:
    text, _ = update_dependencies(NB, add=["a-very-long-package-name-number-one", "another-long-package-name-two"])
    assert "#     \"another-long-package-name-two\",\n# ]" in text  # one entry per line when long
    assert read_script_metadata(text)["tool"]["nbgpu"]["gpu"] == "A10"
    text, deps = update_dependencies(text, remove=["A_Very.Long-Package-Name-Number-One", "torch"])
    assert deps == ["marimo", "another-long-package-name-two"]
    assert read_script_metadata(text)["dependencies"] == deps
    assert "# gpu = \"A10\"  # keep this comment" in text


def test_a_header_without_dependencies_gets_them_after_requires_python() -> None:
    src = NB.replace('# dependencies = ["marimo", "torch"]\n', "")
    text, deps = update_dependencies(src, add=["emoji"])
    assert '# requires-python = ">=3.12"\n# dependencies = ["emoji"]\n' in text and deps == ["emoji"]


def test_a_file_without_a_header_gets_one() -> None:
    text, deps = update_dependencies("import marimo\n", add=["emoji"])
    assert text.startswith('# /// script\n# dependencies = ["emoji"]\n# ///\n\nimport marimo')
    assert update_dependencies("import marimo\n", remove=["x"])[0] == "import marimo\n"


def test_an_invalid_header_is_refused() -> None:
    with pytest.raises(ValueError):
        update_dependencies(NB.replace('gpu = "A10"', "gpu = ["), add=["x"])


@pytest.mark.parametrize("req, name", [("Scikit_Learn>=1.5", "scikit-learn"), ("emoji[full]==2", "emoji")])
def test_requirement_name(req: str, name: str) -> None:
    assert requirement_name(req) == name
