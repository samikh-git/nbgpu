"""Bundled agent skills and `nbgpu skills install`."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from nbgpu import __version__, skills
from nbgpu.cli import main


def test_packaged_skills_exist() -> None:
    root = skills.packaged_skills_dir()
    for name in skills.SKILL_NAMES:
        text = (root / name / "SKILL.md").read_text(encoding="utf-8")
        assert f"name: {name}" in text


def test_copy_install_writes_marker(tmp_path: Path) -> None:
    result = skills.install_skills(tmp_path)
    assert result.installed == list(skills.SKILL_NAMES)
    for name in skills.SKILL_NAMES:
        target = tmp_path / name
        assert not target.is_symlink()
        assert (target / "SKILL.md").is_file()
        assert (target / skills.MARKER).read_text().strip() == __version__
    assert not list(tmp_path.glob(".*nbgpu-tmp"))


def test_reinstall_replaces_owned_copies(tmp_path: Path) -> None:
    skills.install_skills(tmp_path)
    (tmp_path / "nbgpu" / "stale.md").write_text("old")
    result = skills.install_skills(tmp_path)
    assert result.replaced == list(skills.SKILL_NAMES)
    assert not (tmp_path / "nbgpu" / "stale.md").exists()


def test_link_install_and_replacing_a_link(tmp_path: Path) -> None:
    (tmp_path / "nbgpu").symlink_to(tmp_path / "somewhere-else")  # dangling, e.g. an old checkout
    result = skills.install_skills(tmp_path, link=True)
    assert result.replaced == ["nbgpu"] and result.installed == ["nbgpu-notebook"]
    for name in skills.SKILL_NAMES:
        assert (tmp_path / name).is_symlink()
        assert (tmp_path / name / "SKILL.md").is_file()


def test_foreign_directory_is_skipped_unless_forced(tmp_path: Path) -> None:
    mine = tmp_path / "nbgpu"
    mine.mkdir()
    (mine / "SKILL.md").write_text("my own")
    result = skills.install_skills(tmp_path)
    assert "nbgpu" in result.skipped
    assert (mine / "SKILL.md").read_text() == "my own"
    result = skills.install_skills(tmp_path, force=True)
    assert result.replaced == list(skills.SKILL_NAMES)  # nbgpu-notebook was installed by the first call
    assert (mine / skills.MARKER).is_file()


def test_cli_json_and_exit_code(tmp_path: Path) -> None:
    runner = CliRunner()
    out = runner.invoke(main, ["skills", "install", "--dir", str(tmp_path), "--json"])
    assert out.exit_code == 0, out.output
    data = json.loads(out.output)
    assert data["installed"] == list(skills.SKILL_NAMES) and data["skipped"] == {}

    foreign = tmp_path / "other"
    (foreign / "nbgpu").mkdir(parents=True)
    out = runner.invoke(main, ["skills", "install", "--dir", str(foreign)])
    assert out.exit_code == 1
