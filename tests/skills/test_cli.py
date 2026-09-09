from pathlib import Path

from climate_risk.skills import bundle
from climate_risk.skills.agents_file import BLOCK_START
from climate_risk.skills.cli import main


def test_listing_names_the_skills_without_writing_anything(bundled, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", bundled)

    status = main(["--list", "--project", str(tmp_path)])

    assert status == 0
    assert capsys.readouterr().out.split() == ["adding-a-place", "country-data"]
    assert not (tmp_path / ".claude").exists()


def test_a_project_that_does_not_exist_fails_instead_of_raising(bundled, tmp_path, capsys, monkeypatch):
    """resolve() succeeds on a path that is not there, so nothing catches a typo before the write."""
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", bundled)

    status = main(["--project", str(tmp_path / "absent")])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert "No directory at" in captured.err


def test_an_empty_source_fails_on_stderr(tmp_path, capsys, monkeypatch):
    """Reports go to stdout, so an error mixed in there corrupts whatever is reading them."""
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", tmp_path / "absent")

    status = main(["--project", str(tmp_path)])
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert "No skills found" in captured.err


def test_a_project_install_writes_the_skills_and_the_block(bundled, tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", bundled)
    (tmp_path / ".claude").mkdir()

    assert main(["--project", str(tmp_path)]) == 0
    assert (tmp_path / ".claude" / "skills" / "country-data" / "SKILL.md").is_file()
    assert BLOCK_START in (tmp_path / "AGENTS.md").read_text()


def test_a_project_without_claude_gets_the_block_and_no_config_tree(bundled, tmp_path, monkeypatch):
    """The block is the whole point of AGENTS.md: it reaches a harness nobody anticipated."""
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", bundled)

    assert main(["--project", str(tmp_path)]) == 0
    assert BLOCK_START in (tmp_path / "AGENTS.md").read_text()
    assert not (tmp_path / ".claude").exists()


def test_forcing_does_not_rewrite_a_hand_edited_block(bundled, tmp_path, monkeypatch):
    """--force is scoped to skill directories. It must never be a licence to edit this file harder."""
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", bundled)
    (tmp_path / ".claude").mkdir()
    main(["--project", str(tmp_path)])
    agents = tmp_path / "AGENTS.md"
    agents.write_text(agents.read_text().replace(BLOCK_START, "# Kept\n\n" + BLOCK_START))

    main(["--project", str(tmp_path), "--force"])

    assert agents.read_text().startswith("# Kept\n")


def test_a_user_install_writes_skills_and_no_agents_file(bundled, tmp_path, monkeypatch):
    """AGENTS.md is a project-level convention, and no agreed user-level location exists."""
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", bundled)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".claude").mkdir()

    assert main(["--user"]) == 0
    assert (tmp_path / ".claude" / "skills" / "country-data" / "SKILL.md").is_file()
    assert not (tmp_path / "AGENTS.md").exists()


def test_a_user_install_without_claude_writes_nothing_at_all(bundled, tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", bundled)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert main(["--user"]) == 0
    assert list(tmp_path.iterdir()) == [tmp_path / "bundled"]
