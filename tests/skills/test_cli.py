from pathlib import Path

from climate_risk.skills import bundle
from climate_risk.skills.cli import main


def test_listing_names_the_skills_without_writing_anything(bundled, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", bundled)

    status = main(["--list", "--project", str(tmp_path)])

    assert status == 0
    assert capsys.readouterr().out.split() == ["adding-a-place", "country-data"]
    assert not (tmp_path / ".claude").exists()


def test_an_empty_source_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", tmp_path / "absent")

    assert main(["--project", str(tmp_path)]) == 1


def test_a_user_install_without_claude_writes_nothing_at_all(bundled, tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", bundled)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert main(["--user"]) == 0
    assert list(tmp_path.iterdir()) == [tmp_path / "bundled"]
