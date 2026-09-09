from climate_risk.skills.install import claude_skills_dir, install_skill


def test_a_second_install_keeps_a_local_edit(bundled, tmp_path):
    target = tmp_path / "skills" / "country-data"
    install_skill(bundled / "country-data", target)
    (target / "SKILL.md").write_text("edited by hand\n")

    report = install_skill(bundled / "country-data", target)

    assert "skip" in report
    assert (target / "SKILL.md").read_text() == "edited by hand\n"


def test_forcing_replaces_the_edit(bundled, tmp_path):
    target = tmp_path / "skills" / "country-data"
    install_skill(bundled / "country-data", target)
    (target / "SKILL.md").write_text("edited by hand\n")

    report = install_skill(bundled / "country-data", target, force=True)

    assert "install" in report
    assert "edited by hand" not in (target / "SKILL.md").read_text()


def test_forcing_over_a_symlink_unlinks_it_rather_than_writing_through_it(bundled, tmp_path):
    """rmtree through a symlink deletes whatever it points at, which an older scheme left behind."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.md").write_text("not the installer's\n")
    target = tmp_path / "skills" / "country-data"
    target.parent.mkdir(parents=True)
    target.symlink_to(elsewhere)

    install_skill(bundled / "country-data", target, force=True)

    assert (elsewhere / "keep.md").is_file()
    assert (target / "SKILL.md").is_file()


def test_the_skills_directory_is_found_where_claude_is_configured(tmp_path):
    (tmp_path / ".claude").mkdir()

    assert claude_skills_dir(tmp_path) == tmp_path / ".claude" / "skills"


def test_no_skills_directory_is_offered_where_claude_is_not_configured(tmp_path):
    """Writing here would build a config tree for a tool the user has never installed."""
    assert claude_skills_dir(tmp_path) is None
