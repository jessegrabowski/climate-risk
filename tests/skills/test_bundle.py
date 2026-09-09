from climate_risk.skills.bundle import available_skills


def test_discovery_takes_only_directories_holding_a_skill_file(bundled):
    """A bare directory beside the skills would otherwise install as an empty skill."""
    found = available_skills(bundled)

    assert [skill.name for skill in found] == ["adding-a-place", "country-data"]


def test_discovery_of_a_missing_source_is_empty_rather_than_an_error(tmp_path):
    """An install without the package data must not crash the installer."""
    assert available_skills(tmp_path / "absent") == []
