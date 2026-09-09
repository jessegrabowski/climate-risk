from climate_risk.skills import bundle
from climate_risk.skills.bundle import available_skills, docs_source


def test_discovery_takes_only_directories_holding_a_skill_file(bundled):
    """A bare directory beside the skills would otherwise install as an empty skill."""
    found = available_skills(bundled)

    assert [skill.name for skill in found] == ["adding-a-place", "country-data"]


def test_discovery_of_a_missing_source_is_empty_rather_than_an_error(tmp_path):
    """An install without the package data must not crash the installer."""
    assert available_skills(tmp_path / "absent") == []


def test_the_concept_pages_resolve_in_this_checkout(tmp_path):
    """An editable install has no packaged copy, so the repository tree is the only source."""
    found = docs_source()

    assert found is not None
    assert (found / "data-layer.md").is_file()


def test_the_packaged_pages_win_over_the_repository_tree(tmp_path, monkeypatch):
    """Every non-editable install reads the packaged copy, and only an editable one has the repo."""
    packaged = tmp_path / "climate_risk" / "skills"
    packaged.mkdir(parents=True)
    (tmp_path / "climate_risk" / "docs").mkdir()
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", packaged)

    assert docs_source() == tmp_path / "climate_risk" / "docs"


def test_no_pages_are_found_when_neither_copy_is_present(tmp_path, monkeypatch):
    monkeypatch.setattr(bundle, "SKILLS_SOURCE", tmp_path / "nowhere" / "climate_risk" / "skills")

    assert docs_source() is None
