import re

from pathlib import Path

from climate_risk.data.gadm import GADM
from climate_risk.data.geo_disasters import GEO_DISASTERS
from climate_risk.data.pwt import PWT
from climate_risk.data_functions.emdat_processing import EMDAT
from climate_risk.skills import bundle
from climate_risk.skills.bundle import available_skills, docs_source


def test_discovery_takes_only_directories_holding_a_skill_file(bundled):
    """A bare directory beside the skills would otherwise install as an empty skill."""
    found = available_skills(bundled)

    assert [skill.name for skill in found] == ["adding-a-place", "country-data"]


def test_discovery_of_a_missing_source_is_empty_rather_than_an_error(tmp_path):
    """An install without the package data must not crash the installer."""
    assert available_skills(tmp_path / "absent") == []


def test_the_concept_pages_resolve_in_this_checkout():
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


def test_the_package_ships_at_least_one_skill():
    """Discovery finding nothing leaves every test below it passing over an empty loop, and leaves
    the installer with nothing to install."""
    assert available_skills()


def test_every_shipped_skill_declares_a_name_and_a_description():
    """A skill missing either is invisible to the harness that is supposed to trigger it."""
    for skill in available_skills():
        text = (skill / "SKILL.md").read_text()

        assert text.startswith("---\n"), skill.name
        frontmatter = text.split("---\n")[1]
        assert re.search(r"^name:", frontmatter, re.M), skill.name
        assert re.search(r"^description:", frontmatter, re.M), skill.name


def test_every_router_link_reaches_a_file_that_will_exist():
    """A concepts/ link is satisfied by the installer, so a renamed user-guide page breaks it
    silently and only in an install nobody runs from a checkout."""
    pages = docs_source()
    assert pages is not None

    for skill in available_skills():
        links = re.findall(r"\]\(([^)]+)\)", (skill / "SKILL.md").read_text())
        for link in (target for target in links if "://" not in target):
            source = pages / Path(link).name if link.startswith("concepts/") else skill / link

            assert source.is_file(), f"{skill.name} routes to {link}, which is not there"


def test_the_shipped_prose_states_the_real_filename_for_every_licensed_source():
    """These four cannot be downloaded, so a wrong filename in the prose is a reader stuck at the
    first step with no error to search for."""
    prose = "\n".join(page.read_text() for skill in available_skills() for page in (skill / "references").glob("*.md"))

    for source in (EMDAT, GADM, GEO_DISASTERS, PWT):
        assert source.filename in prose, source.filename
        assert source.homepage in prose, source.homepage
