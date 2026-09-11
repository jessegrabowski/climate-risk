from pathlib import Path

import pytest


def write_skill(root: Path, name: str) -> Path:
    """Write a minimal bundled skill under ``root`` and return its directory."""
    skill = root / name
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\n---\n\nrouter\n")
    (skill / "references" / "detail.md").write_text("detail\n")

    return skill


@pytest.fixture
def bundled(tmp_path):
    """Two skills, and a directory beside them that is not one."""
    source = tmp_path / "bundled"
    source.mkdir()
    write_skill(source, "country-data")
    write_skill(source, "adding-a-place")
    (source / "references").mkdir()

    return source
