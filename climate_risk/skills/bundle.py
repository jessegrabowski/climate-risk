from pathlib import Path

SKILLS_SOURCE = Path(__file__).resolve().parent


def available_skills(source: Path | None = None) -> list[Path]:
    """
    Return the bundled skill directories, sorted by name.

    A skill is any subdirectory holding a ``SKILL.md``, so adding one needs no registry.

    Parameters
    ----------
    source : Path or None, optional
        Directory to search. Default None, meaning the skills shipped inside the package.

    Returns
    -------
    skills : list of Path
        One directory per skill.

    Examples
    --------
    .. code-block:: python

        from climate_risk.skills import available_skills

        for skill in available_skills():
            print(skill.name)
    """
    source = SKILLS_SOURCE if source is None else source

    if not source.is_dir():
        return []

    return sorted(path for path in source.iterdir() if (path / "SKILL.md").is_file())


def docs_source() -> Path | None:
    """
    Locate the concept pages: bundled in the wheel, else the repository tree for an editable install.

    Returns
    -------
    docs : Path or None
        Directory of concept pages, or None when neither copy is present.

    Examples
    --------
    .. code-block:: python

        from climate_risk.skills import docs_source

        pages = docs_source()
    """
    packaged = SKILLS_SOURCE.parent / "docs"
    if packaged.is_dir():
        return packaged

    repo = SKILLS_SOURCE.parents[1] / "docs" / "source" / "user_guide"

    return repo if repo.is_dir() else None
