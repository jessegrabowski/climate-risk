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
