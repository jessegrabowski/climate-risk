import shutil

from pathlib import Path

CLAUDE_CONFIG_DIRECTORY = ".claude"


def claude_skills_dir(root: Path) -> Path | None:
    """
    Return the skills directory Claude Code reads under ``root``, if it is configured there.

    The directory is returned only when ``root/.claude`` already exists. Nothing is written into a
    harness the user has not installed, so a machine without Claude Code gets nothing rather than a
    stray configuration tree.

    Parameters
    ----------
    root : Path
        Home directory for a user-scoped install, or a project directory for a project-scoped one.

    Returns
    -------
    skills_dir : Path or None
        Where skills belong, or None when Claude Code is not configured under ``root``.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.skills import claude_skills_dir

        target = claude_skills_dir(Path.home())
    """
    config = root / CLAUDE_CONFIG_DIRECTORY

    return config / "skills" if config.is_dir() else None


def install_skill(source: Path, target: Path, *, force: bool = False) -> str:
    """
    Copy one skill into place, leaving anything already there alone unless ``force``.

    The copy is a real copy rather than a symlink, so the installed skill keeps working when the
    package is upgraded or removed.

    Parameters
    ----------
    source : Path
        The bundled skill directory to copy.
    target : Path
        Where the skill should land.
    force : bool, optional
        Replace ``target`` if something is already there, discarding it. Default False.

    Returns
    -------
    report : str
        One line saying what happened, for printing.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.skills import available_skills, install_skill

        skill = available_skills()[0]
        print(install_skill(skill, Path("proj/.claude/skills") / skill.name))
    """
    if target.exists() or target.is_symlink():
        if not force:
            return f"skip  {target.name}  (exists; pass --force to replace)"

        # A symlink from an older install scheme must be unlinked, never followed: rmtree through
        # one deletes whatever it points at.
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)

    return f"install  {target.name}"


def install_all(skills: list[Path], skills_dir: Path, *, force: bool = False) -> list[str]:
    """
    Install every skill into ``skills_dir``.

    Parameters
    ----------
    skills : list of Path
        The skill directories to install.
    skills_dir : Path
        Directory the skills are written into.
    force : bool, optional
        Replace entries that already exist. Default False.

    Returns
    -------
    reports : list of str
        One line per skill, in the order they were installed.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.skills import available_skills, install_all

        for line in install_all(available_skills(), Path("proj/.claude/skills")):
            print(line)
    """
    return [install_skill(skill, skills_dir / skill.name, force=force) for skill in skills]
