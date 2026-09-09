import argparse

from pathlib import Path

from climate_risk.skills import bundle
from climate_risk.skills.install import CLAUDE_CONFIG_DIRECTORY, claude_skills_dir, install_all


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="install-climate-risk-skills",
        description=(
            "Install the climate_risk agent skills where a coding agent will find them. "
            "Skills are written to .claude/skills/ when Claude Code is configured, and nothing is "
            "written into a harness that is not installed."
        ),
    )
    location = parser.add_mutually_exclusive_group()
    location.add_argument(
        "--user",
        action="store_true",
        help="Install into ~/.claude/skills/, for every project. The default.",
    )
    location.add_argument(
        "--project",
        metavar="DIR",
        help="Install into <DIR>/.claude/skills/, for that project only.",
    )
    parser.add_argument("--force", action="store_true", help="Replace a skill that is already installed.")
    parser.add_argument("--list", action="store_true", help="List the bundled skills and exit.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for ``install-climate-risk-skills``.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Default None, meaning read them from ``sys.argv``.

    Returns
    -------
    status : int
        Process exit status. Zero on success.

    Examples
    --------
    .. code-block:: python

        from climate_risk.skills import main

        main(["--list"])
    """
    args = _build_parser().parse_args(argv)

    skills = bundle.available_skills()
    if not skills:
        print(f"No skills found in {bundle.SKILLS_SOURCE}")
        return 1

    if args.list:
        for skill in skills:
            print(skill.name)
        return 0

    root = Path(args.project).expanduser().resolve() if args.project is not None else Path.home()

    skills_dir = claude_skills_dir(root)
    if skills_dir is None:
        print(f"No {CLAUDE_CONFIG_DIRECTORY} directory under {root}, so no skills were installed.")
    else:
        for line in install_all(skills, skills_dir, bundle.docs_source(), force=args.force):
            print(line)

    return 0
