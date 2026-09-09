from pathlib import Path

AGENTS_FILE = "AGENTS.md"

# The installer owns what sits between these and nothing else in the file. Both must be present, in
# order, for the block to be replaced.
BLOCK_START = "<!-- BEGIN climate_risk -->"
BLOCK_END = "<!-- END climate_risk -->"

# The two things an agent working against this package predictably gets wrong, and which cost a
# reader an afternoon each.
GUIDANCE = """\
Every loader takes a `cache_dir` argument. There is no default, no environment variable and no
project-root search, so resolve the path once and pass it down.

EM-DAT, GADM, Geo-Disasters and the Penn World Table cannot be downloaded by code. A loader that
needs one raises with the license and the exact path the file belongs at. Do not write a downloader
for them."""


def render_agents_block(skills: list[Path], skills_dir: Path | None) -> str:
    """
    Render the block the installer owns inside an ``AGENTS.md``.

    Parameters
    ----------
    skills : list of Path
        The skills that were installed.
    skills_dir : Path or None
        Where they landed, relative to the project, or None when no harness was configured to
        receive them. A path from outside the project would be meaningless to everyone else who
        reads the committed file.

    Returns
    -------
    block : str
        The block, markers included.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.skills import available_skills, render_agents_block

        print(render_agents_block(available_skills(), Path(".claude/skills")))
    """
    lines = [BLOCK_START, "", "## climate_risk", "", GUIDANCE, ""]

    if skills_dir is not None and skills:
        lines.append("Skills for this package, with the concept pages they route to:")
        lines.append("")
        lines += [f"- `{skills_dir / skill.name}/SKILL.md`" for skill in skills]
    else:
        lines.append("Run `install-climate-risk-skills` to place the skills where an agent finds them.")

    lines += ["", BLOCK_END, ""]

    return "\n".join(lines)


def update_agents_file(path: Path, block: str) -> str:
    """
    Write the owned block into ``path``, leaving every other line of it alone.

    The block is appended when the file has no markers and replaced in place when it has both. A
    file carrying one marker without the other is left untouched: it cannot be edited safely, and
    this file belongs to the user rather than to the installer.

    Parameters
    ----------
    path : Path
        The ``AGENTS.md`` to update. Created if absent.
    block : str
        The rendered block, as :func:`render_agents_block` returns.

    Returns
    -------
    report : str
        One line saying what happened, for printing.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.skills import render_agents_block, update_agents_file

        print(update_agents_file(Path("AGENTS.md"), render_agents_block([], None)))
    """
    body = path.read_text() if path.is_file() else ""

    start = body.find(BLOCK_START)
    end = body.find(BLOCK_END)

    if (start == -1) != (end == -1) or start > end:
        return f"skip  {path.name}  (markers are unbalanced; left alone)"

    if start == -1:
        separator = "" if body == "" or body.endswith("\n\n") else ("\n" if body.endswith("\n") else "\n\n")
        path.write_text(body + separator + block)

        return f"update  {path.name}  (block appended)"

    path.write_text(body[:start] + block.rstrip("\n") + body[end + len(BLOCK_END) :])

    return f"update  {path.name}  (block replaced)"
