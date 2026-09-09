from pathlib import Path

AGENTS_FILE = "AGENTS.md"

BLOCK_SOURCE = Path(__file__).resolve().parent / "agents_block.md"

# The installer owns what sits between these and nothing else in the file. Both must be present, in
# order, for the block to be replaced.
BLOCK_START = "<!-- BEGIN climate_risk -->"
BLOCK_END = "<!-- END climate_risk -->"


def agents_block() -> str:
    """
    Return the block the installer owns inside an ``AGENTS.md``.

    Returns
    -------
    block : str
        The block as shipped, markers included.

    Examples
    --------
    .. code-block:: python

        from climate_risk.skills import agents_block

        print(agents_block())
    """
    return BLOCK_SOURCE.read_text()


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
        The block to write, as :func:`agents_block` returns.

    Returns
    -------
    report : str
        One line saying what happened, for printing.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.skills import agents_block, update_agents_file

        print(update_agents_file(Path("AGENTS.md"), agents_block()))
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
