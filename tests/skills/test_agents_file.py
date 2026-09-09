from climate_risk.skills.agents_file import BLOCK_END, BLOCK_START, agents_block, update_agents_file


def test_the_block_is_appended_when_the_file_has_none(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# My project\n\nBuild with make.\n")

    report = update_agents_file(agents, agents_block())
    body = agents.read_text()

    assert "appended" in report
    assert body.startswith("# My project\n\nBuild with make.\n")
    assert BLOCK_START in body


def test_replacing_the_block_leaves_the_rest_of_the_file_intact(tmp_path):
    """The file belongs to the user. Losing the prose around the block is the worst outcome here."""
    agents = tmp_path / "AGENTS.md"
    update_agents_file(agents, agents_block())
    agents.write_text(f"# Mine\n\n{agents.read_text()}\nTrailing note.\n")

    report = update_agents_file(agents, agents_block())
    body = agents.read_text()

    assert "replaced" in report
    assert body.startswith("# Mine\n")
    assert body.rstrip().endswith("Trailing note.")
    assert body.count(BLOCK_START) == 1


def test_a_file_with_one_marker_is_left_alone(tmp_path):
    """Half a block cannot be edited safely, and guessing risks eating the user's text."""
    agents = tmp_path / "AGENTS.md"
    original = f"# Mine\n\n{BLOCK_START}\n\nsomeone deleted the end marker\n"
    agents.write_text(original)

    report = update_agents_file(agents, agents_block())

    assert "unbalanced" in report
    assert agents.read_text() == original


def test_the_shipped_block_carries_both_markers():
    """Losing one silently turns every later install into a refusal to touch the file."""
    block = agents_block()

    assert block.startswith(BLOCK_START)
    assert block.rstrip().endswith(BLOCK_END)
