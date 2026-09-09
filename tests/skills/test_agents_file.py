from climate_risk.skills.agents_file import BLOCK_START, render_agents_block, update_agents_file


def test_the_block_is_appended_when_the_file_has_none(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("# My project\n\nBuild with make.\n")

    report = update_agents_file(agents, render_agents_block([], None))
    body = agents.read_text()

    assert "appended" in report
    assert body.startswith("# My project\n\nBuild with make.\n")
    assert BLOCK_START in body


def test_replacing_the_block_leaves_the_rest_of_the_file_intact(tmp_path):
    """The file belongs to the user. Losing the prose around the block is the worst outcome here."""
    agents = tmp_path / "AGENTS.md"
    update_agents_file(agents, render_agents_block([], None))
    agents.write_text(f"# Mine\n\n{agents.read_text()}\nTrailing note.\n")

    report = update_agents_file(agents, render_agents_block([], None))
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

    report = update_agents_file(agents, render_agents_block([], None))

    assert "unbalanced" in report
    assert agents.read_text() == original
