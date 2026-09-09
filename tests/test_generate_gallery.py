"""The gallery extension is build tooling under ``docs/sphinxext/``, so it has no sister module in
``climate_risk``. It earns a test file anyway: its failure modes are silent, producing a card grid
that links nowhere or a thumbnail nobody wrote, and none of them show up in a build that has no
notebooks to stage."""

import base64
import importlib.util
import json
import subprocess

from pathlib import Path

import pytest

EXTENSION = Path(__file__).resolve().parents[1] / "docs" / "sphinxext" / "generate_gallery.py"

_spec = importlib.util.spec_from_file_location("generate_gallery", EXTENSION)
assert _spec is not None and _spec.loader is not None, f"no extension at {EXTENSION}"
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)

# A 1x1 red PNG, small enough to inline and real enough for matplotlib to read back.
RED_PIXEL = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080200000090"
        "7753de0000000c4944415408d763f8cfc00000030101003e5c9c2d000000"
        "0049454e44ae426082"
    )
).decode()

# A 1x1 grayscale PNG, which matplotlib reads as a 2D array with no channel axis.
GRAY_PIXEL = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108000000003a"
        "7e9b550000000a49444154789c636800000082008177cd72b60000000049"
        "454e44ae426082"
    )
).decode()


def write_notebook(path: Path, *, image: str | None = RED_PIXEL) -> Path:
    """Write a notebook with one cell, optionally carrying a base64 PNG output."""
    outputs = [{"output_type": "display_data", "data": {"image/png": image}}] if image else []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": [], "outputs": outputs}]}))

    return path


@pytest.fixture
def examples(tmp_path, monkeypatch):
    """A git repository holding an examples/ tree, which is what the extension reads."""
    repo = tmp_path / "repo"
    (repo / "examples").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    monkeypatch.setattr(gallery, "REPO_ROOT", repo)
    monkeypatch.setattr(gallery, "NOTEBOOKS_ROOT", repo / "examples")

    return repo / "examples"


def track(repo_examples: Path, notebook: Path) -> None:
    subprocess.run(["git", "add", str(notebook)], cwd=repo_examples.parent, check=True)


def render(examples_root: Path, tmp_path: Path) -> tuple[str, Path, Path]:
    staged, thumbnails = tmp_path / "staged", tmp_path / "thumbs"
    staged.mkdir(exist_ok=True)
    thumbnails.mkdir(exist_ok=True)
    page = gallery.render_page(gallery.ordered_sections(gallery.discover_notebooks()), staged, thumbnails)

    return page, staged, thumbnails


def test_a_notebook_becomes_a_card_linking_to_where_it_was_staged(examples, tmp_path):
    """A card whose link does not match the staged path renders fine and goes nowhere."""
    notebook = write_notebook(examples / "data_loaders" / "co2.ipynb")
    track(examples, notebook)

    page, staged, _ = render(examples, tmp_path)

    assert ":link: data_loaders/co2" in page
    assert (staged / "data_loaders" / "co2.ipynb").is_file()


def test_the_hidden_toctree_lists_every_staged_notebook(examples, tmp_path):
    """Sphinx warns that a document is in no toctree, and the cards still work, so an unpopulated
    toctree survives review."""
    for name in ("co2.ipynb", "gpcc.ipynb"):
        track(examples, write_notebook(examples / "data_loaders" / name))

    page, _, _ = render(examples, tmp_path)
    toctree = page[page.index(".. toctree::") : page.index(".. grid::")]

    assert "data_loaders/co2" in toctree
    assert "data_loaders/gpcc" in toctree


def test_a_notebook_with_an_image_gets_a_thumbnail(examples, tmp_path):
    track(examples, write_notebook(examples / "data_loaders" / "co2.ipynb"))

    _, _, thumbnails = render(examples, tmp_path)

    assert (thumbnails / "data_loaders" / "co2.png").stat().st_size > 0


def test_a_notebook_without_an_image_gets_the_placeholder(examples, tmp_path):
    """The card names a thumbnail path either way, so writing nothing leaves a broken image."""
    track(examples, write_notebook(examples / "data_loaders" / "bare.ipynb", image=None))

    page, _, thumbnails = render(examples, tmp_path)

    assert (thumbnails / "data_loaders" / "bare.png").read_bytes() == gallery.PLACEHOLDER.read_bytes()
    assert ":img-top: /_thumbnails/data_loaders/bare.png" in page


def test_a_single_channel_image_gets_a_thumbnail(examples, tmp_path):
    """A grayscale PNG has no channel axis for the thumbnail border to be written into."""
    track(examples, write_notebook(examples / "data_loaders" / "gray.ipynb", image=GRAY_PIXEL))

    _, _, thumbnails = render(examples, tmp_path)

    assert (thumbnails / "data_loaders" / "gray.png").stat().st_size > 0


def test_an_existing_thumbnail_is_kept(examples, tmp_path):
    """A hand-made thumbnail overrides extraction, so a rebuild must not overwrite it."""
    track(examples, write_notebook(examples / "data_loaders" / "co2.ipynb"))
    thumbnails = tmp_path / "thumbs" / "data_loaders"
    thumbnails.mkdir(parents=True)
    (thumbnails / "co2.png").write_bytes(b"hand made")

    render(examples, tmp_path)

    assert (thumbnails / "co2.png").read_bytes() == b"hand made"


def test_an_untracked_notebook_is_left_out(examples, tmp_path):
    """Work in progress under examples/ would otherwise reach the published gallery."""
    write_notebook(examples / "data_loaders" / "draft.ipynb")

    page, staged, _ = render(examples, tmp_path)

    assert page == ""
    assert not (staged / "data_loaders" / "draft.ipynb").exists()


def test_no_notebooks_writes_no_page(examples, tmp_path):
    """The gallery ships before any content, and a page here would be a toctree entry pointing at
    nothing."""
    page, staged, _ = render(examples, tmp_path)

    assert page == ""
    assert list(staged.iterdir()) == []


def test_sections_follow_the_declared_order(examples, tmp_path):
    """The unlisted section is named so that alphabetical order would put it first, which is what
    the page falls back to when SECTION_ORDER stops being consulted."""
    track(examples, write_notebook(examples / "aaa_other" / "unlisted.ipynb"))
    track(examples, write_notebook(examples / "data_loaders" / "listed.ipynb"))

    page, _, _ = render(examples, tmp_path)

    assert page.index("Data loaders") < page.index("Aaa Other")


def test_an_unsectioned_notebook_leads_the_page(examples, tmp_path):
    track(examples, write_notebook(examples / "intro.ipynb"))
    track(examples, write_notebook(examples / "data_loaders" / "co2.ipynb"))

    page, _, _ = render(examples, tmp_path)

    assert page.index(":link: intro") < page.index("Data loaders")


def test_every_toctree_entry_resolves_to_a_staged_notebook(examples, tmp_path):
    """Sphinx resolves a toctree entry against the source tree, so an entry naming a document that
    was never staged fails the build with a message pointing at the generated page rather than at
    the extension that wrote it."""
    track(examples, write_notebook(examples / "intro.ipynb"))
    track(examples, write_notebook(examples / "data_loaders" / "co2.ipynb"))

    page, staged, _ = render(examples, tmp_path)
    body = page[page.index(".. toctree::") : page.index(".. grid::")]
    entries = [
        line.strip() for line in body.splitlines() if line.startswith("   ") and not line.strip().startswith(":")
    ]

    assert entries
    for entry in entries:
        assert (staged / f"{entry}.ipynb").is_file(), entry


def test_a_dotted_notebook_name_reaches_the_staged_tree_intact(examples, tmp_path):
    """A stem carrying a dot reads as a suffix, and a name truncated on the way to the staged tree
    leaves the toctree entry resolving to nothing."""
    track(examples, write_notebook(examples / "data_loaders" / "co2.v2.ipynb"))

    page, staged, _ = render(examples, tmp_path)

    assert (staged / "data_loaders" / "co2.v2.ipynb").is_file()
    assert "data_loaders/co2.v2" in page
