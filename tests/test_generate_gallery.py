"""The gallery extension is build tooling under ``docs/sphinxext/``, so it has no sister module in
``climate_risk``. It earns a test file anyway: its failure modes are silent, producing a card grid
that links nowhere or a thumbnail nobody wrote, and none of them show up in a build that has no
notebooks to stage."""

import base64
import importlib.util
import json
import struct
import subprocess
import zlib

from pathlib import Path
from types import SimpleNamespace

import pytest

from matplotlib import image

EXTENSION = Path(__file__).resolve().parents[1] / "docs" / "sphinxext" / "generate_gallery.py"

_spec = importlib.util.spec_from_file_location("generate_gallery", EXTENSION)
assert _spec is not None and _spec.loader is not None, f"no extension at {EXTENSION}"
gallery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gallery)

SHIPPED_PLACEHOLDER = EXTENSION.parent / "no_thumbnail.png"

PUBLISHED_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def png(width: int, height: int, *, grayscale: bool = False) -> str:
    """Return a base64 PNG, single-channel when grayscale, which matplotlib reads as a 2D array."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload

        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    row = bytes([64, 128, 192][: 1 if grayscale else 3] * width)
    header = struct.pack(">IIBBBBB", width, height, 8, 0 if grayscale else 2, 0, 0, 0)
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"".join(b"\x00" + row for _ in range(height))))
        + chunk(b"IEND", b"")
    )

    return base64.b64encode(data).decode()


# Wider than it is tall, so a thumbnail that skipped the square crop is visible in the output.
WIDE_IMAGE = png(4, 2)
WIDE_GRAYSCALE_IMAGE = png(4, 2, grayscale=True)


def write_notebook(path: Path, *, output_image: str | None = WIDE_IMAGE) -> Path:
    """Write a notebook with one cell, optionally carrying a base64 PNG output."""
    outputs = [{"output_type": "display_data", "data": {"image/png": output_image}}] if output_image else []
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
    notebook = write_notebook(examples / "climate" / "co2.ipynb")
    track(examples, notebook)

    page, staged, _ = render(examples, tmp_path)

    assert ":link: climate/co2" in page
    assert (staged / "climate" / "co2.ipynb").is_file()


def test_the_hidden_toctree_lists_every_staged_notebook(examples, tmp_path):
    """Sphinx warns that a document is in no toctree, and the cards still work, so an unpopulated
    toctree survives review."""
    for name in ("co2.ipynb", "gpcc.ipynb"):
        track(examples, write_notebook(examples / "climate" / name))

    page, _, _ = render(examples, tmp_path)
    toctree = page[page.index(".. toctree::") : page.index(".. grid::")]

    assert "climate/co2" in toctree
    assert "climate/gpcc" in toctree


def test_a_notebook_with_an_image_gets_a_square_thumbnail_of_its_own(examples, tmp_path):
    """A card grid lays out square images, and the source is wider than it is tall."""
    track(examples, write_notebook(examples / "climate" / "co2.ipynb"))

    _, _, thumbnails = render(examples, tmp_path)
    thumbnail = thumbnails / "climate" / "co2.png"
    height, width = image.imread(thumbnail).shape[:2]

    assert height == width
    assert thumbnail.read_bytes() != SHIPPED_PLACEHOLDER.read_bytes()


def test_a_notebook_without_an_image_gets_the_placeholder(examples, tmp_path):
    """The card names a thumbnail path either way, so writing nothing leaves a broken image."""
    track(examples, write_notebook(examples / "climate" / "bare.ipynb", output_image=None))

    page, _, thumbnails = render(examples, tmp_path)

    assert (thumbnails / "climate" / "bare.png").read_bytes() == SHIPPED_PLACEHOLDER.read_bytes()
    assert ":img-top: /_thumbnails/climate/bare.png" in page


def test_a_single_channel_image_gets_a_thumbnail(examples, tmp_path):
    """A grayscale PNG has no channel axis for the thumbnail border to be written into."""
    track(examples, write_notebook(examples / "climate" / "gray.ipynb", output_image=WIDE_GRAYSCALE_IMAGE))

    _, _, thumbnails = render(examples, tmp_path)
    thumbnail = thumbnails / "climate" / "gray.png"

    assert thumbnail.read_bytes() != SHIPPED_PLACEHOLDER.read_bytes()


def test_an_existing_thumbnail_is_kept(examples, tmp_path):
    """A hand-made thumbnail overrides extraction, so a rebuild must not overwrite it."""
    track(examples, write_notebook(examples / "climate" / "co2.ipynb"))
    thumbnails = tmp_path / "thumbs" / "climate"
    thumbnails.mkdir(parents=True)
    (thumbnails / "co2.png").write_bytes(b"hand made")

    render(examples, tmp_path)

    assert (thumbnails / "co2.png").read_bytes() == b"hand made"


def test_an_untracked_notebook_is_left_out(examples, tmp_path):
    """Work in progress under examples/ would otherwise reach the published gallery."""
    track(examples, write_notebook(examples / "climate" / "co2.ipynb"))
    write_notebook(examples / "climate" / "draft.ipynb")

    page, staged, _ = render(examples, tmp_path)

    assert "draft" not in page
    assert not (staged / "climate" / "draft.ipynb").exists()
    assert (staged / "climate" / "co2.ipynb").is_file()


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
    track(examples, write_notebook(examples / "climate" / "listed.ipynb"))

    page, _, _ = render(examples, tmp_path)

    assert page.index("Data: Climate") < page.index("Aaa Other")


def test_every_published_section_is_declared():
    """Every other test here renders a fixture, so nothing looks at the sections the repository
    actually publishes. A folder in neither constant still renders, last on the page and headed by
    its own name title-cased, which is a section heading nobody chose.
    """
    sections = {path.parent.name for path in PUBLISHED_EXAMPLES.glob("*/*.ipynb")}

    assert sections <= set(gallery.SECTION_ORDER)
    assert sections <= set(gallery.SECTION_TITLES)


def test_an_unsectioned_notebook_leads_the_page(examples, tmp_path):
    track(examples, write_notebook(examples / "intro.ipynb"))
    track(examples, write_notebook(examples / "climate" / "co2.ipynb"))

    page, _, _ = render(examples, tmp_path)

    assert page.index(":link: intro") < page.index("Data: Climate")


def test_every_toctree_entry_resolves_to_a_staged_notebook(examples, tmp_path):
    """Sphinx resolves a toctree entry against the source tree, so an entry naming a document that
    was never staged fails the build with a message pointing at the generated page rather than at
    the extension that wrote it."""
    track(examples, write_notebook(examples / "intro.ipynb"))
    track(examples, write_notebook(examples / "climate" / "co2.ipynb"))

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
    track(examples, write_notebook(examples / "climate" / "co2.v2.ipynb"))

    page, staged, _ = render(examples, tmp_path)

    assert (staged / "climate" / "co2.v2.ipynb").is_file()
    assert "climate/co2.v2" in page


def test_a_checkpoint_copy_is_left_out(examples, tmp_path):
    """Jupyter writes .ipynb_checkpoints beside a notebook, and a committed one is tracked."""
    track(examples, write_notebook(examples / "climate" / "co2.ipynb"))
    track(examples, write_notebook(examples / "climate" / ".ipynb_checkpoints" / "co2-checkpoint.ipynb"))

    page, staged, _ = render(examples, tmp_path)

    assert "co2-checkpoint" not in page
    assert not (staged / "climate" / ".ipynb_checkpoints").exists()


def test_the_gallery_page_is_written_under_the_source_tree(examples, tmp_path):
    track(examples, write_notebook(examples / "climate" / "co2.ipynb"))
    source = tmp_path / "source"
    source.mkdir()

    gallery.build_gallery(SimpleNamespace(builder=SimpleNamespace(srcdir=source)))

    assert (source / "examples" / "gallery.rst").is_file()
    assert (source / "examples" / "climate" / "co2.ipynb").is_file()
    assert (source / "_thumbnails" / "climate" / "co2.png").is_file()


def test_no_notebooks_leaves_the_source_tree_without_a_gallery_page(examples, tmp_path):
    """Every build takes this path until the first notebook lands, and a page here would be a
    toctree entry naming a document nothing wrote."""
    source = tmp_path / "source"
    source.mkdir()

    gallery.build_gallery(SimpleNamespace(builder=SimpleNamespace(srcdir=source)))

    assert not (source / "examples" / "gallery.rst").exists()
