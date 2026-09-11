import base64
import json
import logging
import shutil
import subprocess

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import image

try:
    from sphinx.util import logging as sphinx_logging

    logger = sphinx_logging.getLogger(__name__)
except ModuleNotFoundError:
    # Sphinx is only in the docs environment. Falling back keeps this module importable by the
    # test suite, which runs where it is not installed.
    logger = logging.getLogger(__name__)

# docs/sphinxext/generate_gallery.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_ROOT = REPO_ROOT / "examples"
PLACEHOLDER = Path(__file__).resolve().parent / "no_thumbnail.png"

GALLERY_DOCUMENT = "gallery"
GALLERY_TITLE = "Example gallery"

# A notebook's path decides where it lands. `examples/<section>/nb.ipynb` puts it under a section
# heading, and `examples/nb.ipynb` puts it on the page with no heading. List a folder in
# SECTION_ORDER to pin where it sits on the page. Anything unlisted follows, alphabetically.
SECTION_ORDER: tuple[str, ...] = ("climate", "hazard", "economic", "geospatial")

# Only where title-casing the folder name would get it wrong.
SECTION_TITLES: dict[str, str] = {
    "climate": "Data: Climate",
    "hazard": "Data: Hazard",
    "economic": "Data: Economic",
    "geospatial": "Data: Geospatial",
}

THUMBNAIL_SIZE = 275
THUMBNAIL_BORDER = 4
THUMBNAIL_DPI = 100

PAGE_TITLE = """
{title}
{underlines}
"""

TOCTREE_HEAD = """
.. toctree::
   :hidden:

"""

GRID_HEAD = """
.. grid:: 1 2 3 3
   :gutter: 4

"""

SECTION_TEMPLATE = (
    """
.. _gallery-{section_id}:

{section_title}
{underlines}
"""
    + GRID_HEAD
)

ITEM_TEMPLATE = """
   .. grid-item-card:: :doc:`{document}`
      :img-top: {image}
      :link: {document}
      :link-type: doc
      :shadow: none
"""


def is_tracked_by_git(path: Path) -> bool:
    """
    Report whether ``path`` is tracked, so an untracked draft stays out of the gallery.

    Parameters
    ----------
    path : Path
        The notebook to check.

    Returns
    -------
    tracked : bool
        True when git knows the file, and True when git is not available at all.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            capture_output=True,
            check=False,
            cwd=REPO_ROOT,
        )
    except FileNotFoundError:
        return True

    return result.returncode == 0


def crop_to_thumbnail(path: Path) -> None:
    """Crop the image at ``path`` to a bordered square, in place."""
    picture = image.imread(path)
    if picture.ndim == 2:
        picture = np.repeat(picture[:, :, None], 3, axis=2)

    rows, columns = picture.shape[:2]
    size = min(rows, columns)

    if size == columns:
        top = min(max(0, rows // 2 - size // 2), rows - size)
        cropped = picture[top : top + size, 0:size]
    else:
        left = min(max(0, columns // 2 - size // 2), columns - size)
        cropped = picture[0:size, left : left + size]

    border = THUMBNAIL_BORDER
    cropped[:border, :, :3] = cropped[-border:, :, :3] = 0
    cropped[:, :border, :3] = cropped[:, -border:, :3] = 0

    inches = THUMBNAIL_SIZE / THUMBNAIL_DPI
    figure = plt.figure(figsize=(inches, inches), dpi=THUMBNAIL_DPI)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0), aspect="auto", frameon=False, xticks=[], yticks=[])
    axes.imshow(cropped, aspect="auto", resample=True, interpolation="bilinear")
    figure.savefig(path, dpi=THUMBNAIL_DPI)
    plt.close(figure)


def last_image_output(notebook: Path) -> bytes | None:
    """
    Return the last PNG output in ``notebook``, which becomes its thumbnail.

    Parameters
    ----------
    notebook : Path
        The notebook to read.

    Returns
    -------
    png : bytes or None
        Decoded image data, or None when the notebook has no image output.
    """
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    encoded = None
    for cell in cells:
        for output in cell.get("outputs", []):
            if "image/png" in output.get("data", {}):
                encoded = output["data"]["image/png"]

    return base64.b64decode(encoded) if encoded is not None else None


def write_thumbnail(notebook: Path, target: Path) -> None:
    """
    Write the gallery thumbnail for ``notebook``, leaving an existing file alone.

    A notebook with no image output gets the shipped placeholder.

    Parameters
    ----------
    notebook : Path
        The notebook to take the image from.
    target : Path
        Where the thumbnail belongs. An existing file here wins, so a hand-made thumbnail overrides
        extraction.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        logger.info(f"Keeping the existing thumbnail for {notebook.name}")
        return

    preview = last_image_output(notebook)
    if preview is None:
        logger.warning(
            f"No image output in {notebook.name}, so its card gets the placeholder. Re-run the "
            f"notebook with outputs saved, or put a PNG at {target}."
        )
        shutil.copyfile(PLACEHOLDER, target)
        return

    target.write_bytes(preview)
    crop_to_thumbnail(target)


def discover_notebooks() -> dict[str | None, list[Path]]:
    """
    Group every notebook under ``examples/`` by the section its folder puts it in.

    Returns
    -------
    grouped : dict mapping str or None to list of Path
        Notebook paths keyed by section, where None holds the notebooks sitting directly in
        ``examples/``.
    """
    if not NOTEBOOKS_ROOT.is_dir():
        return {}

    grouped: dict[str | None, list[Path]] = {}
    for path in sorted(NOTEBOOKS_ROOT.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue

        parts = path.relative_to(NOTEBOOKS_ROOT).parts
        grouped.setdefault(parts[0] if len(parts) > 1 else None, []).append(path)

    return grouped


def ordered_sections(grouped: dict[str | None, list[Path]]) -> list[tuple[str | None, list[Path]]]:
    """
    Order the sections for the page, unsectioned notebooks first.

    Parameters
    ----------
    grouped : dict mapping str or None to list of Path
        Sections as :func:`discover_notebooks` returns them.

    Returns
    -------
    sections : list of tuple of str or None and list of Path
        Section title and its notebooks, in the order they appear on the page.
    """
    ranking = {section: index for index, section in enumerate(SECTION_ORDER)}

    def position(section: str | None) -> tuple[bool, int, str]:
        if section is None:
            return False, 0, ""

        return True, ranking.get(section, len(ranking)), section

    return [
        (None if section is None else SECTION_TITLES.get(section, section.replace("_", " ").title()), grouped[section])
        for section in sorted(grouped, key=position)
    ]


def render_page(sections: list[tuple[str | None, list[Path]]], staged: Path, thumbnails: Path) -> str:
    """
    Stage each notebook, write its thumbnail, and render the gallery page.

    Parameters
    ----------
    sections : list of tuple of str or None and list of Path
        Sections as :func:`ordered_sections` returns them.
    staged : Path
        Directory the notebook copies are written into.
    thumbnails : Path
        Directory the thumbnails are written into.

    Returns
    -------
    page : str
        The gallery page, or an empty string when no notebook survived the git filter.
    """
    toctree: list[str] = []
    body: list[str] = []

    for title, notebooks in sections:
        cards: list[str] = []
        for notebook in notebooks:
            if not is_tracked_by_git(notebook):
                logger.info(f"Skipping {notebook.name}, which git does not track")
                continue

            # Staged under the folders it came from, so two notebooks sharing a name in different
            # sections cannot collide.
            folder = notebook.parent.relative_to(NOTEBOOKS_ROOT)
            document = f"{folder}/{notebook.stem}" if folder.parts else notebook.stem

            staged_notebook = staged / f"{document}.ipynb"
            staged_notebook.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(notebook, staged_notebook)
            write_thumbnail(notebook, thumbnails / f"{document}.png")

            toctree.append(document)
            cards.append(ITEM_TEMPLATE.format(document=document, image=f"/_thumbnails/{document}.png"))

        if not cards:
            continue

        if title is None:
            body.append(GRID_HEAD)
        else:
            body.append(
                SECTION_TEMPLATE.format(
                    section_title=title,
                    section_id=title.lower().replace(" ", "-"),
                    underlines="-" * len(title),
                )
            )
        body.extend(cards)

    if not toctree:
        return ""

    lines = [PAGE_TITLE.format(title=GALLERY_TITLE, underlines="=" * len(GALLERY_TITLE)), TOCTREE_HEAD]
    lines.extend(f"   {entry}\n" for entry in toctree)
    lines.append("\n")
    lines.extend(body)

    return "\n".join(lines)


def build_gallery(app) -> None:
    """
    Stage the notebooks and write the gallery page, or write nothing when there are none.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The application, read for its source directory.
    """
    source = Path(app.builder.srcdir)
    staged = source / "examples"
    thumbnails = source / "_thumbnails"
    staged.mkdir(parents=True, exist_ok=True)
    thumbnails.mkdir(parents=True, exist_ok=True)

    page = render_page(ordered_sections(discover_notebooks()), staged, thumbnails)
    if not page:
        # No notebooks is the expected state until the gallery has content, and a page written here
        # would be a toctree entry pointing at nothing.
        logger.info("No notebooks under examples/, so no gallery page was written")
        return

    (staged / f"{GALLERY_DOCUMENT}.rst").write_text(page, encoding="utf-8")
    logger.info(f"Wrote the gallery to examples/{GALLERY_DOCUMENT}.rst")


def setup(app):
    app.connect("builder-inited", build_gallery)

    return {"parallel_read_safe": True, "parallel_write_safe": True}
