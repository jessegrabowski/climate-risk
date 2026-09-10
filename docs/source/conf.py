import os
import sys

from importlib.metadata import version as installed_version
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))
# The gallery extension is imported by bare module name from here.
sys.path.insert(0, str(root_dir / "docs" / "sphinxext"))

# -- Project information -----------------------------------------------------
project = "climate_risk"
copyright = "2026, Camilo Saldarriaga, Jesse Grabowski, Andrew Walters"
author = "Camilo Saldarriaga, Jesse Grabowski, Andrew Walters"
language = "en"
html_baseurl = "https://climate-risk.readthedocs.io"

# -- Version handling --------------------------------------------------------
# Keeps the RTD version selector labels matching the installed package. The version comes from
# metadata rather than from climate_risk itself: hatch-vcs writes climate_risk/_version.py at build
# time and does not track it, and the package does not re-export __version__.
package_version = installed_version("climate-risk")
version = package_version
on_readthedocs = os.environ.get("READTHEDOCS", None)
rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
if on_readthedocs:
    if rtd_version.lower() == "stable":
        version = package_version.split("+")[0]
    elif rtd_version.lower() == "latest":
        version = "dev"
    else:
        version = rtd_version
else:
    rtd_version = "local"
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    "sphinx_sitemap",
    "notfound.extension",
    "generate_gallery",
]

# Use the document path as prefix for autosectionlabel anchors so the same section title in two
# files doesn't collide.
autosectionlabel_prefix_document = True

templates_path = ["_templates"]

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    # Autosummary templates are Jinja, not documents.
    "*/autosummary/*.rst",
    "Thumbs.db",
    ".DS_Store",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

master_doc = "index"

# -- Autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
# numpydoc renders parameter types from the docstring; letting autodoc render them too prints
# every signature twice.
autodoc_typehints = "none"
autoclass_content = "class"
# Class method pages live under api/.../classmethods/ — keep them out of the global toctree so
# they don't pollute the sidebar.
remove_from_toctrees = ["**/classmethods/*"]

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {
    "of",
    "or",
    "optional",
    "default",
    "numeric",
    "type",
    "scalar",
    "instance",
    "array",
    "array_like",
    "1D",
    "2D",
    "3D",
    "nD",
    "M",
    "N",
    "D",
    "K",
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "climate_risk"
html_short_title = "climate_risk"
html_last_updated_fmt = ""

sitemap_url_scheme = f"{{lang}}{rtd_version}/{{link}}"

html_theme_options = {
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "navbar_start": ["navbar-logo"],
    "show_prev_next": True,
    "icon_links": [
        {
            "url": "https://github.com/jessegrabowski/climate-risk",
            "icon": "fa-brands fa-github",
            "name": "GitHub",
            "type": "fontawesome",
        },
    ],
}

github_version = version if "." in rtd_version else "main"
html_context = {
    "github_url": "https://github.com",
    "github_user": "jessegrabowski",
    "github_repo": "climate-risk",
    "github_version": github_version,
    "doc_path": "docs/source",
    "default_mode": "dark",
}

html_sidebars = {"**": ["sidebar-nav-bs.html", "searchbox.html"]}
html_static_path = ["_static"]

# -- MyST / MyST-NB config ---------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "substitution",
]
myst_dmath_double_inline = True

# myst-nb parses every .md page, and nothing in the docs is meant to run at build time. The data a
# page would need runs to tens of gigabytes, and four of the sources are licensed.
nb_execution_mode = "off"

# A cell emitting text/markdown is parsed as commonmark by default, which has no table syntax.
nb_render_markdown_format = "myst"

# -- Intersphinx -------------------------------------------------------------
# Bound the per-inventory fetch. An unreachable or throttled docs host otherwise stalls the whole
# build; with a timeout its cross-references degrade to plain text and the build finishes.
intersphinx_timeout = 15

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/latest/", None),
    "arviz": ("https://python.arviz.org/en/latest/", None),
}
