Working on the documentation
============================

This page documents the documentation: where the files are, what is written by hand against what a
build generates, and how to add each kind of page.

Building
--------

.. code-block:: bash

    pixi run docs-build    # docs/source -> docs/build/html
    pixi run docs-serve    # the same, served on :8000 and rebuilt on edit

Both run in the ``docs`` environment, which carries Sphinx and its extensions and not the package's
own development tools. A new documentation dependency goes in the ``docs`` feature of
``pyproject.toml``, and so does any task that needs it. A task declared outside the feature that
owns its dependencies cannot find them.

Nothing in CI builds the site. Read the Docs is the only other builder, so a broken build is caught
after a merge unless you run it before pushing.

Layout
------

Source content lives under ``docs/source/``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Path
     - Purpose
   * - ``index.rst``
     - Landing page and the root toctree.
   * - ``get_started/``
     - Install, quickstart, the ``cache_dir`` contract, data acquisition. Hand-written narrative.
   * - ``user_guide/``
     - Concept pages. Plain MyST, no Sphinx-only syntax. See below.
   * - ``dev/``
     - This page, the contributing guide and the style guide.
   * - ``api.rst`` and ``api/*.rst``
     - Autosummary entry points, one file per public subpackage.
   * - ``api/**/generated/``
     - One stub per documented object. **Generated** at build time.
   * - ``api/**/classmethods/``
     - Method pages, kept out of the global toctree by ``remove_from_toctrees``. **Generated**.
   * - ``_templates/autosummary/``
     - The autosummary class template that produces per-method subpages.
   * - ``examples/``
     - Notebooks staged from the repository-root ``examples/`` tree, and the gallery page listing
       them. **Generated**.
   * - ``_thumbnails/``
     - One image per gallery card. **Generated**.

Everything marked **Generated** is written back into ``source/`` by a build and listed in
``docs/.gitignore``. Never commit any of it, and never edit a generated stub, because the next
build overwrites it. ``docs-serve`` also has to ignore every generated path, or writing one
retriggers the watcher and the build never settles.

Adding an API page
------------------

The API reference is curated, not swept. Each ``source/api/<subpackage>.rst`` holds an
``autosummary`` directive with ``:toctree: generated/``, listing its subpackage's public names one
per line, fully qualified. Adding a public symbol means adding a line to the page for its
subpackage. ``source/api/data.rst`` is the one to copy the shape from.

The directive itself cannot be quoted here. autosummary scans source files for it by pattern, and
the scan does not notice that a match is inside a code block, so an example would generate a stray
stub page for whatever it named.

Nothing checks this. The build does not warn about an undocumented public name, because numpydoc
validates only when asked to and it is not asked to. The package is small enough that the check is
done by hand when a pull request adds public surface: every name in ``climate_risk.__all__`` needs a
page and a docstring.

``autodoc_typehints`` is ``"none"``. numpydoc renders parameter types out of the docstring, so
letting autodoc render the annotations too prints every signature twice. Types belong in the
docstring's ``Parameters`` section, and every return value is named, as in ``panel : DataFrame``,
even where the function's caller never binds it.

Adding a gallery notebook
-------------------------

Gallery notebooks live in the repository-root ``examples/`` tree, not under ``docs/source/``. The
path a notebook sits at decides where its card lands: ``examples/<section>/name.ipynb`` puts it
under a heading named after the folder, and ``examples/name.ipynb`` puts it on the page with no
heading. ``SECTION_ORDER`` in ``docs/sphinxext/generate_gallery.py`` pins the order of the sections
that have one, anything unlisted follows alphabetically, and ``SECTION_TITLES`` in the same file
overrides the folder name where title-casing it reads wrong.

Commit the notebook. The extension stages only files git tracks, so a draft left untracked under
``examples/`` stays off the published site.

Commit it executed, with its outputs saved. ``nb_execution_mode`` is ``"off"``, so nothing runs a
notebook at build time and an unexecuted one renders as a page of empty cells. The last
``image/png`` output becomes the card thumbnail, center-cropped to a square with a border, which
makes the closing figure worth choosing deliberately. A notebook carrying no image output gets a
placeholder and a build warning naming it.

A build stages the notebooks into ``source/examples/`` and writes their thumbnails to
``source/_thumbnails/``. Sphinx renders the staged copy, so editing one changes nothing.

**The first notebook to land also adds ``examples/gallery`` to the root toctree in ``index.rst``.**
With no notebooks the extension writes no gallery page, and a toctree entry naming a page nothing
wrote fails the build.

``tests/test_generate_gallery.py`` covers the extension. It has no sister module, because
``docs/sphinxext/`` is build tooling rather than package code, and the file states that reason.

Writing a page
--------------

Pages are reStructuredText or MyST markdown. ``source_suffix`` accepts ``.rst``, ``.md`` and
``.ipynb``, and the markdown ones go through myst-nb. Enabled MyST extensions are ``colon_fence``,
``deflist``, ``dollarmath``, ``amsmath`` and ``substitution``.

**One rule constrains ``user_guide/`` specifically: those pages carry no Sphinx-only syntax.** No
``{eval-rst}``, no ``{jupyter-execute}``, no roles. Standard markdown, fenced code blocks, and
dollar-math only. These pages have to stay legible as plain text, on GitHub or in an editor or
anywhere the source is read directly, and a page full of roles is not. Cross-reference them with
plain markdown links, which read correctly either way.

Elsewhere the full role set is available. Cross-reference code with ``:func:``, ``:class:`` and
``:meth:``, which resolve into the generated API pages, and other documents with ``:doc:``.

Intersphinx and cross-references
--------------------------------

``intersphinx_mapping`` covers Python, NumPy, SciPy, pandas, xarray, geopandas, PyMC, PyTensor and
ArviZ. ``intersphinx_timeout`` is 15 seconds: an unreachable or throttled documentation host
otherwise stalls the whole build, and with a timeout its cross-references degrade to plain text and
the build finishes.

``autosectionlabel_prefix_document`` is on, so a section label carries its document path and the same
heading in two files does not collide.

Versions and hosting
--------------------

The version in the sidebar comes from installed package metadata, not from ``climate_risk``, which
does not re-export ``__version__``. hatch-vcs writes ``climate_risk/_version.py`` at build time from
git tags and does not track it.

Read the Docs builds through pixi. ``.readthedocs.yaml`` overrides ``build.commands`` entirely:
mamba installs pixi, pixi installs the ``docs`` environment from the committed lock, and pixi runs
Sphinx. Read the Docs therefore builds no environment of its own. The ``sphinx``, ``python`` and
``conda`` keys must stay absent from that file, and the checkout is unshallowed first so the tags
hatch-vcs needs are present.

Checking links
--------------

.. code-block:: bash

    cd docs && make linkcheck

Worth running against the getting-started pages in particular: they link to every upstream data
publisher, and a dead link there is a user blocked before they have loaded anything.
