Contributing
============

Setting up
----------

Everything runs through pixi, which resolves from the committed ``pixi.lock``:

.. code-block:: bash

    git clone https://github.com/jessegrabowski/climate-risk.git
    cd climate-risk
    pixi install
    pixi run pre-commit install

There are three environments. ``default`` carries the package and the development tools, and is what
every task below runs in. ``notebooks`` adds Jupyter. ``docs`` carries Sphinx and its extensions and
nothing else. A dependency needed only to build the documentation goes in the ``docs`` feature of
``pyproject.toml``, and so does any task that uses it. A task declared outside the feature that
owns its dependencies cannot find them.

Tasks
-----

.. code-block:: bash

    pixi run test          # the suite, offline
    pixi run lint          # ruff check and ruff format --check
    pixi run typecheck     # mypy
    pixi run fmt           # ruff format
    pixi run docs-build    # build the site into docs/build/html
    pixi run docs-serve    # build, serve on :8000, and rebuild on edit

``pixi run test-network`` un-skips the tests that hit real hosts. Nothing runs them automatically.

``fetch-geonames``, ``verify-geocoder``, ``verify-placement`` and ``check-corrections`` are the
scripts under ``tools/``, run the same way. They need a real cache and take minutes, which is why
they are scripts rather than tests.

Rebuilding the geocoder
-----------------------

Placing an event on a map needs two archives placed by hand, ``emdat.xlsx`` at the top of the cache
directory and ``gadm_410.gpkg`` under ``gadm/``. :doc:`../get_started/data` has the licenses and the
download pages. The gazetteer downloads itself:

.. code-block:: bash

    pixi run fetch-geonames          # every country EM-DAT names, around 200 dumps
    pixi run fetch-geonames PHL IDN  # or just the ones you need

Each dump is indexed into ``<cache_dir>/geonames/places__iso=XXX.parquet``, one row per distinct
name, the most populous place keeping a name where several share it. Re-running skips whatever is
already there.

To score the result:

.. code-block:: bash

    pixi run verify-geocoder             # how many names have a unit to be scored against
    pixi run verify-geocoder geonames    # score the geocoder against them

The answer key is every written name that already resolves to exactly one GADM unit. A point is
scored as landing in that unit, in the level-1 unit containing it, or somewhere else. Names reaching
more than one unit are excluded, because whichever the geocoder picked the other was available, so
they cannot judge anything.

:doc:`../user_guide/locating-events` explains what the pipeline does with a location before any of
this runs.

Running the tests
-----------------

**The suite is offline, and that is enforced.** An autouse fixture refuses ``connect``,
``connect_ex``, ``sendto``, ``sendmsg`` and ``getaddrinfo`` for every test not marked ``network``. A
test needing upstream data writes the file the loader expects into a ``tmp_path`` cache directory. It
does not mock the transport, because non-blocking connects and UDP sends walk straight past a
patched ``connect``.

**``tests/`` mirrors the package.** Every module has one sister file and no more:

.. code-block:: text

    climate_risk/sample.py                    -> tests/test_sample.py
    climate_risk/data_functions/emdat.py      -> tests/data_functions/test_emdat.py

A test file that does not correspond to a module needs a reason to exist. ``conftest.py`` and genuine
cross-cutting invariants qualify. Files named after a topic rather than a module do not. When a
module is split or renamed, its sister file moves with it.

**Markers.** ``slow`` for PyMC sampling, ``network`` for anything hitting a real host, and
``requires_emdat``, ``requires_gadm``, ``requires_geo_disasters`` and ``requires_ghsl`` for tests
needing a licensed or very large file that CI does not have. Those four skip wherever the file is
absent.

**``xfail`` marks live bugs.** Known bugs are pinned with ``@pytest.mark.xfail`` naming the mechanism
that produces them. ``xfail_strict`` is on, so fixing one turns it into a hard failure and forces the
marker off in the same commit. Do not delete an ``xfail`` to make the suite green.

**Warnings are errors.** ``filterwarnings`` is set to ``error`` with a short allowlist for the
shapefile driver's known complaints. A new warning fails the suite, which is deliberate.

Checks and CI
-------------

``ruff`` formats and lints, and ``mypy`` type-checks the modules named in the ``[tool.mypy]``
``files`` list. That list only grows: a package joins it when it is typed and never leaves, and a
module absent from it is unchecked rather than known-failing.

Pre-commit runs both, plus hooks that reject ``print`` statements, relative imports, and a
``pixi.lock`` out of sync with ``pyproject.toml``. A clean ``pre-commit run --all-files`` is the
cheapest way to keep a pull request green.

CI runs pre-commit, then mypy and the suite, on every pull request. A pull request touching
``docs/**``, ``climate_risk/**``, ``.readthedocs.yaml``, ``pyproject.toml`` or ``pixi.lock`` also
gets a Read the Docs preview link posted as a comment. **Nothing in CI builds the documentation**, so
run ``pixi run docs-build`` yourself before pushing a change that touches it.

For the conventions no tool enforces, see the :doc:`style guide <style_guide>`. The layering rules in
:doc:`../user_guide/layering` are among them. Nothing checks those either, so read them before
adding an import to ``climate_risk.data`` or ``climate_risk.geo``.

Commits and pull requests
-------------------------

**The suite is green at every commit**, not merely at every pull request. A commit that leaves it red
cannot be bisected through or reverted cleanly.

**Every behavior change arrives in the same commit as its tests**, bug fixes included. The commit
that changes a wrong number changes the assertion that documented it being wrong.

**Mechanical churn never shares a commit with a logic change.** A rename commit contains only the
rename.

One logical change per commit. The :doc:`style guide <style_guide>` has the message conventions.

Adding a dependency
-------------------

Every dependency carries a real lower bound, and an upper bound at the next major for anything that
breaks its API across majors. ``*`` is not a version specification. Lower bounds mean "verified
against this", so they track what ``pixi.lock`` resolved. Do not invent a bound that was never
solved.

Adding one means editing ``pyproject.toml`` and running ``pixi lock``. Commit the updated lock in the
same pull request as the change that caused it: a lock that has drifted out of sync fails somewhere
far from the change that broke it.
