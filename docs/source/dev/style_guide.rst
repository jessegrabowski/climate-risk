Style Guide
===========

``ruff`` settles anything mechanical: formatting, import order, the 120-column line. CI runs it, so
``pixi run lint`` decides those questions before review does. This page covers the rest: the
judgment the tools cannot make.

Most of it you will get right by writing code that looks like the file around it.

Code
----

Prefer vectorized operations to loops, and clarity to micro-optimization. A loader runs over a
worldwide panel, so an invariant recomputed inside a loop is paid for once per country per year, but
a clever line nobody can read is paid for by every future maintainer.

Errors should be specific and loud. A bare ``except:``, or an ``except Exception: pass`` that
swallows a failure, turns a bug into a mystery three layers away. Validate where a wrong value would
otherwise surface as a confusing error deep in a pipeline, and nowhere else. A check that cannot
fire is noise.

Where you do raise, name the fix in the message. It reaches the reader at the moment they need it,
which no amount of documentation does. :meth:`~climate_risk.data.source.ManualSource.require` is the
model: it names the license, the homepage, and the exact path the file has to be placed at.

.. code-block:: python

    raise NotImplementedError(
        f"No {self.filename} was found at `{path}`.\n"
        f"License: {self.license}\n"
        f"Obtain it from {self.homepage} and place it at `{path}`."
    )

Design
------

Write the simplest thing that works. Generality for a future that has not arrived is a cost paid now
against a benefit that may never come. Two pieces of code that look alike today but change for
different reasons are not duplication, and forcing them together couples them wrongly, so the rule of
three is a good prior before extracting a helper.

Keep one function at one altitude: raw string-mangling next to high-level orchestration means a
helper is missing. Be consistent about failure, too. One module should not raise for some errors,
return ``None`` for others, and return a sentinel for a third kind of the same failure.

The usual anti-patterns to watch for:

* A boolean flag that makes one function do two things, and the ``do_thing(True, False)`` call site
  it leads to.
* Parameter lists too long to hold in your head.
* A dict standing in for an object, where a dataclass would say what the fields are.
* Mutable default arguments, and hidden global state.
* A function that both computes and mutates.

Two rules here are specific to this package, and both are easy to violate with good intentions.

**Constants live next to their consumer.** There is no constants module and reintroducing one is a
regression. A URL sits in the module of the loader that fetches it, a column mapping sits in the
module that renames the columns. A constant genuinely shared by two modules goes in the one that owns
the concept, and the other imports it.

**Path resolution is functional.** ``cache_dir`` is an argument, resolved once at the edge and
threaded down, and the package reads no environment variables at all. Do not add a default: see
:doc:`../get_started/cache-directory` for why.

Naming and shape
----------------

A name should say what something is for, so the code reads as its own documentation. Avoid names
that describe a type rather than a role: ``data``, ``tmp``, ``obj``, ``result2``. Single letters are
fine where they are the mathematical convention and nowhere else. Name the same concept the same way
the surrounding code does, and name the constants. A bare ``0.9`` tells the reader nothing where
``decay`` tells them what it is for.

These conventions are settled and do not vary:

* ``lon``, never ``long``.
* ``gpd``, never ``geo``.
* ``cache_dir`` for every path argument, never ``data_path``, ``output_path`` or ``folder_path``.
* Function names lead with the verb.

Shape carries meaning too. Prefer guard clauses to nesting, so the happy path stays prominent and
error handling sits at the edges. Break up expression soup with named intermediates, since the name
*is* the documentation and costs one line. When two branches do the same kind of thing, shape them
the same way, so an asymmetry signals a real difference rather than drift. Within a module, read top
down: public API first, helpers below it, reading order roughly matching call order.

Pass arguments by keyword wherever the keyword carries information. Omit it only where it adds
nothing, as in ``np.add(a, b)``.

Comments
--------

Comments explain the **why**. The code already says what it does. Fewer is better, because every
comment can drift out of sync and so has to change a reader's understanding to earn its place. A
better name usually beats a comment.

**No changelog in source.** Not in comments, not in docstrings, not in config files: a
``.gitignore``, a ``pyproject.toml`` and a YAML hook config are source too. The failure mode is
writing for *the person reviewing this change* rather than for the next reader of the file, and
reviewer-facing prose belongs in the commit body. Check a comment against these before writing it --
any one of them means cut it or move it:

* It contrasts the code with an earlier version or a road not taken: "rather than", "instead of",
  "X does not", "still works", "previously". Prose may draw a contrast that informs the reader. A
  comment doing it is almost always addressing the reviewer.
* It justifies a decision instead of stating a fact the code cannot state itself.
* It cites a measurement, size, or date taken while doing the work.

The test: **would this still make sense to someone who cloned the repository today and has never
seen an earlier version?** What survives is short and factual: what a thing is for, or a constraint
that is invisible from the code. One line, usually.

Do not commit comments that narrate the code (``# increment the counter``), commented-out code, or a
``TODO`` with no owner and no context. Reflow comment prose to the full 120 columns.

Docstrings
----------

Docstrings are numpydoc, and they document the **current contract**. Write each one as if the
function appeared in the codebase fresh today. A reader who cloned the repository an hour ago should
never meet a sentence that only makes sense if they know what the code used to do. The no-changelog
rule above applies here in full, and a "Notes" section justifying a recent change is its commonest
disguise.

The rest of the rules:

* **Active voice.** "Compute the gradient", not "The gradient is computed".
* **Every parameter gets a human-readable type**: ``list of int``, not ``list[int]``. Describe a
  genuinely nested type in prose rather than pasting a type hint into the docstring.
* **Optional arguments say so**, with ``, optional`` on the type line, and the **default goes in
  the last sentence** of the description rather than on the type line.
* **Return values are named**, even when nothing ever binds them: ``panel : DataFrame``.
* **No Raises sections.** The error message is the documentation.
* **No module-level docstrings.** If a module's purpose is not evident from its name and contents,
  the fix is a better name or a split.
* **Math goes in** ``.. math::`` **directives**, never as backticked ASCII. Use a raw string so the
  backslashes survive. Inline, a mathematical symbol takes a ``:math:`` role and a code identifier
  takes double backticks.
* Cross-reference with Sphinx roles: ``:func:``, ``:class:``, ``:mod:``.

Only write a docstring if it adds information the code does not. A short private helper whose name
and signature already say everything can go without one. What it cannot have is a chatty paragraph
standing in for a short structured one.

Examples
--------

Every public entry point carries an ``Examples`` section. Three rules:

#. **A lead-in sentence, always**, even when the code looks self-evident. Never open the section with
   the directive. Where two entry points have nearly identical code, the lead-in is the only thing
   telling them apart, so it says what to reach for this one for.
#. ``.. code-block:: python``, **never the** ``>>>`` **prompt.** A prompt cannot be pasted into a
   script.
#. **Complete and runnable on its own**, imports included, small and tight.

Every example takes a cache directory as an argument, because every loader does. ``Path("data")`` is
the conventional stand-in.

.. code-block:: rst

    Examples
    --------
    The first call downloads. Later ones read the cache:

    .. code-block:: python

        from pathlib import Path

        from climate_risk import load_co2_data

        co2 = load_co2_data(Path("data"))

Modern Python
-------------

The supported floor is Python 3.12, so write for it:

* PEP 604 unions (``int | None``) and PEP 585 generics (``list[int]``), not ``Optional`` or
  ``typing.List``.
* **No** ``from __future__ import annotations``. It is unnecessary here.
* f-strings, context managers, ``pathlib`` over ``os.path`` string-mangling, ``enumerate`` and
  ``zip`` over index bookkeeping.
* Comprehensions where they read more clearly than an accumulator loop, and not where they get dense
  enough to obscure what is happening.
* **Imports at the top of the module**, and **never relative**. A pre-commit hook rejects
  ``from .module import name``. A function-local import is warranted only to break a genuine
  circular dependency or to guard an optional dependency.
* Public functions and methods carry type hints. They are read as documentation.

Tests
-----

Test code is code, and everything above applies to it, with one deliberate exception.
**Duplicated setup in tests is usually worth keeping.** A test earns its value by being auditable as
one self-contained block: what was seeded, sampled, patched and asserted, all visible without jumping
to a fixture defined four hundred lines away. Repeated arrange-phase boilerplate is a smaller cost
than a fragmented test, even at six or eight occurrences.

Extract a fixture when the block is long enough to bury the assertion it exists to set up, when it
encodes an invariant that must stay identical across tests, or when a signature change would
otherwise mean editing it everywhere.

Two ways a test here fools you, and both look fine in review. **Asserting what the fixture
wrote**: the warm-cache path of most loaders is a bare
``pd.read_csv``, so asserting its columns or values tests pandas, since the fixture supplied them.
What is genuinely the loader's is narrow, being ``index_col``, ``parse_dates``, which branch it took
and which file it chose. And **deriving fixture paths from the code under test**: a fixture calling
``shapefile_dir(cache_dir)`` writes wherever the loader looks, so the test passes with any cache
path, correct or not. State the layout literally, as ``tmp_path / "shapefiles"``.

Beyond that, the judgment that decides whether a test is worth its runtime:

* **A test must be able to fail for a real reason.** Asserting that an import worked, that ``__all__``
  is non-empty, or that a dataclass has the fields it was declared with tests Python, not this code.
* **Test behavior, not implementation.** Changing how a function works must not require touching its
  test, so long as the contract holds.
* **Coverage is not the goal.** Zero-event countries, empty geometries, boundary years and missing
  lat/lon catch bugs. Exhaustive enumeration does not.
* **Assert on the value, not on its existence.** ``assert result is not None`` and
  ``assert len(out) > 0`` rarely fail when the logic breaks. Bound a quantity on both sides: an
  assertion that a distance is under a thousand kilometers passes just as happily when a unit
  conversion is applied the wrong way.
* **Mutation-test a new fixture before trusting it.** Break the thing it covers, confirm the test
  fails, put it back. If it stays green, it is asserting nothing.

The offline fixture, the markers and ``xfail`` are in :doc:`contributing`.

Commit messages
---------------

Subject line in the imperative, under 60 characters, naming the thing that changed. No "and": a
subject that needs one is two commits.

**Never hard-wrap the body.** One paragraph is one line, however long. Manual line breaks at 72
characters become unreadable the moment anything reflows them, and they fossilize a width nothing
actually uses. Blank lines separate paragraphs, and deliberately structured content keeps its own
line breaks.

A body is worth writing when the *why* is invisible in the diff. It is not worth writing to restate
the diff in prose.

Do not add a ``Co-Authored-By:`` trailer or any AI-attribution footer.

Cruft
-----

The diff should look like it was written by someone who cleaned up after themselves: no leftover
``print`` statements (a pre-commit hook rejects them), no dead code, no unused imports or locals, no
stray scratch files. A file that genuinely should not be tracked belongs in ``.gitignore``, not
deleted and recreated.

Every dependency is a liability. Question a new one that the standard library, NumPy, polars or
geopandas already covers, and see :doc:`contributing` for how a pin is chosen.
