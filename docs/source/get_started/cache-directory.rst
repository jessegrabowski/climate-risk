The cache directory
===================

Every function in ``climate_risk`` that reads or writes data takes a ``cache_dir`` argument. There
is no default value, no environment variable, and no search for a project root. This page says what
that means in practice and why it is the way it is.

The contract
------------

.. code-block:: python

    from pathlib import Path

    import climate_risk as cr

    cache_dir = Path("/data/climate-risk")

    co2 = cr.load_co2_data(cache_dir)
    panel = cr.build_country_year_panel(cache_dir)

Making the argument required has three consequences.

**One process can work against several caches.** Comparing a run against a frozen snapshot is
passing a different path, not editing a configuration file or exporting a variable and restarting.

**Nothing writes to a directory you did not name.** A loader creates ``cache_dir`` and its
subdirectories if they are absent, but it never writes outside the path it was handed.

**The path is resolved once, at the edge.** A script or notebook decides where the cache lives on
its first line and threads the value down. Library code never asks where it is, because there is
nobody to ask.

If threading the argument through a deep call stack becomes tedious, bind it once with
:func:`functools.partial` or a small wrapper of your own. Adding a default to the library is not the
answer.

Why it is an argument
---------------------

The alternative is an ambient root, discovered by walking up from ``__file__`` or read from an
environment variable. That makes every function's behavior depend on state the caller cannot see in
the call. Two runs of the same script then differ for reasons that do not appear in the script, and
a test that forgets to isolate the variable writes into the developer's real cache. The cache here
runs to tens of gigabytes, so that failure is expensive rather than merely confusing.

What lives under it
-------------------

Raw downloads keep the filename their publisher uses, at the top of the cache:

.. code-block:: text

    co2_annmean_mlo.csv
    HadCRUT.5.0.2.0.analysis.anomalies.ensemble_mean.nc
    emdat.xlsx

Sources that unpack into many files, or that are fetched per country or per epoch, get a
subdirectory of their own: ``gadm/``, ``geo_disasters/``, ``geonames/``, ``ghsl/``, ``gpcc/``,
``osm/``, ``rivers/`` and ``shapefiles/``.

Derived frames, meaning anything a loader computed rather than downloaded, are stored as parquet
under a name built from what was asked for. :func:`~climate_risk.data.cache.cache_key` builds the
stem, so ``points`` requested with a grid size of 400 over Southeast Asia is stored as:

.. code-block:: text

    points__grid_size=400__region=sea.parquet

Two calls that ask for the same thing therefore hit the same entry however the caller ordered the
parameters, and two calls that ask for different things cannot collide.

Refreshing an entry
-------------------

Loaders that download take ``force_reload``:

.. code-block:: python

    co2 = cr.load_co2_data(cache_dir, force_reload=True)

That re-downloads the source and rebuilds anything derived from it. Deleting the file works equally
well, because the cache holds no index and its state is exactly what is on disk.

An entry whose *builder* has changed turns over on its own. Several loaders key their cached frame
on :func:`~climate_risk.data.cache.builder_fingerprint`, a digest of the building function's source
together with the rules it reads, so editing the transformation writes a new entry rather than
reading back a stale one.

Interrupted downloads
---------------------

:func:`~climate_risk.data.fetch.fetch` writes to a ``.part`` file beside the destination and moves it
into place only when the transfer is complete, so a cancelled download can never be read back as a
cache hit. A dropped connection is retried, continuing from the bytes already on disk, which matters
for the multi-hundred-megabyte rasters. A ``.part`` file left behind by an earlier call is discarded
rather than resumed, because nothing proves it is a prefix of the file being fetched now.

How large it gets
-----------------

A full cache holds every source, worldwide, at every GHS-POP epoch, and runs to tens of gigabytes.
GPCC and GHS-POP are most of that. Leaving those two out brings it to a few hundred megabytes. Nothing
prunes the cache, so the directory only grows.
