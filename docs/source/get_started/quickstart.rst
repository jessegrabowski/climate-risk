Quickstart
==========

This page runs the library end to end on data it can download for itself. It touches nothing that
needs a license, so it works on a machine that has only just installed the package.

Pick a cache directory
----------------------

Every loader takes a cache directory. Nothing is resolved from the working directory, from an
environment variable, or from a project root, so the choice is yours and it is made once:

.. code-block:: python

    from pathlib import Path

    cache_dir = Path.home() / "climate-risk-data"

That directory does not have to exist. Loaders create it on the first write.

The rules that follow from making the path an argument are in :doc:`cache-directory`.

Load a worldwide series
-----------------------

The atmospheric CO2 record is a single CSV published by NOAA. The first call downloads it, and every
later call reads the copy in the cache:

.. code-block:: python

    import climate_risk as cr

    co2 = cr.load_co2_data(cache_dir)
    print(co2.head())

``load_co2_data`` returns a polars frame of ``year`` and ``co2``. Ocean heat content
(:func:`~climate_risk.data.ocean_heat.load_ocean_heat_data`) and the HadCRUT5 temperature anomaly
(:func:`~climate_risk.data.hadcrut.load_hadcrut_data`) follow the same shape.

:func:`~climate_risk.data_functions.combine_data.build_time_series` merges the worldwide annual
series onto one row per year:

.. code-block:: python

    series = cr.build_time_series(cache_dir)

That call also pulls GPCC precipitation, which is a few gigabytes of gridded NetCDF and takes some
minutes on a cold cache. It is downloaded once.

Load a country panel
--------------------

:func:`~climate_risk.data_functions.combine_data.build_country_year_panel` is the modeling frame:
disaster counts and damages from EM-DAT, development indicators from the World Bank, and annual
precipitation, on one row per country and year.

.. code-block:: python

    panel = cr.build_country_year_panel(cache_dir)

This one needs EM-DAT, which cannot be downloaded by code. On a cache that does not have it the call
raises and tells you where to get the file and where to put it:

.. code-block:: text

    NotImplementedError: No emdat.xlsx was found at `/home/you/climate-risk-data/emdat.xlsx`.
    License: Free for non-commercial use with attribution. Redistribution of the database is not
    permitted; users must download it themselves after registering.
    Obtain it from https://public.emdat.be/ and place it at `/home/you/climate-risk-data/emdat.xlsx`.

Three other sources behave the same way. :doc:`data` lists all four, with licenses and the exact
filename each loader looks for.

Run a country
-------------

A country is a TOML file under ``climate_risk/config/places/``, read by
:func:`~climate_risk.config.registry.load_place`:

.. code-block:: python

    from climate_risk.config.registry import load_place

    lao = load_place("lao")

The returned :class:`~climate_risk.config.schema.CountryConfig` carries the ISO codes, the event
filters and the geometry specification that the loaders and the models read. Adding a country of
your own is one such file and no Python at all. See :doc:`../user_guide/adding-a-country`.

Where to go next
----------------

- :doc:`cache-directory`: why the path is an argument, and what lives under it.
- :doc:`data`: every upstream source, what downloads itself, and what does not.
- :doc:`../user_guide/index`: the data layer, place configuration, and the design decisions
  behind them.
- :doc:`../api`: the full public surface.
