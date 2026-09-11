climate_risk
============

Bayesian disaster-frequency and damage modeling for climate adaptation policy.

Quick install
-------------

.. code-block:: bash

    pixi install

See the :doc:`installation guide <get_started/install>` for the pip path and the geospatial
dependencies.

Quick example
-------------

.. code-block:: python

    from pathlib import Path

    import climate_risk as cr

    # The cache directory is always explicit. There is no default and no environment variable.
    cache_dir = Path("data")

    co2 = cr.load_co2_data(cache_dir)
    panel = cr.build_country_year_panel(cache_dir)

``load_co2_data`` downloads its source on the first call and reads the cache on every later one, so
a warm run never touches the network. Loaders whose upstream forbids automated download raise
instead, and tell you where to obtain the file.

.. toctree::
   :maxdepth: 1
   :hidden:
   :titlesonly:

   get_started/index
   user_guide/index
   examples/gallery
   api
   dev/index
