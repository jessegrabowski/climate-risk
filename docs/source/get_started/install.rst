Installation
============

``climate_risk`` requires Python 3.12 or newer. It depends on the geospatial stack: GDAL, GEOS and
PROJ, reached by way of ``geopandas``, ``rasterio`` and ``exactextract``. Those libraries have to be
built against a common ABI. Installing from conda-forge is therefore the supported path, and
``pixi`` is how this project does it.

With pixi
---------

.. code-block:: bash

    git clone https://github.com/jessegrabowski/climate-risk.git
    cd climate-risk
    pixi install

That solves the locked environment and installs ``climate_risk`` into it in editable mode. Run
anything through ``pixi run``:

.. code-block:: bash

    pixi run test
    pixi run docs-build

With pip
--------

``climate_risk`` is not published to PyPI, so install it from the repository:

.. code-block:: bash

    pip install git+https://github.com/jessegrabowski/climate-risk.git

Pip will build or fetch wheels for the geospatial dependencies, which succeeds on most platforms
but is the harder path when it does not. If ``rasterio`` or ``geopandas`` fail to install, use
conda-forge rather than fighting the wheels.

Getting the data
----------------

Installing the package does not get you any data. Every loader takes a cache directory as an
explicit argument and downloads into it on first use, and several upstream sources are licensed
and have to be fetched by hand.
