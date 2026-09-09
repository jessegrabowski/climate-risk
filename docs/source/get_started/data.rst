Getting the data
================

``climate_risk`` ships no data. Every source is fetched into the :doc:`cache directory
<cache-directory>` you name, and most of them are fetched by the library itself the first time
you ask for them. Four are not, because their publishers forbid automated download. Those four are
the wall a new user hits, so they come first.

Sources you must obtain by hand
-------------------------------

Each of these is declared as a :class:`~climate_risk.data.source.ManualSource`. Nothing downloads
them. A loader that needs one and does not find it raises
:class:`NotImplementedError` naming the license, the homepage, and the exact path the file must be
placed at.

.. list-table::
   :header-rows: 1
   :widths: 14 26 24 18 18

   * - Source
     - What it is
     - license
     - Place it at
     - Read by
   * - EM-DAT
     - The disaster event record: one row per recorded event, with damages and dates. The panel
       is built from this.
     - Free for non-commercial use with attribution. Redistribution is not permitted; users must
       register and download it themselves.
     - ``<cache_dir>/emdat.xlsx``
     - :func:`~climate_risk.data_functions.emdat_processing.load_emdat_events`
   * - GADM 4.1
     - Administrative boundaries worldwide, to level 4. Used to place events and to aggregate onto
       units.
     - Academic and other non-commercial use only. Redistribution, and use in a commercial product
       or service, require prior written permission.
     - ``<cache_dir>/gadm/gadm_410.gpkg``
     - :func:`~climate_risk.data.gadm.load_admin_units`
   * - Geo-Disasters
     - Geocoded EM-DAT footprints, 1990-2023, on GAUL 2015 geometries.
     - Geometries are (c) FAO 2015 under the GAUL 2015 Data License, non-commercial with
       attribution. Non-spatial attributes are CC BY 4.0.
     - ``<cache_dir>/geo_disasters/disaster_subnational_90_23.gpkg``
     - :func:`~climate_risk.data.geo_disasters.load_event_footprints`
   * - Penn World Table 10.0
     - Cross-country national accounts: output, capital stock, employment and prices.
     - CC BY 4.0.
     - ``<cache_dir>/pwt100.xlsx``
     - :func:`~climate_risk.data.pwt.load_pwt_data`

Where to obtain each:

- EM-DAT: https://public.emdat.be/, after registering for an account. Export the full record to
  ``xlsx``.
- GADM: https://gadm.org/download_world.html, the "Geopackage" download of version 4.1.
- Geo-Disasters: https://doi.org/10.5281/zenodo.15487667.
- Penn World Table: https://www.rug.nl/ggdc/productivity/pwt/, version 10.0.

Rename the download to the filename in the table. The loaders look for that exact name, and PWT in
particular publishes under a version-stamped name that does not match.

Sources that download themselves
--------------------------------

These are declared as :class:`~climate_risk.data.source.DataSource` and fetched on first use by
:func:`~climate_risk.data.fetch.fetch`. No account, no key, nothing to place by hand.

.. list-table::
   :header-rows: 1
   :widths: 18 34 16 32

   * - Source
     - What it is
     - license
     - Loader
   * - NOAA CO2
     - Annual mean atmospheric CO2 at Mauna Loa.
     - Public domain
     - :func:`~climate_risk.data.co2.load_co2_data`
   * - NOAA ocean heat
     - Seasonal ocean heat content, 0-700 m.
     - Public domain
     - :func:`~climate_risk.data.ocean_heat.load_ocean_heat_data`
   * - HadCRUT5
     - Gridded surface temperature anomalies, ensemble mean.
     - Open Government license v3
     - :func:`~climate_risk.data.hadcrut.load_hadcrut_data`
   * - GPCC
     - Gridded monthly precipitation from gauges: the reanalyzed record to 2020, continued by the
       near-real-time monitoring product.
     - CC BY 4.0
     - :func:`~climate_risk.data.gpcc.load_gpcc_data`
   * - GeoNames
     - The place-name gazetteer, per country, used to geocode event locations.
     - CC BY 4.0
     - :func:`~climate_risk.data.geonames.load_place_points`
   * - HydroRIVERS v1.0
     - River network geometries with stream order.
     - HydroSHEDS License Agreement
     - :func:`~climate_risk.data_functions.rivers_data_loader.load_rivers_data`
   * - World Bank boundaries
     - Country polygons at 1:10m.
     - CC BY 4.0
     - :func:`~climate_risk.data_functions.shapefiles_data_loader.load_shapefile`
   * - GSHHG 2.3.7
     - Global coastlines.
     - LGPL
     - :func:`~climate_risk.data_functions.shapefiles_data_loader.load_shapefile`
   * - GHS-POP R2023A
     - Global population rasters at 30 arcsecond resolution, one per five-year epoch from 1975.
     - European Commission reuse notice, attribution required
     - :func:`~climate_risk.data.ghsl.population_on_cells`

GPCC and GHS-POP are the large ones. GPCC is fetched an archive per decade and a further archive per
month after 2020, and GHS-POP is roughly a gigabyte per epoch. A country panel needs GPCC but not
GHS-POP, so a cache can stay small as long as the geospatial models are not being run.

Services queried live
---------------------

Four sources answer a query instead of serving a file, and are declared as
:class:`~climate_risk.data.source.ApiSource`. The response is cached like anything else, so a warm
run does not call them again.

.. list-table::
   :header-rows: 1
   :widths: 18 34 16 32

   * - Service
     - What it is
     - license
     - Loader
   * - World Bank Indicators
     - Development indicators: population, GDP, urbanization and the rest of the country-year
       panel's covariates.
     - CC BY 4.0
     - :func:`~climate_risk.data.world_bank.load_wb_data`
   * - FRED
     - The foreign block, meaning the external conditions a small open economy takes as given:
       US and world macroeconomic series.
     - Terms of the St. Louis Fed
     - :func:`~climate_risk.data.fred.load_fred_data`
   * - IMF IMTS
     - Bilateral merchandise trade, used to build partner activity.
     - (c) International Monetary Fund
     - :func:`~climate_risk.data.partner_activity.load_partner_activity`
   * - Nominatim
     - OpenStreetMap geocoding, as a fallback when the gazetteer cannot place a name.
     - ODbL 1.0
     - :func:`~climate_risk.data.osm.osm_geocoder`

Nominatim's usage policy caps requests at one per second, and the loader honors it. Geocoding a
country's events for the first time therefore takes as long as it takes.

Shipped with the package
------------------------

One file is redistributed inside the wheel: the IPCC AR6 SYR CSB.2 Figure 1(a) emissions workbook,
read by :func:`~climate_risk.data.ipcc.process_ipcc_scenarios`. It is CC BY 4.0, and it is vendored
because SEDAC, which published it, was decommissioned in June 2025 and no successor serves the file
to an unattended client. Its full attribution is in ``climate_risk/data/vendored/ATTRIBUTION.md``.

Attribution
-----------

The licenses above are the publishers', not this package's. Redistributing anything derived from
EM-DAT, GADM or the GAUL geometries in Geo-Disasters is restricted, and the rest require
attribution. Every source carries its citation string on its declaration, as ``CO2.citation`` or
``GADM.citation``, so the text to credit is available from the object the loader used.
