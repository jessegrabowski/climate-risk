.. _api_data:

Data sources and loaders
========================

.. currentmodule:: climate_risk.data

Every loader takes the cache directory and returns a frame. It comes first, except where a loader
is scoped to one country, shapefile or epoch, which takes that first: ``load_shapefile("world",
cache_dir)``. There is no default cache directory and no environment variable, so the path is
resolved once at the edge and passed down.

Loaders
-------

These fetch on a cold cache, read from disk on a warm one, and re-fetch under ``force_reload=True``.

.. autosummary::
    :toctree: generated/

    ~co2.load_co2_data
    ~ocean_heat.load_ocean_heat_data
    ~hadcrut.load_hadcrut_data
    ~gpcc.load_gpcc_data
    ~ipcc.process_ipcc_scenarios
    ~world_bank.load_wb_data
    ~fred.load_fred_data
    ~ilo.load_labour_income_share
    ~deflate.load_price_level
    ~partner_activity.load_partner_activity
    ~ghsl.population_on_cells

Loaders for manually obtained sources
-------------------------------------

These four upstreams are published under terms that forbid automated download. Nothing in the
library fetches them: the file has to be placed in the cache by hand, and the loader raises with
the license and the expected path until it is there.

.. autosummary::
    :toctree: generated/

    ~pwt.load_pwt_data
    ~gadm.load_admin_units
    ~gadm.load_units_in_country
    ~geo_disasters.load_event_locations
    ~geo_disasters.load_event_footprints
    ~geo_disasters.load_resolved_units

Pure transforms
---------------

The parsing half of each loader, separated from fetching so it can be exercised without the
network. Each takes what the loader read from disk, plus any reference data the shaping needs, and
returns the frame the loader caches.

.. autosummary::
    :toctree: generated/

    ~co2.transform_co2
    ~ocean_heat.transform_ocean_heat
    ~hadcrut.transform_hadcrut
    ~gpcc.transform_gpcc
    ~ipcc.transform_ipcc
    ~world_bank.transform_world_bank
    ~fred.transform_fred
    ~ilo.transform_labour_share
    ~partner_activity.transform_partner_activity
    ~pwt.transform_pwt

Constant dollars
----------------

A money column is comparable across years only once the price level is divided out of it. Use
:func:`~deflate.deflate` on amounts measured in the dollars of the year beside them, and
:func:`~deflate.rebase` on amounts already held constant in one year's dollars, as EM-DAT's
adjusted damages are.

.. autosummary::
    :toctree: generated/

    ~deflate.annual_price_level
    ~deflate.deflation_factors
    ~deflate.deflate
    ~deflate.rebase

Declaring a source
------------------

A source is declared beside the loader that reads it, carrying its license and citation. The four
kinds differ in what they can do: :class:`~source.DataSource` is fetchable,
:class:`~source.ManualSource` has no URL because automated download is
forbidden, :class:`~source.ApiSource` answers queries rather than serving a
file, and :class:`~source.VendoredSource` ships inside the package because
nothing serves it any more.

.. autosummary::
    :toctree: generated/

    ~source.DataSource
    ~source.ManualSource
    ~source.ApiSource
    ~source.VendoredSource
    ~source.ShapefileArchive
    ~fetch.fetch

Caching
-------

.. autosummary::
    :toctree: generated/

    ~cache.cached
    ~cache.CacheFormat
    ~cache.pandas_parquet
    ~cache.polars_parquet
    ~cache.geo_parquet
    ~cache.cache_key
    ~cache.builder_fingerprint

Cache layout
------------

Where each source lives inside the cache directory.

.. autosummary::
    :toctree: generated/

    ~gadm.gadm_dir
    ~gadm.gadm_path
    ~geo_disasters.geo_disasters_dir
    ~geo_disasters.geo_disasters_path
    ~geonames.geonames_dir
    ~ghsl.ghsl_dir
    ~osm.osm_dir

Resolving place names
---------------------

EM-DAT records locations as free text. These turn that text into administrative units, and rank
candidates when one name reaches more than one place.

.. autosummary::
    :toctree: generated/

    ~place_names.Gazetteer
    ~place_names.read_gazetteer
    ~place_names.resolve_place
    ~place_names.resolve_event_places
    ~place_names.Placement
    ~place_names.Unit
    ~place_names.match_key
    ~place_names.name_shapes
    ~place_names.names_no_unit
    ~place_names.nearest_name
    ~place_names.container_parts
    ~place_names.successor_state
    ~place_names.repair_mojibake
    ~place_names.read_name_corrections
    ~place_names.keying_fingerprint

Geocoders
---------

A geocoder maps a place name to points. :func:`~placement.available_geocoders` yields the point
sources that can answer for a country, most trusted first, and the scoring helpers turn candidate
points into units.

.. autosummary::
    :toctree: generated/

    ~placement.available_geocoders
    ~geonames.geonames_geocoder
    ~geonames.load_place_points
    ~geonames.country_dump
    ~geonames.read_country_codes
    ~osm.osm_geocoder
    ~osm.search_nominatim
    ~osm.read_osm_places
    ~osm.record_lookups
    ~osm.NominatimAnswer
    ~geocoding.score_geocoder
    ~geocoding.ScoredName
    ~geocoding.units_containing_points
    ~geocoding.units_from_geocoders
    ~geocoding.unambiguous_units

Administrative units and event footprints
-----------------------------------------

.. autosummary::
    :toctree: generated/

    ~gadm.administered_territories
    ~geo_disasters.resolve_to_gadm
    ~geo_disasters.compare_event_units
    ~geo_disasters.event_unit_ids
    ~geo_disasters.unit_names
    ~geo_disasters.normalize_unit_name
    ~ghsl.population_source
    ~ghsl.population_raster
    ~gpcc.GriddedProduct
    ~gpcc.coverage_of
    ~model_frame.event_windows
