.. _api_data_functions:

Panels and event processing
===========================

.. currentmodule:: climate_risk.data_functions

The assembly layer: EM-DAT events become a filtered, geocoded table, and the merged frames the
models are fitted against are built on top of it.

Analysis frames
---------------

.. autosummary::
    :toctree: generated/

    ~combine_data.build_country_year_panel
    ~combine_data.build_time_series
    ~combine_data.annual_precipitation

Disaster events
---------------

.. autosummary::
    :toctree: generated/

    ~emdat_processing.load_emdat_events
    ~emdat_processing.event_filter
    ~emdat_processing.count_events_by_type
    ~emdat_processing.total_damage
    ~emdat_processing.country_year_grid

Event geography
---------------

.. autosummary::
    :toctree: generated/

    ~emdat_processing.event_geography
    ~emdat_processing.event_units
    ~emdat_processing.named_places
    ~emdat_processing.events_missing_units
    ~emdat_processing.NamedPlace
    ~emdat_processing.UncodedEvent

Boundaries and rivers
---------------------

.. autosummary::
    :toctree: generated/

    ~shapefiles_data_loader.load_shapefile
    ~shapefiles_data_loader.load_place_boundary
    ~shapefiles_data_loader.load_archive
    ~shapefiles_data_loader.download_shapefile
    ~shapefiles_data_loader.extract_shapefiles
    ~shapefiles_data_loader.shapefile_dir
    ~shapefiles_data_loader.repair_iso_codes
    ~rivers_data_loader.load_rivers_data
    ~rivers_data_loader.transform_rivers
    ~rivers_damage.create_hydro_rivers_damage
    ~rivers_damage.create_floods_rivers_damage
