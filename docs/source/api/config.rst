.. _api_config:

Places and regions
==================

.. currentmodule:: climate_risk.config

A country is one TOML file in ``climate_risk/config/places/``, and a region is a list of countries
in ``climate_risk/config/regions/``. Nothing in the library names a country in code.

Reading a place
---------------

.. autosummary::
    :toctree: generated/

    ~registry.load_place
    ~registry.read_place
    ~registry.resolve_isos
    ~registry.all_event_location_overrides

Configuration schema
--------------------

.. autosummary::
    :toctree: generated/

    ~schema.CountryConfig
    ~schema.RegionConfig
    ~schema.EventFilters
    ~schema.GeometrySpec
