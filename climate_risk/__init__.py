import logging

from climate_risk.data_functions import (
    build_country_panel,
    build_time_series,
    load_co2_data,
    load_emdat_events,
    load_gpcc_data,
    load_hadcrut_data,
    load_ocean_heat_data,
    load_rivers_data,
    load_shapefile,
    load_wb_data,
    process_ipcc_scenarios,
    total_precipitation,
)

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

__all__ = [
    "build_country_panel",
    "build_time_series",
    "load_co2_data",
    "load_emdat_events",
    "load_gpcc_data",
    "load_hadcrut_data",
    "load_ocean_heat_data",
    "load_rivers_data",
    "load_shapefile",
    "load_wb_data",
    "process_ipcc_scenarios",
    "total_precipitation",
]
