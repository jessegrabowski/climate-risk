from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from climate_risk.config.registry import all_event_location_overrides
from climate_risk.config.schema import EventFilters
from climate_risk.data.cache import builder_fingerprint, cached, geo_parquet
from climate_risk.data_functions.emdat_processing import HYDROMETEOROLOGICAL, event_filter, load_emdat_events
from climate_risk.data_functions.rivers_data_loader import load_rivers_data
from climate_risk.data_functions.shapefiles_data_loader import load_shapefile
from climate_risk.geo.crs import to_km
from climate_risk.geo.distance import get_distance_to

DAMAGE_COLUMNS = ("ISO", "End Year", "Latitude", "Longitude", "River Basin", "Total_Damage", "Total_Affected", "Deaths")

# The damage regressions are specified over events reaching more than this many people.
REPLICATION_FILTERS = EventFilters(min_total_affected=1_000)


def _create_rivers_damage(
    cache_dir: Path,
    name: str,
    query: str,
    total_suffix: str,
    log_suffix: str,
    extra_columns: Sequence[str] = (),
    locations: dict[str, tuple[float, float]] | None = None,
) -> gpd.GeoDataFrame:
    """
    Join EM-DAT damages to the nearest major river, and cache the result.

    Parameters
    ----------
    cache_dir : Path
        Directory holding the EM-DAT workbook, the shapefiles and the cached output.
    name : str
        Logical name the result is cached under.
    query : str
        Pandas query selecting the events, applied to the adjusted EM-DAT frame.
    total_suffix : str
        Suffix for the damage and affected totals, giving ``Total_Damage_{total_suffix}``.
    log_suffix : str
        Suffix for their logarithms, giving ``log_damage_{log_suffix}``.
    extra_columns : sequence of str, optional
        Columns to carry through beyond ``DAMAGE_COLUMNS``. Default empty.
    locations : dict mapping str to tuple of float, optional
        Longitude and latitude to force onto specific ``DisNo.`` events. Default None, which forces
        nothing.

    Returns
    -------
    damage : GeoDataFrame
        One row per event, with its distance in kilometers to the nearest major river.
    """
    totals = {"Total_Damage": f"Total_Damage_{total_suffix}", "Total_Affected": f"Total_Affected_{total_suffix}"}

    def build() -> gpd.GeoDataFrame:
        big_rivers = load_rivers_data(cache_dir)
        emdat = load_emdat_events(cache_dir).filter(event_filter(REPLICATION_FILTERS)).to_pandas()
        world = load_shapefile("world", cache_dir, repair_ISO_codes=True)

        events = emdat.query(query)

        if locations is not None:
            for disaster_number, (lon, lat) in locations.items():
                matching = events[events["DisNo."] == disaster_number].index
                events.loc[matching, "Latitude"] = lat
                events.loc[matching, "Longitude"] = lon

        damage_df = gpd.GeoDataFrame(
            (
                events[[*DAMAGE_COLUMNS, *extra_columns]]
                .dropna(how="any", subset=["Latitude", "Longitude"])
                .assign(
                    geometry=lambda x: gpd.points_from_xy(x.Longitude, x.Latitude),
                    year=lambda x: pd.to_datetime(x["End Year"], format="%Y"),
                )
                .drop(columns=["End Year"])
                .replace({0.0: np.nan})
            ),
            crs=world.crs,
        )

        closest_river = get_distance_to(big_rivers, points=damage_df, return_columns=["ORD_FLOW", "HYRIV_ID"]).rename(
            columns={"distance_to_closest": "closest_river"}
        )
        closest_river["closest_river"] = to_km(closest_river["closest_river"])

        damage_df = damage_df.join(closest_river).rename(columns=totals)
        damage_df = damage_df.assign(
            **{
                f"log_damage_{log_suffix}": np.log(damage_df[totals["Total_Damage"]]),
                f"log_affected_{log_suffix}": np.log(damage_df[totals["Total_Affected"]]),
            }
        )

        return damage_df

    reading = builder_fingerprint(
        build,
        query,
        totals,
        log_suffix,
        tuple(extra_columns),
        locations,
        DAMAGE_COLUMNS,
        REPLICATION_FILTERS,
    )

    return cached(cache_dir, name, build, geo_parquet(), params={"reading": reading})


def create_hydro_rivers_damage(cache_dir: Path) -> gpd.GeoDataFrame:
    """
    Build the damage frame for hydrometeorological events, with each event's distance to a river.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.

    Returns
    -------
    damage : GeoDataFrame
        One row per event, carrying its damage totals and its distance to the nearest river.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data_functions.rivers_damage import create_hydro_rivers_damage

        damage = create_hydro_rivers_damage(Path("data"))
    """
    return _create_rivers_damage(
        cache_dir,
        name="rivers_hydro_damage",
        query=f'disaster_class == "{HYDROMETEOROLOGICAL}"',
        total_suffix="Hydro",
        log_suffix="hydro",
    )


def create_floods_rivers_damage(cache_dir: Path) -> gpd.GeoDataFrame:
    """
    Build the damage frame for floods, with each event's distance to a river.

    Parameters
    ----------
    cache_dir : Path
        Directory the source caches live under.

    Returns
    -------
    damage : GeoDataFrame
        One row per flood event, carrying its damage totals and its distance to the nearest river.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data_functions.rivers_damage import create_floods_rivers_damage

        floods = create_floods_rivers_damage(Path("data"))
    """
    return _create_rivers_damage(
        cache_dir,
        name="rivers_floods_damage",
        query='`Disaster Type` == "Flood"',
        total_suffix="Flood",
        log_suffix="floods",
        extra_columns=["Location"],
        locations=all_event_location_overrides(),
    )
