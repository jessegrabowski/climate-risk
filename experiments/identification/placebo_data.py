"""Geophysical placebo target aligned to the replication panel (scratch only; repo loaders untouched)."""

from pathlib import Path

import polars as pl

from climate_risk.data_functions.emdat_processing import EventFilters, event_filter, load_emdat_events
from climate_risk.replication_data import create_replication_data

GEOPHYSICAL = ("Earthquake", "Volcanic activity", "Mass movement (dry)")


def placebo_panel(cache: Path = Path("data")) -> pl.DataFrame:
    panel = create_replication_data(cache)
    ev = load_emdat_events(cache).with_columns(pl.col("date").dt.truncate("1y"))
    ev = ev.filter(event_filter(EventFilters()) & pl.col("Disaster Type").is_in(GEOPHYSICAL))
    counts = ev.group_by("ISO", "date").len().rename({"len": "geo"})
    # Same convention as frequency._prepare_panel: a country-year with no record counts as zero.
    out = panel.join(counts, on=["ISO", "date"], how="left").with_columns(pl.col("geo").fill_null(0).cast(pl.Float64))
    return out.with_columns(pl.col("geo").alias("hydrological_disasters"))  # model TARGET slot


if __name__ == "__main__":
    print(load := load_emdat_events(Path("data"))["Disaster Type"].value_counts().sort("count", descending=True))
    p = placebo_panel()
    print(
        p.shape, "geo>0 share:", (p["hydrological_disasters"] > 0).mean(), "mean:", p["hydrological_disasters"].mean()
    )
    print("date dtype", p["date"].dtype)
