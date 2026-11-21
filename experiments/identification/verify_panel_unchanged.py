"""Check the published panel is unchanged by the identification helpers. Usage: verify_panel_unchanged.py BEFORE.parquet"""

import sys

from pathlib import Path

import polars as pl

from climate_risk.data_functions.emdat_processing import load_emdat_events
from climate_risk.models.identification import (
    HYDROLOGICAL_TYPES,
    recount_events,
    with_geophysical_disasters,
    with_high_impact_hydro,
)
from climate_risk.replication_data import create_replication_data

cache = Path("data")
before = pl.read_parquet(sys.argv[1])  # a parquet snapshot of the panel taken BEFORE the code change
after = create_replication_data(cache)
print("published panel identical before/after:", after.equals(before))
for c in ("hydrological_disasters", "climatological_disasters"):
    print(c, "identical:", after[c].equals(before[c]))

events = load_emdat_events(cache)
# Does the helper reproduce the published hydro column when given the default filters?
re = recount_events(after, events, HYDROLOGICAL_TYPES, column="hydrological_disasters")
print(
    "recount (default filters) == published hydro:",
    re["hydrological_disasters"].cast(pl.Float64).equals(after["hydrological_disasters"].cast(pl.Float64)),
)

geo = with_geophysical_disasters(after, events)
print("existing columns untouched by geophysical add:", geo.select(after.columns).equals(after))
g = geo["geophysical_disasters"]
print(
    "geophysical: rows",
    geo.height,
    "null",
    g.null_count(),
    "nonnull",
    g.drop_nulls().len(),
    "events total",
    g.sum(),
    "share nonzero of all rows",
    round(g.drop_nulls().len() / geo.height, 3),
)

hi = with_high_impact_hydro(after, events, min_deaths=100)
h = hi["hydrological_disasters"]
print(
    "hydro all: nonnull",
    after["hydrological_disasters"].drop_nulls().len(),
    "events",
    after["hydrological_disasters"].sum(),
)
print("hydro >100 deaths: nonnull", h.drop_nulls().len(), "events kept", h.sum())
