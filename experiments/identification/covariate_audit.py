from pathlib import Path

import numpy as np
import polars as pl

from climate_risk.models.frequency import FEATURES
from climate_risk.replication_data import create_replication_data

p = create_replication_data(Path("data")).with_columns(pl.col("population").log().alias("ln_population"))
s = p.drop_nulls(FEATURES).with_columns(pl.col("date").dt.year().alias("year")).to_pandas()
print("estimation sample rows", len(s), "countries", s.ISO.nunique(), "years", s.year.min(), s.year.max())
cols = [*FEATURES, "dev_from_trend_ocean_temp"]
print("\ncorr with calendar year (pooled rows):")
for c in cols:
    print(f"  {c:28s} {np.corrcoef(s[c], s.year)[0, 1]: .3f}")
print(
    "\nshare of variance that is BETWEEN years (same value for every country in a year => year effect can absorb it):"
)
for c in cols:
    ym = s.groupby("year")[c].transform("mean")
    print(f"  {c:28s} {ym.var() / s[c].var(): .3f}   (within-year sd {np.sqrt(((s[c] - ym) ** 2).mean()):.3f})")
print(
    "\nco2 and dev_from_trend_ocean_temp: distinct values per year:",
    s.groupby("year")["co2"].nunique().max(),
    s.groupby("year")["dev_from_trend_ocean_temp"].nunique().max(),
)
print(
    "corr(co2, dev_from_trend_ocean_temp) over years:",
    round(s.groupby("year")[["co2", "dev_from_trend_ocean_temp"]].first().corr().iloc[0, 1], 3),
)
# precip_deviation: year-demeaned correlation with year is 0 by construction; check its pooled mean drift
print(
    "\nprecip_deviation yearly mean (first/last 5 yrs):",
    s.groupby("year").precip_deviation.mean().round(2).iloc[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]].to_dict(),
)
