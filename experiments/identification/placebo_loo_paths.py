from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

from climate_risk.replication_data import create_replication_data

d = Path("data/idata")
ps = xr.open_datatree(d / "placebo_geophysical_stage2.nc")
py = xr.open_datatree(d / "placebo_geophysical_yeareffect.nc")
hy = xr.open_datatree(d / "frequency_stage2_yeareffect_long.nc")
print("arviz", az.__version__, "groups", list(py.children))

print("\n=== PSIS-LOO on placebo (geophysical) ===")
for n, t in [("stage2", ps), ("yeareffect", py)]:
    print(n, az.loo(t))
print(az.compare({"stage2": ps, "yeareffect": py}))


def path(t):
    return t.posterior["year_effect"].stack(s=("chain", "draw")).transpose("s", "year")


P, H = path(py), path(hy)
yrs = np.intersect1d(P.year.values, H.year.values)
P, H = P.sel(year=yrs), H.sel(year=yrs)
pm_, hm_ = P.mean("s").values, H.mean("s").values
print("\nyears", yrs[0], yrs[-1])
print("posterior-mean path: corr(placebo, hydro) =", np.corrcoef(pm_, hm_)[0, 1].round(3))
print("posterior-mean first differences corr =", np.corrcoef(np.diff(pm_), np.diff(hm_))[0, 1].round(3))
print(
    f"end-of-panel level  placebo {pm_[-1]:.2f} (sd {P.isel(year=-1).std():.2f})   hydro {hm_[-1]:.2f} (sd {H.isel(year=-1).std():.2f})"
)
print(
    "path every 6th year:\n  year   ",
    list(yrs[::6]),
    "\n  placebo",
    pm_[::6].round(2),
    "\n  hydro  ",
    hm_[::6].round(2),
)

co2 = create_replication_data(Path("data")).select("date", "co2").unique().sort("date").to_pandas()
co2["year"] = co2["date"].dt.year
c = co2.set_index("year")["co2"].reindex(yrs)
print(
    "\ncorr(co2 level, placebo path) =",
    np.corrcoef(c, pm_)[0, 1].round(3),
    "  corr(co2, hydro path) =",
    np.corrcoef(c, hm_)[0, 1].round(3),
)
print("corr(co2, calendar year) =", np.corrcoef(c, yrs)[0, 1].round(4))

# posterior correlation between beta[co2] and the final year effect
for n, t in [("placebo", py), ("hydro", hy)]:
    b = t.posterior["beta"].sel(feature="co2").stack(s=("chain", "draw")).values
    e = t.posterior["year_effect"].isel(year=-1).stack(s=("chain", "draw")).values
    print(f"posterior corr(beta[co2], year_effect[-1]) {n}: {np.corrcoef(b, e)[0, 1]:.3f}")
