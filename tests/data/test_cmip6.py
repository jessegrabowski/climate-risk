import numpy as np
import xarray as xr

from climate_risk.data import cmip6


def block(models, member, experiments, periods, versions):
    return {
        "variable": [cmip6.VARIABLE],
        "temporal_aggregation": ["yearly"],
        "model": models,
        "ensemble_member": [member],
        "experiment": experiments,
        "period": periods,
        "version": versions,
    }


def test_the_plan_uses_one_member_and_the_newest_version_per_model():
    constraints = [
        block(["a"], "r1i1p1f1", ["historical"], ["1850_2014"], ["1_0", "2_0"]),
        block(["a"], "r2i1p1f1", ["historical"], ["1850_2014"], ["2_0"]),
        block(["b"], "r1i1p1f2", ["historical"], ["1850_2014"], ["1_0"]),
    ]
    plan = cmip6.plan_requests(constraints)

    assert [(r["model"], r["ensemble_member"], r["version"]) for r in plan] == [
        (["a"], ["r1i1p1f1"], ["2_0"]),
        (["b"], ["r1i1p1f2"], ["1_0"]),
    ]


def test_model_and_experiment_are_read_from_the_file_name():
    name = "rx5dayETCCDI_yr_FGOALS-g3_ssp245_r1i1p1f1_no-base_v20190826_2015-2100_v2-0.nc"

    assert cmip6.split_name(name) == ("FGOALS-g3", "ssp245")


def test_years_are_right_in_a_calendar_without_leap_days(tmp_path):
    data = xr.Dataset(
        {"rx5dayETCCDI": (("time", "lat", "lon"), np.ones((2, 1, 2)))},
        coords={"time": [182.0, 547.0], "lat": [0.0], "lon": [0.0, 180.0]},
    )
    data["time"].attrs = {"units": "days since 1850-1-1", "calendar": "365_day"}
    data.to_netcdf(tmp_path / "x.nc")
    out = cmip6.read_rx5day(tmp_path / "x.nc")

    assert out["year"].to_numpy().tolist() == [1850, 1851]
    assert out["longitude"].to_numpy().tolist() == [-180.0, 0.0]
