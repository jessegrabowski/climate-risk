"""
Does the frequency response to rx5day flatten at high values? (task 2026-11-26, part 1)

Same rows and terms as the A0 fit (H1: base covariates + precipitation anomaly + rx5day, within-country z), with the
linear rx5day term replaced by
    quadratic   z and z squared
    piecewise   z plus hinge terms max(z - k, 0) for knots k = 0, 1, 2
Each is compared with the linear fit by PSIS-LOO.

The CMIP6 multipliers are then rebuilt on the model's own link: for each country, mean over its years of
rate(z + dz) / rate(z), with rate = softplus(linear predictor) as fitted. dz is the CMIP6 shift from
``results/2026-11-24-cmip6-rx5day/country_deltas.parquet``. A country-scenario whose median dz is above the observed
99th percentile of z is flagged and gets no multiplier.

Usage: ``pixi run python experiments/hazard_frequency/nonlinear.py [fit|report]``
"""

import gc
import sys

from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import polars as pl

from model import build_hazard_model
from run import BASE, CACHE, FITS, PRECIP, SAMPLE_KWARGS, build_panel, interval

from climate_risk.models.frequency import TARGET
from climate_risk.models.identification import _prepare, _year_index
from climate_risk.sample import sample_or_load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results/2026-11-26-rx5day-nonlinear")
DELTAS = Path("results/2026-11-24-cmip6-rx5day/country_deltas.parquet")
RX = "rx5day_popmean"
KNOTS = (0.0, 1.0, 2.0)
FORMS = {
    "linear": [RX],
    "quadratic": [RX, "rx_squared"],
    "piecewise": [RX, *(f"rx_hinge{int(k)}" for k in KNOTS)],
}
FOCUS = ("LAO", "CRI", "BFA", "NPL", "ZMB", "MAR")
THINNED = 500  # posterior draws used for the multipliers


def basis(z: np.ndarray, form: str) -> np.ndarray:
    """Raw (unstandardised) basis columns of one functional form, last axis in the order of ``FORMS[form]``."""
    columns = [z]
    if form == "quadratic":
        columns.append(z**2)
    if form == "piecewise":
        columns += [np.maximum(z - k, 0.0) for k in KNOTS]
    return np.stack(columns, axis=-1)


def panel_with_basis() -> pl.DataFrame:
    """The A0 rows with the non-linear basis columns added (computed from the within-country z)."""
    panel = build_panel().drop_nulls(PRECIP)
    z = panel[RX].to_numpy()
    extra = [pl.Series("rx_squared", z**2)]
    extra += [pl.Series(f"rx_hinge{int(k)}", np.maximum(z - k, 0.0)) for k in KNOTS]
    return panel.with_columns(extra)


def features(form: str) -> list[str]:
    return [*BASE, "precip_anom_std_popmean", *FORMS[form]]


def fit(form: str, panel: pl.DataFrame):
    path = FITS / ("H1.nc" if form == "linear" else f"nonlinear_{form}.nc")
    model = build_hazard_model(panel, features(form))
    return sample_or_load(str(path), model=model, sample_kwargs=SAMPLE_KWARGS)


def scaling(prepared, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Centre and scale ``_prepare`` applied to each column, recovered from the raw and standardised values."""
    centre, scale = [], []
    for column in columns:
        raw, std = prepared[column].to_numpy(), prepared[f"{column}__standardized"].to_numpy()
        slope, intercept = np.polyfit(std, raw, 1)
        assert np.allclose(slope * std + intercept, raw), f"{column}: standardisation is not linear"
        centre.append(intercept)
        scale.append(slope)
    return np.array(centre), np.array(scale)


def posterior_arrays(idata) -> dict[str, np.ndarray]:
    """Thinned draws of every term of the linear predictor, draw axis first."""
    posterior = idata["posterior"].to_dataset().stack(sample=("chain", "draw"))
    keep = np.linspace(0, posterior.sizes["sample"] - 1, THINNED).astype(int)
    posterior = posterior.isel(sample=keep).transpose("sample", ...)
    country = next(name for name in ("country_effect", "country") if name in posterior)
    return {
        "beta": posterior["beta"].transpose("sample", "feature").to_numpy(),
        "country": posterior[country].to_numpy(),
        "year": posterior["year_effect"].to_numpy(),
    }


def rebuild_multipliers(form: str, panel: pl.DataFrame, idata) -> tuple[pl.DataFrame, np.ndarray]:
    """Per CMIP model, scenario, window and country: mean over years of rate(z + dz) / rate(z), posterior mean."""
    columns = features(form)
    prepared = _prepare(panel, columns, [TARGET])
    X = prepared[[f"{c}__standardized" for c in columns]].to_numpy()
    centre, scale = scaling(prepared, FORMS[form])
    where = [columns.index(c) for c in FORMS[form]]
    z = prepared[RX].to_numpy()
    isos = np.array(sorted(prepared["ISO"].unique()))
    iso_idx = np.searchsorted(isos, prepared["ISO"].to_numpy())
    _, year_idx = _year_index(prepared)
    draws = posterior_arrays(idata)
    assert draws["country"].shape[1] == len(isos)

    deltas = pl.read_parquet(DELTAS)
    rows = []
    for position, iso in enumerate(isos):
        mine = iso_idx == position
        country = deltas.filter(pl.col("ISO") == iso, pl.col("dz").is_not_null(), pl.col("dz").is_not_nan())
        if country.height == 0:
            continue
        linear = draws["country"][:, [position]] + draws["year"][:, year_idx[mine]] + draws["beta"] @ X[mine].T
        base_rate = np.logaddexp(0.0, linear)  # softplus
        beta_rx = draws["beta"][:, where]
        old = (basis(z[mine], form) - centre) / scale
        for model, scenario, window, dz in country.select("model", "scenario", "window", "dz").iter_rows():
            new = (basis(z[mine] + dz, form) - centre) / scale
            shifted = np.logaddexp(0.0, linear + beta_rx @ (new - old).T)
            ratio = (shifted / base_rate).mean(axis=1)  # one value per draw
            rows.append((iso, model, scenario, window, dz, float(ratio.mean()), ratio))

    table = pl.DataFrame(
        [row[:6] for row in rows], schema=["ISO", "model", "scenario", "window", "dz", "multiplier"], orient="row"
    )
    return table, np.stack([row[6] for row in rows])


def summarise_multipliers(table: pl.DataFrame, per_draw: np.ndarray, p99: float) -> pl.DataFrame:
    """Median and 10-90% range across CMIP models, posterior 89% interval of the median, and the extrapolation flag."""
    table = table.with_row_index("row")
    out = []
    for keys, group in table.group_by("ISO", "scenario", "window", maintain_order=True):
        median_dz = float(group["dz"].median())
        flagged = median_dz > p99
        median_by_draw = np.median(per_draw[group["row"].to_numpy()], axis=0)
        low, high = interval(median_by_draw)
        values = {
            "multiplier_median": float(group["multiplier"].median()),
            "multiplier_p10": float(group["multiplier"].quantile(0.1, "linear")),
            "multiplier_p90": float(group["multiplier"].quantile(0.9, "linear")),
            "posterior_hdi89_lb": low,
            "posterior_hdi89_ub": high,
        }
        out.append(
            {
                "ISO": keys[0],
                "scenario": keys[1],
                "window": keys[2],
                "models": group.height,
                "dz_median": median_dz,
                "share_models_dz_above_p99": float((group["dz"] > p99).mean()),
                "flag_beyond_p99": flagged,
                **{name: None if flagged else value for name, value in values.items()},
            }
        )
    return pl.DataFrame(out, infer_schema_length=None)


def curve(form: str, panel: pl.DataFrame, idata, grid: np.ndarray) -> np.ndarray:
    """Draws of the rx5day contribution to the linear predictor, relative to z = 0, shape (draw, grid)."""
    columns = features(form)
    prepared = _prepare(panel, columns, [TARGET])
    _, scale = scaling(prepared, FORMS[form])
    where = [columns.index(c) for c in FORMS[form]]
    beta = idata["posterior"]["beta"].transpose("chain", "draw", "feature").to_numpy().reshape(-1, len(columns))
    difference = (basis(grid, form) - basis(np.zeros(1), form)) / scale
    return beta[:, where] @ difference.T


def plot(curves: dict[str, np.ndarray], grid: np.ndarray, z: np.ndarray) -> None:
    """One panel per non-linear form, each with the linear fit for reference."""
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    figure, axes = plt.subplots(
        2, 2, figsize=(9, 5.4), sharex=True, sharey="row", height_ratios=[4, 1], layout="constrained"
    )
    for column, form in enumerate(("quadratic", "piecewise")):
        top, bottom = axes[0, column], axes[1, column]
        for name, colour, style in (("linear", orange, "--"), (form, blue, "-")):
            draws = curves[name]
            bounds = np.array([interval(draws[:, i]) for i in range(len(grid))])
            top.fill_between(grid, bounds[:, 0], bounds[:, 1], color=colour, alpha=0.18, linewidth=0)
            top.plot(grid, draws.mean(axis=0), color=colour, linestyle=style, linewidth=2, label=name)
        top.axhline(0, color=muted, linewidth=0.6)
        top.set_title(f"{form.capitalize()} against linear", color=ink, fontsize=11, loc="left")
        top.legend(frameon=False, loc="upper left")
        bottom.hist(z, bins=40, color=muted)
        bottom.set_xlabel("rx5day, z-score within country", color=ink)
    axes[0, 0].set_ylabel("Contribution to the linear\npredictor (0 at z = 0)", color=ink)
    axes[1, 0].set_ylabel("Country-years", color=ink)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=muted)
    figure.savefig(RESULTS / "response.png", dpi=160)


def main(step: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    panel = panel_with_basis()
    prepared = _prepare(panel, features("linear"), [TARGET])
    z = prepared[RX].to_numpy()
    assert len(prepared) == 7710
    distribution = {
        "rows": len(z),
        "min": z.min(),
        "p01": np.quantile(z, 0.01),
        "p50": np.quantile(z, 0.5),
        "p99": np.quantile(z, 0.99),
        "max": z.max(),
        **{f"rows_above_{int(k)}": int((z > k).sum()) for k in KNOTS},
    }
    pl.DataFrame([distribution]).write_csv(RESULTS / "z_distribution.csv")
    print(distribution, flush=True)

    grid = np.linspace(z.min(), z.max(), 120)
    loos, curves, diagnostics = {}, {}, []
    for form in FORMS:  # one fit in memory at a time
        print(f"== {form}", flush=True)
        idata = fit(form, panel)
        if step == "fit":
            del idata
            gc.collect()
            continue
        loos[form] = az.loo(idata, var_name=TARGET, pointwise=True)
        curves[form] = curve(form, panel, idata, grid)
        summary = az.summary(idata, var_names=["beta", "psi", "alpha", "year_effect_sd"], kind="diagnostics")
        diagnostics.append(
            {
                "fit": form,
                "elpd": float(loos[form].elpd),
                "divergences": int(np.asarray(idata["sample_stats"]["diverging"].values).sum()),
                "max_r_hat": float(summary["r_hat"].max()),
                "pareto_k_gt_0.7": int((loos[form].pareto_k > 0.7).sum()),
            }
        )
        table, per_draw = rebuild_multipliers(form, panel, idata)
        table.write_parquet(CACHE / "processed" / f"nonlinear_multipliers_{form}.parquet")
        summarise_multipliers(table, per_draw, float(distribution["p99"])).write_csv(
            RESULTS / f"multipliers_{form}.csv"
        )
        del idata
        gc.collect()
    if step == "fit":
        return

    comparison = []
    for form in ("quadratic", "piecewise"):
        difference = np.asarray(loos[form].elpd_i) - np.asarray(loos["linear"].elpd_i)
        comparison.append(
            {
                "fit": form,
                "elpd_gain_vs_linear": float(difference.sum()),
                "dse": float(np.sqrt(len(difference) * difference.var(ddof=1))),
            }
        )
    pl.DataFrame(diagnostics).join(pl.DataFrame(comparison), on="fit", how="left").write_csv(RESULTS / "loo.csv")
    pl.DataFrame({"z": grid, **{f"{f}_mean": curves[f].mean(axis=0) for f in FORMS}}).write_csv(RESULTS / "curves.csv")
    plot(curves, grid, z)
    print(pl.DataFrame(diagnostics), pl.DataFrame(comparison), sep="\n")


if __name__ == "__main__":
    pl.Config.set_tbl_cols(12)
    main(sys.argv[1] if len(sys.argv) > 1 else "report")
