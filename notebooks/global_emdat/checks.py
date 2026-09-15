import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from arviz_plots import plot_dist, plot_ppc_rootogram
from preliz.distributions.distributions import Continuous

from climate_risk.plotting import PALETTE

STATISTICS = ["zero share", "mean", "max"]
INTERVALS = np.linspace(0.05, 0.95, 25)
NOMINAL_WIDTHS = np.linspace(0.01, 0.99, 99)
ENVELOPE_DRAWS = 500
ENVELOPE_PROB = 0.99


def summarize(idata: xr.DataTree, names: list[str]) -> pd.DataFrame:
    """One az.summary call per variable: with several names at once, arviz 1.2 mislabels the rows of a 2-d one."""
    return pd.concat([az.summary(idata, var_names=[name], round_to=3) for name in names])


def count_summaries(draws: xr.DataArray) -> pd.DataFrame:
    """Share of zeros, mean, and maximum over the country-years, one row per class and draw."""
    summaries = xr.Dataset({"zero share": (draws == 0).mean(dim="obs_idx"),
                            "mean": draws.mean(dim="obs_idx"),
                            "max": draws.max(dim="obs_idx")})

    return summaries.to_dataframe()[STATISTICS]


def observed_count_summaries(counts: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"zero share": (counts == 0).mean(), "mean": counts.mean(), "max": counts.max()})


def plot_predictive_summaries(summaries: pd.DataFrame,
                              observed: pd.DataFrame,
                              classes: list[str],
                              title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))

    for row, name in zip(axes, classes, strict=True):
        for axis, statistic in zip(row, STATISTICS, strict=True):
            values = summaries.xs(name, level="disaster")[statistic]
            reference = observed.loc[name, statistic]
            if statistic == "max":
                values, reference = np.log10(values + 1), np.log10(reference + 1)

            axis.hist(values, bins=40, color=PALETTE["primary"], alpha=0.8)
            axis.axvline(reference, color=PALETTE["observed"], linewidth=1.5)
            axis.set_title(f"{name}: {statistic}" + (" (log10)" if statistic == "max" else ""), loc="left")
            axis.set_yticks([])

    fig.suptitle(title, x=0.01, ha="left")
    plt.show()


def plot_rootograms(idata: xr.DataTree, counts: pd.DataFrame, classes: list[str]) -> None:
    """One rootogram per class, the axis stopping a little past the observed maximum."""
    for name in classes:
        grid = plot_ppc_rootogram(idata,
                                  var_names="y",
                                  coords={"disaster": name},
                                  backend="matplotlib",
                                  figure_kwargs={"figsize": (7, 3.5)},
                                  visuals={"title": False})
        axis = grid.viz["plot"]["y"].item()
        axis.set_title(f"{name}: country-years by recorded count", loc="left")
        axis.set_xlim(-0.5, counts[name].max() + 5)

    plt.show()


def coverage_minus_nominal(predictive: xr.DataArray,
                           observed: xr.DataArray,
                           nominal: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Share of observations inside each central predictive interval, less the nominal share.

    Counts are discrete, so the probability integral transform is randomized within the atom at the observed
    value, which makes it uniform under a calibrated model.
    """
    below = (predictive < observed).mean(dim=("chain", "draw"))
    at = (predictive == observed).mean(dim=("chain", "draw"))
    pit = below + rng.uniform(size=below.shape) * at
    width = 2 * np.abs(pit - 0.5)

    return np.array([(width <= level).mean().item() for level in nominal]) - nominal


def plot_coverage(idata: xr.DataTree, iso: np.ndarray, classes: list[str], seed: int = 0) -> None:
    """Coverage of the central predictive intervals per country-year, and per country total.

    The envelope is the pointwise band of the same curve computed from uniform draws, which is what a calibrated
    model would produce with that many observations.
    """
    # arviz_plots.plot_ppc_pit(coverage=True) evaluates its ECDF on the range of the PIT values instead of [0, 1]
    # (arviz-stats issue 450), so the curve is built here.
    rng = np.random.default_rng(seed)
    nominal = NOMINAL_WIDTHS
    by_iso = xr.DataArray(iso, dims="obs_idx", name="iso")
    levels = {
        "country-year": (idata.posterior_predictive["y"], idata.observed_data["y"]),
        "country total": (idata.posterior_predictive["y"].groupby(by_iso).sum(),
                          idata.observed_data["y"].groupby(by_iso).sum()),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

    for (level, (predictive, observed)), row in zip(levels.items(), axes, strict=True):
        n_observations = observed.isel(disaster=0).size
        uniform_width = 2 * np.abs(rng.uniform(size=(ENVELOPE_DRAWS, n_observations)) - 0.5)
        envelope = np.stack([(uniform_width <= level).mean(axis=1) - level for level in nominal], axis=1)
        low, high = np.quantile(envelope, [(1 - ENVELOPE_PROB) / 2, (1 + ENVELOPE_PROB) / 2], axis=0)

        for disaster, axis in zip(classes, row, strict=True):
            curve = coverage_minus_nominal(predictive.sel(disaster=disaster, drop=True),
                                           observed.sel(disaster=disaster, drop=True),
                                           nominal,
                                           rng)
            axis.fill_between(100 * nominal, low, high, color=PALETTE["primary"], alpha=0.15, linewidth=0)
            axis.plot(100 * nominal, curve, color=PALETTE["primary"])
            axis.axhline(0, color=PALETTE["observed"], linewidth=1, linestyle="--")
            axis.set_title(f"{disaster}, per {level}", loc="left")

    for axis in axes[-1]:
        axis.set_xlabel("interval width (%)")
    for axis in axes[:, 0]:
        axis.set_ylabel("coverage minus nominal")
    fig.suptitle(f"Interval coverage against nominal, with the {ENVELOPE_PROB:.0%} envelope of a calibrated model",
                 x=0.01,
                 ha="left")
    plt.show()


def plot_posterior_with_prior(posterior: xr.Dataset, priors: dict[str, Continuous], n_cols: int = 4) -> plt.Figure:
    """Facet one KDE per named parameter, with its prior's density drawn over it."""
    n_rows = -(-len(priors) // n_cols)
    grid = plot_dist(posterior[list(priors)],
                     kind="kde",
                     ci_kind="hdi",
                     ci_prob=0.94,
                     point_estimate="mean",
                     backend="matplotlib",
                     figure_kwargs={"figsize": (14, 3 * n_rows)},
                     cols=["__variable__"],
                     col_wrap=n_cols)

    for name, prior in priors.items():
        axis = grid.viz["plot"][name].item()
        prior.plot_pdf(ax=axis, legend=False, color="tab:orange")

    return grid.viz["figure"].item()


def plot_country_counts(posterior_y: xr.DataArray,
                        panel: pd.DataFrame,
                        counts: pd.DataFrame,
                        isos: list[str],
                        classes: list[str]) -> None:
    """Observed counts, the predictive mean, and stacked central intervals, one row per country."""
    fig, axes = plt.subplots(len(isos), len(classes), figsize=(12, 3 * len(isos)), sharex=True)

    for iso, row in zip(isos, axes, strict=True):
        rows = panel.index[panel["iso"] == iso]
        years = panel.loc[rows, "date"].dt.year.to_numpy()

        for axis, name in zip(row, classes, strict=True):
            predictive = posterior_y.sel(obs_idx=rows, disaster=name).stack(sample=("chain", "draw"))

            for prob in INTERVALS[::-1]:
                low, high = predictive.quantile([(1 - prob) / 2, (1 + prob) / 2], dim="sample").to_numpy()
                axis.fill_between(years, low, high, color=PALETTE["primary"], alpha=0.08, linewidth=0)

            axis.plot(years, predictive.mean(dim="sample"), color=PALETTE["primary"], label="posterior predictive mean")
            axis.plot(years, counts.loc[rows, name], "o", color=PALETTE["observed"], markersize=3, label="observed")
            axis.set_title(f"{iso}, {name}", loc="left")

    for axis in axes[:, 0]:
        axis.set_ylabel("events recorded")
    for axis in axes[-1]:
        axis.set_xlabel("year")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles=handles,
               labels=labels,
               loc="lower center",
               bbox_to_anchor=(0.5, -0.03),
               ncol=len(labels),
               frameon=False)
    plt.show()
