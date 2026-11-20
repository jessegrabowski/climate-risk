from collections.abc import Mapping, Sequence
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from scipy import stats

REGIONS = ("Asia", "Europe", "Africa", "Oceania", "Americas")


# Matplotlib's cycle assigns a color per call site, so one series drawn in two figures comes out in
# two colors. Figures name a role here instead.
PALETTE = {
    "primary": "tab:blue",
    "secondary": "tab:red",
    "observed": "0.15",
    "divider": "0.35",
}


def configure_plot_style(add_grid: bool = False) -> None:
    """
    Apply the project's matplotlib and pandas display settings to the current session.

    Sets figure size, constrained layout, spine visibility, and a two-decimal float format for
    pandas. Call it once, before plotting.

    Parameters
    ----------
    add_grid : bool, optional
        Draw a dashed horizontal grid on every axes. Default False.

    Examples
    --------
    .. code-block:: python

        import matplotlib.pyplot as plt

        from climate_risk.plotting import configure_plot_style

        configure_plot_style(add_grid=True)
        figure, axis = plt.subplots()
    """
    pd.set_option("display.float_format", "{:.2f}".format)

    plt.rcParams["figure.figsize"] = (14, 4)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.facecolor"] = "w"
    plt.rcParams["axes.grid"] = add_grid
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def prepare_gridspec_figure(n_cols: int, n_plots: int, figure: plt.Figure | None = None) -> tuple[GridSpec, list]:
    """
    Lay out a grid of subplots, insetting the last row when it is not full.

    Parameters
    ----------
    n_cols : int
        Columns in the grid.
    n_plots : int
        Subplots to make room for.
    figure : Figure, optional
        Figure to attach the layout to. Default None, which leaves it unattached.

    Returns
    -------
    gridspec : GridSpec
        The layout the subplots are cut from.
    locations : list of tuple of slice
        Row and column slice for each subplot, in order.

    Examples
    --------
    Lay out seven panels across three columns:

    .. code-block:: python

        import matplotlib.pyplot as plt

        from climate_risk.plotting import prepare_gridspec_figure

        figure = plt.figure()
        gridspec, positions = prepare_gridspec_figure(3, 7, figure)
    """
    remainder = n_plots % n_cols
    has_remainder = remainder > 0
    n_rows = n_plots // n_cols + int(has_remainder)

    gs = GridSpec(2 * n_rows, 2 * n_cols, figure=figure)
    plot_locs = []

    for i in range(n_rows - int(has_remainder)):
        for j in range(n_cols):
            plot_locs.append((slice(i * 2, (i + 1) * 2), slice(j * 2, (j + 1) * 2)))

    if has_remainder:
        last_row = slice((n_rows - 1) * 2, n_rows * 2)
        left_pad = int(n_cols - remainder)
        for j in range(remainder):
            col_slice = slice(left_pad + j * 2, left_pad + (j + 1) * 2)
            plot_locs.append((last_row, col_slice))

    return gs, plot_locs


def panel_grid(
    *,
    names: Sequence[str],
    labels: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    n_cols: int = 4,
    height: float = 2.9,
    width: float = 4.6,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Return a figure and one titled axis per name, laid out on a shared grid.

    Parameters
    ----------
    names : sequence of str
        One axis per entry, in order. Doubles as the lookup key into ``labels`` and ``units``.
    labels : mapping of str to str, optional
        Axis titles, keyed by name. A name absent from the mapping titles itself. Default None.
    units : mapping of str to str, optional
        Y-axis labels, keyed by name. A name absent from the mapping gets none. Default None.
    n_cols : int, optional
        Columns in the grid, capped at the number of names. Default 4.
    height : float, optional
        Inches per row. Default 2.9.
    width : float, optional
        Inches per column. Default 4.6.

    Returns
    -------
    figure : Figure
        The figure the axes belong to.
    axes : list of Axes
        One axis per name, in the order given.

    Examples
    --------
    .. code-block:: python

        from climate_risk.plotting import panel_grid

        figure, axes = panel_grid(names=["co2", "precip"], units={"co2": "ppm"})
    """
    labels = labels or {}
    units = units or {}
    columns = min(len(names), n_cols)

    figure = plt.figure(figsize=(width * columns, height * int(np.ceil(len(names) / n_cols))), dpi=200)
    grid, locations = prepare_gridspec_figure(n_cols=columns, n_plots=len(names), figure=figure)
    axes = [figure.add_subplot(grid[location]) for location in locations]

    for axis, name in zip(axes, names, strict=True):
        axis.set_title(labels.get(name, name), size=11, pad=5)
        axis.set_ylabel(units.get(name, ""), size=8)
        axis.tick_params(axis="both", labelsize=8)

    return figure, axes


def plot_fan(
    *,
    draws: xr.DataArray,
    axis: plt.Axes,
    observed: pd.Series | None = None,
    probs: Sequence[float] = (0.50, 0.89),
    color: str = PALETTE["primary"],
    divider: Any = None,
    shades: tuple[float, float] = (0.16, 0.34),
) -> None:
    """
    Draw a posterior median with nested credible bands, optionally over an observed series.

    Wider intervals are drawn first and shaded lightest, so a narrow band stays legible on top of a
    wide one.

    Parameters
    ----------
    draws : DataArray
        Carrying ``chain``, ``draw`` and ``time``, with ``time`` convertible to dates.
    axis : Axes
        Where to draw.
    observed : Series, optional
        Realised values, drawn as a dashed dark line over the bands. Default None.
    probs : sequence of float, optional
        Credible masses, one band each. Default (0.50, 0.89).
    color : str, optional
        Color of the bands and the median line. Default ``PALETTE["primary"]``.
    divider : optional
        X position for a vertical rule, typically the last observation. Default None, drawing none.
    shades : tuple of float, optional
        Alpha for the widest and narrowest band, interpolated across the rest. Default (0.16, 0.34).

    Examples
    --------
    .. code-block:: python

        import matplotlib.pyplot as plt

        from climate_risk.plotting import plot_fan

        figure, axis = plt.subplots()
        plot_fan(draws=posterior["co2"], axis=axis, observed=observed)
    """
    time = pd.to_datetime(draws.coords["time"].values)

    for prob, shade in zip(sorted(probs, reverse=True), np.linspace(*shades, len(probs)), strict=True):
        band = az.hdi(draws, prob=prob).transpose("ci_bound", ...).values
        axis.fill_between(time, *band, color=color, alpha=shade, lw=0)

    axis.plot(time, draws.median(dim=["chain", "draw"]), color=color, lw=1.6)

    if observed is not None:
        axis.plot(observed.index, observed.to_numpy(), color=PALETTE["observed"], lw=1.0, ls="--")

    if divider is not None:
        axis.axvline(divider, color=PALETTE["divider"], ls=":", lw=0.9)

    axis.margins(x=0)


def _plot_single_kde(
    data: pd.Series,
    axis: plt.Axes | None = None,
    bins: int = 30,
    color: str = PALETTE["primary"],
    leg_loc: str = "upper left",
    set_title: bool = True,
    add_sum_box: bool = True,
) -> plt.Axes:
    """
    Plot one series as a histogram with a kernel density estimate over it.

    Parameters
    ----------
    data : Series
        Values to plot. Missing values are dropped.
    axis : Axes, optional
        Axis to draw on. Default None, which creates a figure and axis.
    bins : int, optional
        Histogram bins. Default 30.
    color : str, optional
        Fill color of the histogram. Default ``PALETTE["primary"]``.
    leg_loc : str, optional
        Where to anchor the summary box. Default ``"upper left"``.
    set_title : bool, optional
        Whether to title the axis with the series name. Default True.
    add_sum_box : bool, optional
        Whether to annotate with descriptive statistics and a Jarque-Bera statistic. Default True.

    Returns
    -------
    axis : Axes
        The axis drawn on.
    """
    data = data.dropna()
    if axis is None:
        _fig, axis = plt.subplots()

    axis.hist(data, bins=bins, density=True, facecolor="none", edgecolor="k", lw=0.5)
    axis.hist(data, bins=bins, density=True, facecolor=color, alpha=0.25)
    sns.kdeplot(data, ax=axis, lw=2, c="k", ls="--")

    if add_sum_box:
        observations = np.asarray(data.values).squeeze()
        summary = stats.describe(observations)
        smallest, largest = summary.minmax
        jb = stats.jarque_bera(observations)

        names = ["N", "Min", "Max", "Mean", "Std", "Skew", "Kurt", "JB"]
        values = [
            summary.nobs,
            smallest,
            largest,
            summary.mean,
            np.sqrt(summary.variance),
            summary.skewness,
            summary.kurtosis,
            jb.statistic,
        ]

        text = "\n".join(
            f"{name:<5} = {' ' if value > 0 else ''}{value:<3.3f}" for name, value in zip(names, values, strict=True)
        )
        box = AnchoredText(text, loc=leg_loc, prop={"fontfamily": "monospace"})
        box.patch.set_alpha(0.5)
        axis.add_artist(box)
    if set_title:
        axis.set_title(str(data.name))

    return axis


def plot_descriptive(
    df: pd.DataFrame | pd.Series,
    n_cols: int = 3,
    bins: int = 30,
    color: str = PALETTE["primary"],
    leg_loc: str = "upper left",
    labels_size: int = 14,
    add_sum_box: bool = True,
    **figure_kwargs: Any,
) -> plt.Figure | plt.Axes:
    """
    Plot one histogram and kernel density estimate per column.

    Parameters
    ----------
    df : DataFrame or Series
        Values to plot, one panel per column.
    n_cols : int, optional
        Columns in the grid, capped at the number of panels. Default 3.
    bins : int, optional
        Histogram bins. Default 30.
    color : str, optional
        Fill color of the histograms. Default ``PALETTE["primary"]``.
    leg_loc : str, optional
        Where to anchor each summary box. Default ``"upper left"``.
    labels_size : int, optional
        Point size of the y-axis labels. Default 14.
    add_sum_box : bool, optional
        Whether to annotate each panel with descriptive statistics. Default True.
    **figure_kwargs
        Passed to :func:`matplotlib.pyplot.figure`.

    Returns
    -------
    figure : Figure or Axes
        The figure the panels were drawn on, or the single axis when there is only one column.

    Examples
    --------
    .. code-block:: python

        from climate_risk.plotting import plot_descriptive

        plot_descriptive(panel[["co2", "precip", "damage"]])
    """
    figsize = figure_kwargs.pop("figsize", (14, 4))
    dpi = figure_kwargs.pop("dpi", 144)

    n_plots = df.shape[1] if isinstance(df, pd.DataFrame) else 1

    if n_plots == 1:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        only = df if isinstance(df, pd.Series) else df.iloc[:, 0]
        return _plot_single_kde(only, axis=ax, bins=bins, color=color, add_sum_box=add_sum_box)

    fig = plt.figure(figsize=figsize, dpi=dpi, **figure_kwargs)

    n_cols = min(n_cols, n_plots)
    gs, locs = prepare_gridspec_figure(n_cols=n_cols, n_plots=n_plots)

    for name, loc in zip(df.columns, locs, strict=True):
        axis = fig.add_subplot(gs[loc])
        axis.set_ylabel("Density", fontsize=labels_size)
        _plot_single_kde(
            df[name],
            axis=axis,
            bins=bins,
            color=color,
            leg_loc=leg_loc,
            add_sum_box=add_sum_box,
        )

    return fig


AGGREGATE_COLUMNS = 2

Aggregation = Literal["count", "first", "last", "max", "mean", "median", "min", "nunique", "std", "sum", "var"]


def plot_aggregated_series(
    df: pd.DataFrame,
    variables: list[str],
    index: str,
    aggregation: Aggregation,
    title: str,
    graph_rows: int = 2,
    figure_size: tuple[float, float] = (20, 18),
    subplot_title_fontsize: int = 14,
) -> plt.Figure:
    """
    Plot one variable per panel, each aggregated over ``index``.

    Parameters
    ----------
    df : DataFrame
        The observations to aggregate.
    variables : list of str
        Columns to plot, one panel each, filling the grid left to right.
    index : str
        Column to aggregate over, which becomes the x axis.
    aggregation : str
        How to combine rows sharing an index value, as named in ``Aggregation``.
    title : str
        Heading for the figure.
    graph_rows : int, optional
        Rows in the grid, which is two columns wide. Default 2.
    figure_size : tuple of float, optional
        Figure width and height in inches. Default (20, 18).
    subplot_title_fontsize : int, optional
        Point size for each panel's title. The figure heading is ten points larger. Default 14.

    Returns
    -------
    figure : Figure
        The figure the panels were drawn on.

    Examples
    --------
    .. code-block:: python

        from climate_risk.plotting import plot_aggregated_series

        figure = plot_aggregated_series(panel, ["damage"], "date", "sum")
    """
    fig, axes = plt.subplots(graph_rows, AGGREGATE_COLUMNS, figsize=figure_size)

    for position, variable in enumerate(variables):
        axis = axes[position // AGGREGATE_COLUMNS, position % AGGREGATE_COLUMNS]
        axis.plot(df.pivot_table(values=variable, index=index, aggfunc=aggregation)[variable])
        axis.set_title(variable, fontsize=subplot_title_fontsize)

    if len(variables) % AGGREGATE_COLUMNS:
        axes[graph_rows - 1, 1].set_axis_off()

    plt.suptitle(title, fontsize=subplot_title_fontsize + 10)
    fig.tight_layout()

    return fig


def plot_aggregated_series_by_region(
    df: pd.DataFrame,
    variables: list[str],
    index: str,
    aggregation: Aggregation,
    title: str,
    graph_rows: int = 2,
    figure_size: tuple[float, float] = (20, 18),
    subplot_title_fontsize: int = 14,
) -> plt.Figure:
    """
    Plot one variable per panel, aggregated over ``index`` and drawn once per region.

    Parameters
    ----------
    df : DataFrame
        The observations to aggregate, carrying a ``Region`` column.
    variables : list of str
        Columns to plot, one panel each, filling the grid left to right.
    index : str
        Column to aggregate over, which becomes the x axis.
    aggregation : str
        How to combine rows sharing an index value, as named in ``Aggregation``.
    title : str
        Heading for the figure.
    graph_rows : int, optional
        Rows in the grid, which is two columns wide. Default 2.
    figure_size : tuple of float, optional
        Figure width and height in inches. Default (20, 18).
    subplot_title_fontsize : int, optional
        Point size for each panel's title. The figure heading is ten points larger. Default 14.

    Returns
    -------
    figure : Figure
        The figure the panels were drawn on.

    Examples
    --------
    .. code-block:: python

        from climate_risk.plotting import plot_aggregated_series_by_region

        figure = plot_aggregated_series_by_region(panel, ["damage"], "date", "sum")
    """
    fig, axes = plt.subplots(graph_rows, AGGREGATE_COLUMNS, figsize=figure_size)

    for position, variable in enumerate(variables):
        axis = axes[position // AGGREGATE_COLUMNS, position % AGGREGATE_COLUMNS]
        for region in REGIONS:
            in_region = df[df["Region"] == region]
            axis.plot(in_region.pivot_table(values=variable, index=index, aggfunc=aggregation), label=region)
        axis.set_title(variable, fontsize=subplot_title_fontsize)

    if len(variables) % AGGREGATE_COLUMNS:
        axes[graph_rows - 1, 1].set_axis_off()

    fig.legend(list(REGIONS), loc="lower right", ncol=len(REGIONS), fontsize=16)
    plt.suptitle(title, fontsize=subplot_title_fontsize + 10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    return fig


def plot_ppc_loopit(
    idata: xr.DataTree, target_name: str, title: str | None = None, **ppc_kwargs: Any
) -> list[plt.Axes]:
    """
    Plot a posterior predictive check above its LOO-PIT diagnostics, as one figure.

    Parameters
    ----------
    idata : DataTree
        Inference data carrying the posterior predictive samples.
    target_name : str
        Variable to check.
    title : str, optional
        Heading for the posterior predictive panel. Default None, which uses ``target_name``.
    **ppc_kwargs
        Passed to :func:`arviz.plot_ppc`.

    Returns
    -------
    axes : list of Axes
        The posterior predictive panel and the two LOO-PIT panels.

    Examples
    --------
    .. code-block:: python

        from climate_risk.plotting import plot_ppc_loopit

        axes = plot_ppc_loopit(idata, "counts")
    """
    fig = plt.figure(figsize=(12, 9))
    gs = plt.GridSpec(2, 2, figure=fig)
    ax_ppc = fig.add_subplot(gs[0, :])
    ax_loo = fig.add_subplot(gs[1, 0])
    ax_ecdf = fig.add_subplot(gs[1, 1])

    az.plot_ppc(idata, ax=ax_ppc, var_names=[target_name], **ppc_kwargs)
    for ax, ecdf in zip([ax_loo, ax_ecdf], [False, True], strict=True):
        az.plot_loo_pit(idata, y=target_name, ecdf=ecdf, ax=ax)

    if title is None:
        title = target_name
    ax_ppc.set_title(title)
    ax_ppc.set_xlabel("")
    return fig.axes


SAMPLE_DIMS = ("chain", "draw")


def _aligned_prediction_mean(draws: xr.DataArray, df: pd.DataFrame) -> np.ndarray:
    """Return the posterior-predictive mean, checked against the frame it will be attached to."""
    predictions = draws.mean(dim=SAMPLE_DIMS)
    if predictions.size != len(df):
        raise ValueError(
            f"posterior_predictive holds {predictions.size} observations but df has {len(df)} rows; "
            f"they must be the observations the model saw, in the same order."
        )
    return np.asarray(predictions.values)


def attach_count_predictions(idata: xr.DataTree, df: pd.DataFrame) -> pd.DataFrame:
    """Attach posterior-predictive means and HDI bounds to the observed frame.

    Parameters
    ----------
    idata : DataTree
        Inference data carrying a ``posterior_predictive`` group with a ``y_hat`` variable.
    df : DataFrame
        The observed frame, in the order the model saw it.

    Returns
    -------
    annotated : DataFrame
        ``df`` with ``predictions`` and the 95% and 50% HDI bounds appended.

    Examples
    --------
    .. code-block:: python

        from climate_risk.plotting import attach_count_predictions

        annotated = attach_count_predictions(idata, panel)
    """
    y_hat = idata.posterior_predictive["y_hat"]
    predictions = _aligned_prediction_mean(y_hat, df)

    hdi_95 = az.hdi(y_hat, prob=0.95)
    hdi_50 = az.hdi(y_hat, prob=0.5)

    return df.assign(
        predictions=predictions,
        lower_y_hat_95=hdi_95.sel(ci_bound="lower").values,
        higher_y_hat_95=hdi_95.sel(ci_bound="upper").values,
        lower_y_hat_50=hdi_50.sel(ci_bound="lower").values,
        higher_y_hat_50=hdi_50.sel(ci_bound="upper").values,
    )


def plot_predicted_counts(idata: xr.DataTree, df: pd.DataFrame, country: str) -> plt.Figure:
    """
    Plot one country's predicted disaster counts against what was observed.

    Parameters
    ----------
    idata : DataTree
        Inference data carrying a ``posterior_predictive`` group with a ``y_hat`` variable.
    df : DataFrame
        The observed frame, in the order the model saw it.
    country : str
        ISO 3166-1 alpha-3 code of the country to draw.

    Returns
    -------
    figure : Figure
        Predicted mean, observed counts, and the 50% and 95% HDI bands.

    Examples
    --------
    .. code-block:: python

        from climate_risk.plotting import plot_predicted_counts

        figure = plot_predicted_counts(idata, panel, "LAO")
    """
    df_predictions = attach_count_predictions(idata=idata, df=df)

    data = df_predictions.query("ISO == @country")

    fig, ax = plt.subplots()
    ax.plot(
        data["date"],
        data["predictions"],
        zorder=1000,
        color=PALETTE["secondary"],
        label="Mean Predicted Disaster Count",
    )
    ax.scatter(data["date"], data["is_disaster"], color="k", label="Actual prob")
    ax.fill_between(
        data["date"],
        data["higher_y_hat_95"],
        data["lower_y_hat_95"],
        alpha=0.25,
        color="tab:blue",
        label="95% HDI",
    )
    ax.fill_between(
        data["date"],
        data["lower_y_hat_50"],
        data["higher_y_hat_50"],
        alpha=0.5,
        color="tab:blue",
        label="50% HDI",
    )
    ax.legend(loc="upper left")

    ax.set_xlabel("date")
    ax.set_ylabel("Disaster Count")

    return fig


def attach_damage_predictions(idata: xr.DataTree, df: pd.DataFrame) -> pd.DataFrame:
    """Attach posterior-predictive damage means and HDI bounds to the observed frame.

    Parameters
    ----------
    idata : DataTree
        Inference data carrying a ``posterior_predictive`` group with a ``damage_millions`` variable.
    df : DataFrame
        The observed frame, in the order the model saw it.

    Returns
    -------
    annotated : DataFrame
        ``df`` with ``predictions`` and the 75% and 50% HDI bounds appended.

    Examples
    --------
    .. code-block:: python

        from climate_risk.plotting import attach_damage_predictions

        annotated = attach_damage_predictions(idata, panel)
    """
    damages = idata.posterior_predictive["damage_millions"]
    predictions = _aligned_prediction_mean(damages, df)

    hdi_75 = az.hdi(damages, prob=0.75)
    hdi_50 = az.hdi(damages, prob=0.5)

    return df.assign(
        predictions=predictions,
        lower_damage_75=hdi_75.sel(ci_bound="lower").values,
        higher_damage_75=hdi_75.sel(ci_bound="upper").values,
        lower_damage_50=hdi_50.sel(ci_bound="lower").values,
        higher_damage_50=hdi_50.sel(ci_bound="upper").values,
    )


def plot_predicted_damages(idata: xr.DataTree, df: pd.DataFrame, country: str, target_variable: str) -> plt.Figure:
    """
    Plot one country's predicted damages against what was observed.

    Parameters
    ----------
    idata : DataTree
        Inference data carrying a ``posterior_predictive`` group with a ``damage_millions`` variable.
    df : DataFrame
        The observed frame, in the order the model saw it.
    country : str
        ISO 3166-1 alpha-3 code of the country to draw.
    target_variable : str
        Column holding the observed damages to plot against.

    Returns
    -------
    figure : Figure
        Predicted mean, observed damages, and the 50% and 75% HDI bands.

    Examples
    --------
    .. code-block:: python

        from climate_risk.plotting import plot_predicted_damages

        figure = plot_predicted_damages(idata, panel, "LAO", "total_damage")
    """
    df_predictions = attach_damage_predictions(idata=idata, df=df)

    data = df_predictions.query("ISO == @country")

    fig, ax = plt.subplots()
    ax.scatter(
        data["date"],
        (data[target_variable].astype(float)),
        color="k",
        label=("Real hydrometereological events damage in millions of dollars"),
    )
    ax.fill_between(
        data["date"],
        data["higher_damage_75"],
        data["lower_damage_75"],
        alpha=0.25,
        color="tab:blue",
        label="75% HDI",
    )
    ax.fill_between(
        data["date"],
        data["lower_damage_50"],
        data["higher_damage_50"],
        alpha=0.5,
        color="tab:blue",
        label="50% HDI",
    )
    ax.legend(loc="upper left", fontsize=14)
    ax.set_xlabel("date", fontsize=14)
    ax.set_ylabel("Disaster damages in 2000 USD millions", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    return fig


def plot_estimate(ax: plt.Axes, row: Any, y_position: float, color: str) -> None:
    """
    Draw one published parameter estimate on a horizontal row.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    row : namedtuple
        One estimate, with ``value``, ``lower``, ``upper`` and ``status`` fields. The interval is dotted when
        ``status`` is ``"prior-dominated"``, and the value hollow when it is ``"fixed"``. A non-finite ``lower`` draws
        no interval.
    y_position : float
        Height of the row.
    color : str
        Color of the estimate's source.

    Examples
    --------
    One estimated and one fixed value of the same parameter, on two rows:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import pandas as pd

        from climate_risk.plotting import plot_estimate

        estimates = pd.DataFrame(
            {
                "value": [0.6, 0.75],
                "lower": [0.5, float("nan")],
                "upper": [0.7, float("nan")],
                "status": ["estimated", "fixed"],
            }
        )
        figure, ax = plt.subplots()
        for y_position, row in enumerate(estimates.itertuples()):
            plot_estimate(ax=ax, row=row, y_position=y_position, color="tab:blue")
    """
    if np.isfinite(row.lower):
        ax.hlines(
            y=y_position,
            xmin=row.lower,
            xmax=row.upper,
            color=color,
            ls=":" if row.status == "prior-dominated" else "-",
            lw=2.0,
        )
    ax.scatter(
        x=row.value,
        y=y_position,
        s=50,
        color="white" if row.status == "fixed" else color,
        edgecolor=color,
        linewidth=1.5,
        zorder=3,
    )


def plot_impact_heatmap(
    ax: plt.Axes,
    impact: xr.DataArray,
    literature_points: pd.DataFrame,
    literature_colors: Mapping[str, str],
    xlabel: str,
    ylabel: str,
    yscale: str,
    norm: Normalize,
) -> QuadMesh:
    """
    Draw an impact over a two-parameter grid, its zero contour, and published estimates of both parameters.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    impact : DataArray
        GDP at impact as a share, with dims ``(y, x)`` named for the two parameters.
    literature_points : DataFrame
        One row per source, indexed by source, with a column for each of the two parameters.
    literature_colors : dict of str to str
        The sources to mark, mapped to their colors.
    xlabel, ylabel : str
        Axis labels.
    yscale : {"linear", "log"}
        Scale of the y axis.
    norm : Normalize
        Color normalization of the impact in percent, shared by every panel that shares a colorbar.

    Returns
    -------
    mesh : QuadMesh
        The heatmap, for the caller's colorbar.

    Examples
    --------
    An impact over two parameters, with one published estimate marked:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import xarray as xr

        from matplotlib.colors import Normalize

        from climate_risk.plotting import plot_impact_heatmap

        x, y = np.linspace(0.1, 0.9, 20), np.linspace(1.0, 5.0, 20)
        impact = xr.DataArray(
            0.05 - 0.1 * x[None, :] / y[:, None], dims=["chi", "phi_H"], coords={"chi": y, "phi_H": x}
        )
        points = pd.DataFrame({"phi_H": [0.3], "chi": [2.0]}, index=["source A"])

        figure, ax = plt.subplots()
        mesh = plot_impact_heatmap(
            ax=ax,
            impact=impact,
            literature_points=points,
            literature_colors={"source A": "tab:green"},
            xlabel="habit",
            ylabel="inverse Frisch",
            yscale="linear",
            norm=Normalize(vmin=-5, vmax=5),
        )
        figure.colorbar(mesh, ax=ax)
    """
    y_name, x_name = impact.dims
    x_grid = impact[x_name].to_numpy()
    y_grid = impact[y_name].to_numpy()
    mesh = ax.pcolormesh(x_grid, y_grid, 100 * impact.values, cmap="RdBu", norm=norm, shading="auto")
    contour = ax.contour(x_grid, y_grid, impact.values, levels=[0.0], colors="k", linewidths=1.5)
    ax.clabel(contour, fmt={0.0: "GDP = 0"})

    for source, color in literature_colors.items():
        point = literature_points.loc[source].astype(float)
        ax.scatter(
            x=point[str(x_name)],
            y=point[str(y_name)],
            s=90,
            color=color,
            edgecolor="k",
            zorder=10,
            label=source,
        )

    ax.set(yscale=yscale, xlim=(x_grid[0], x_grid[-1]), ylim=(y_grid[0], y_grid[-1]), xlabel=xlabel, ylabel=ylabel)
    ax.grid(visible=False)

    return mesh


def _plot_response_band(
    ax: plt.Axes, quantiles: xr.DataArray, reference: xr.DataArray | None, reference_label: str
) -> None:
    """The median and outer interval of one response over ``quantile``, with ``reference`` dashed on top when given."""
    lower, upper = quantiles.isel(quantile=0), quantiles.isel(quantile=-1)
    interval = float(quantiles["quantile"][-1] - quantiles["quantile"][0])
    ax.fill_between(quantiles["time"], lower, upper, color=PALETTE["primary"], alpha=0.3, linewidth=0)
    ax.plot(
        quantiles["time"],
        quantiles.sel(quantile=0.5),
        color=PALETTE["primary"],
        lw=1.5,
        label=f"median and {interval:.0%} interval",
    )
    if reference is not None:
        ax.plot(reference["time"], reference, color="k", ls="--", lw=1.2, label=reference_label)
    ax.axhline(0.0, color="0.6", lw=0.75)


def _add_figure_legend(figure: plt.Figure, ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(
        handles=handles, labels=labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels), frameon=False
    )


def plot_response_grid(
    quantiles: xr.DataArray,
    reference: xr.DataArray | None,
    shock_labels: Mapping[str, str],
    variable_labels: Mapping[str, str],
    reference_label: str = "reference",
) -> plt.Figure:
    """
    Draw a grid of shocks by variables, each panel the distribution of one impulse response.

    Parameters
    ----------
    quantiles : DataArray
        Quantiles of the responses in display units, with dims ``shock``, ``quantile``, ``time`` and ``variable``. The
        band spans the first and last quantile, and the line is the median.
    reference : DataArray or None
        One response path per shock and variable in the same units, with dims ``shock``, ``time`` and ``variable``,
        drawn dashed on top of the band. None draws no reference.
    shock_labels : dict of str to str
        The shocks to plot, one row each, mapped to their row labels.
    variable_labels : dict of str to str
        Column titles for the variables of ``quantiles``, one column each.
    reference_label : str, optional
        Legend entry for ``reference``. Default "reference".

    Returns
    -------
    figure : Figure
        The grid.

    Examples
    --------
    Two shocks by two variables, without a reference path:

    .. code-block:: python

        import numpy as np
        import xarray as xr

        from climate_risk.plotting import plot_response_grid

        rng = np.random.default_rng(0)
        draws = xr.DataArray(
            rng.normal(size=(2, 200, 12, 2)).cumsum(axis=-2),
            dims=["shock", "draw", "time", "variable"],
            coords={"shock": ["a", "b"], "variable": ["GDP", "C"]},
        )
        quantiles = draws.quantile([0.03, 0.5, 0.97], dim="draw")
        figure = plot_response_grid(
            quantiles=quantiles,
            reference=None,
            shock_labels={"a": "shock a", "b": "shock b"},
            variable_labels={"GDP": "GDP (%)", "C": "consumption (%)"},
        )
    """
    variables = list(quantiles["variable"].to_numpy())
    figure, axes = plt.subplots(
        nrows=len(shock_labels),
        ncols=len(variables),
        figsize=(2.3 * len(variables), 2 * len(shock_labels)),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for row, (shock, shock_label) in enumerate(shock_labels.items()):
        for column, variable in enumerate(variables):
            ax = axes[row, column]
            _plot_response_band(
                ax=ax,
                quantiles=quantiles.sel(shock=shock, variable=variable),
                reference=None if reference is None else reference.sel(shock=shock, variable=variable),
                reference_label=reference_label,
            )
            if row == 0:
                ax.set_title(variable_labels[variable], loc="left", fontsize=10)
            if column == 0:
                ax.set_ylabel(shock_label, rotation=0, ha="right", va="center")

    _add_figure_legend(figure=figure, ax=axes[0, 0])
    figure.supxlabel("quarters after the shock")

    return figure


def plot_shock_responses(
    quantiles: xr.DataArray,
    reference: xr.DataArray | None,
    variable_labels: Mapping[str, str],
    reference_label: str = "reference",
) -> plt.Figure:
    """
    Draw one panel per variable, four to a row, each the distribution of the response to one shock.

    Parameters
    ----------
    quantiles : DataArray
        Quantiles of the responses in display units, with dims ``quantile``, ``time`` and ``variable``.
    reference : DataArray or None
        One response path per variable in the same units, with dims ``time`` and ``variable``, drawn dashed on top of
        the band. None draws no reference.
    variable_labels : dict of str to str
        Panel titles for the variables of ``quantiles``.
    reference_label : str, optional
        Legend entry for ``reference``. Default "reference".

    Returns
    -------
    figure : Figure
        The panels.

    Examples
    --------
    The responses of two variables to one shock, without a reference path:

    .. code-block:: python

        import numpy as np
        import xarray as xr

        from climate_risk.plotting import plot_shock_responses

        rng = np.random.default_rng(0)
        draws = xr.DataArray(
            rng.normal(size=(200, 12, 2)).cumsum(axis=-2),
            dims=["draw", "time", "variable"],
            coords={"variable": ["GDP", "C"]},
        )
        quantiles = draws.quantile([0.03, 0.5, 0.97], dim="draw")
        figure = plot_shock_responses(
            quantiles=quantiles, reference=None, variable_labels={"GDP": "GDP (%)", "C": "consumption (%)"}
        )
    """
    variables = list(quantiles["variable"].to_numpy())
    n_rows = int(np.ceil(len(variables) / 4))
    figure, axes = plt.subplots(
        nrows=n_rows, ncols=4, figsize=(14, 3 * n_rows), sharex=True, squeeze=False, layout="constrained"
    )
    for ax, variable in zip(axes.flat, variables, strict=False):
        _plot_response_band(
            ax=ax,
            quantiles=quantiles.sel(variable=variable),
            reference=None if reference is None else reference.sel(variable=variable),
            reference_label=reference_label,
        )
        ax.set_title(variable_labels[variable], loc="left")
    for ax in axes.flat[len(variables) :]:
        ax.set_axis_off()

    _add_figure_legend(figure=figure, ax=axes[0, 0])
    figure.supxlabel("quarters after the shock")

    return figure


def plot_density_against_data(ax: plt.Axes, values: Any, data_value: float) -> None:
    """
    Draw the density of simulated values with the data's value dashed on top.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    values : array_like
        The simulated values. Where they do not vary, a vertical line replaces the density.
    data_value : float
        The value in the data.

    Examples
    --------
    A posterior ratio against its data counterpart:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np

        from climate_risk.plotting import plot_density_against_data

        figure, ax = plt.subplots()
        plot_density_against_data(ax=ax, values=np.random.default_rng(0).normal(0.3, 0.02, 500), data_value=0.31)
    """
    values = np.asarray(values)
    median = np.median(values)
    if np.ptp(values) <= 1e-9 * max(1.0, abs(median)):
        ax.axvline(median, color=PALETTE["primary"], lw=2.0)
        half_width = max(abs(data_value - median), 0.01 * abs(median))
        center = (data_value + median) / 2
        ax.set_xlim(center - 1.5 * half_width, center + 1.5 * half_width)
    else:
        density = az.kde(xr.DataArray(values, dims=["sample"]), dim="sample")
        ax.fill_between(
            density.sel(plot_axis="x"), density.sel(plot_axis="y"), color=PALETTE["primary"], alpha=0.3, linewidth=0
        )
        ax.plot(density.sel(plot_axis="x"), density.sel(plot_axis="y"), color=PALETTE["primary"], lw=1.5)
    ax.axvline(data_value, color="k", ls="--", lw=1.5)
    ax.set(yticks=[])
    ax.grid(visible=False, axis="y")


OUTER_INTERVAL_WIDTH = 2.5
INNER_INTERVAL_WIDTH = 8.0


def _plot_correlation_interval(ax: plt.Axes, interval: xr.DataArray, data_value: float) -> None:
    ax.hlines(
        y=0,
        xmin=float(interval.sel(quantile=0.03)),
        xmax=float(interval.sel(quantile=0.97)),
        color=PALETTE["primary"],
        lw=OUTER_INTERVAL_WIDTH,
    )
    ax.hlines(
        y=0,
        xmin=float(interval.sel(quantile=0.25)),
        xmax=float(interval.sel(quantile=0.75)),
        color=PALETTE["primary"],
        lw=INNER_INTERVAL_WIDTH,
    )
    ax.scatter(
        x=float(interval.sel(quantile=0.5)),
        y=0,
        color=PALETTE["primary"],
        edgecolors="white",
        s=90,
        linewidths=1.0,
        zorder=3,
    )
    ax.scatter(x=data_value, y=0, facecolors="none", edgecolors="k", s=45, linewidths=1.2, zorder=4)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), yticks=[])
    ax.grid(visible=False, axis="y")


def plot_moment_grid(
    predictive_moments: xr.Dataset, data_moments: xr.Dataset, label: str, short_names: Mapping[str, str]
) -> plt.Figure:
    """
    Draw densities of each observable's sample sd on the diagonal, and intervals of each correlation below it.

    Parameters
    ----------
    predictive_moments, data_moments : Dataset
        Moments of the simulated observables and of the data: ``std`` over ``variable``, and ``acf`` over ``lag``,
        ``variable`` and ``variable_aux``. The predictive moments also carry a ``sample`` dimension.
    label : str
        Name of the predictive distribution in the legend, such as "prior predictive".
    short_names : dict of str to str
        The observables to plot, in order, mapped to the short names their axes are labeled with.

    Returns
    -------
    figure : Figure
        The grid of panels.

    Examples
    --------
    Two observables, with 50 simulated samples against the data:

    .. code-block:: python

        import numpy as np
        import xarray as xr

        from climate_risk.plotting import plot_moment_grid

        rng = np.random.default_rng(0)
        names = ["a", "b"]
        coords = {"lag": [0], "variable": names, "variable_aux": names}
        data_moments = xr.Dataset(
            {
                "std": ("variable", [0.01, 0.02]),
                "acf": (("lag", "variable", "variable_aux"), rng.uniform(-1, 1, (1, 2, 2))),
            },
            coords=coords,
        )
        predictive_moments = xr.Dataset(
            {
                "std": (("sample", "variable"), rng.uniform(0.005, 0.03, (50, 2))),
                "acf": (("sample", "lag", "variable", "variable_aux"), rng.uniform(-1, 1, (50, 1, 2, 2))),
            },
            coords=coords,
        )
        figure = plot_moment_grid(
            predictive_moments=predictive_moments,
            data_moments=data_moments,
            label="posterior predictive",
            short_names={"a": "A", "b": "B"},
        )
    """
    observables = list(short_names)
    predictive_std_percent = 100 * predictive_moments["std"]
    data_std_percent = 100 * data_moments["std"].to_series()
    correlation_quantiles = predictive_moments["acf"].sel(lag=0).quantile([0.03, 0.25, 0.5, 0.75, 0.97], dim="sample")
    data_correlation = data_moments["acf"].sel(lag=0)

    figure, axes = plt.subplots(nrows=len(observables), ncols=len(observables), figsize=(17, 15), layout="constrained")
    for row, row_name in enumerate(observables):
        for column, column_name in enumerate(observables):
            ax = axes[row, column]
            if column > row:
                ax.set_axis_off()
                continue
            if column == row:
                std = predictive_std_percent.sel(variable=row_name)
                plot_density_against_data(ax=ax, values=std, data_value=float(data_std_percent[row_name]))
                ax.set_xlim(0, max(float(std.quantile(0.99)), 1.2 * float(data_std_percent[row_name])))
                ax.set_title(f"sd({short_names[row_name]}), %", loc="left", fontsize=10)
            else:
                _plot_correlation_interval(
                    ax=ax,
                    interval=correlation_quantiles.sel(variable=row_name, variable_aux=column_name),
                    data_value=float(data_correlation.sel(variable=row_name, variable_aux=column_name)),
                )
            if column == 0 and row > 0:
                ax.set_ylabel(short_names[row_name], rotation=0, ha="right", va="center")
            if row == len(observables) - 1:
                ax.set_xlabel(short_names[column_name])

    legend_handles = [
        plt.Line2D(
            [],
            [],
            color=PALETTE["primary"],
            marker="o",
            markersize=9,
            markeredgecolor="white",
            lw=INNER_INTERVAL_WIDTH,
            label=f"{label} median and 50%",
        ),
        plt.Line2D([], [], color=PALETTE["primary"], lw=OUTER_INTERVAL_WIDTH, label=f"{label} 94%"),
        plt.Line2D([], [], color="k", marker="o", markerfacecolor="none", lw=0.0, label="data correlation"),
        plt.Line2D([], [], color="k", ls="--", lw=1.5, label="data standard deviation"),
    ]
    figure.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(legend_handles), frameon=False
    )

    return figure


def plot_clipped_densities(
    samples: xr.Dataset,
    colors: Sequence[str],
    xlabel: str,
    title: str,
    interval_probability: float = 0.98,
    mean_labels: Literal["curve", "legend"] = "curve",
) -> plt.Axes:
    """
    Overlay the density of each variable in ``samples`` over its HDI, with each mean marked and labeled.

    Parameters
    ----------
    samples : Dataset
        One variable per density, with a ``sample`` dimension.
    colors : sequence of str
        One color per variable.
    xlabel, title : str
        Axis label and figure title.
    interval_probability : float, optional
        Probability of the HDI each density is clipped to. Default 0.98.
    mean_labels : {"curve", "legend"}, optional
        Label each mean above the curves, or in the legend entries when the means sit too close together to label
        apart. Default "curve".

    Returns
    -------
    ax : Axes
        The axes holding the densities.

    Examples
    --------
    Two welfare costs overlaid, each clipped to its 98% HDI:

    .. code-block:: python

        import numpy as np
        import xarray as xr

        from climate_risk.plotting import plot_clipped_densities

        rng = np.random.default_rng(0)
        samples = xr.Dataset(
            {"consumption": ("sample", rng.normal(-0.4, 0.1, 1000)), "hours": ("sample", rng.normal(0.1, 0.05, 1000))}
        )
        ax = plot_clipped_densities(
            samples=samples, colors=["tab:blue", "tab:orange"], xlabel="welfare cost (%)", title="Welfare cost"
        )
    """
    if mean_labels == "legend":
        samples = samples.rename({name: f"{name}, mean {float(samples[name].mean()):.3g}" for name in samples})

    intervals = az.hdi(samples, prob=interval_probability, dim=["sample"])
    densities = az.kde(samples, dim="sample")
    grid = densities.sel(plot_axis="x")
    inside_interval = (grid >= intervals.sel(ci_bound="lower")) & (grid <= intervals.sel(ci_bound="upper"))
    clipped = xr.Dataset(
        {name: density.where(inside_interval[name]).assign_attrs(kind="kde") for name, density in densities.items()}
    )

    plot = az.plot_dist(
        samples,
        sample_dims=["sample"],
        kind="kde",
        ci_kind="hdi",
        ci_prob=interval_probability,
        point_estimate="mean",
        cols=[],
        aes={"color": ["__variable__"]},
        aes_by_visuals={"point_estimate": ["color"]},
        color=list(colors),
        stats={"dist": clipped},
        visuals={"face": True, "credible_interval": False, "point_estimate_text": False, "title": False},
        figure_kwargs={"figsize": (9, 4)},
    )
    plot.add_legend(
        "__variable__",
        title=f"{interval_probability:.0%} HDI",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(samples.data_vars),
        frameon=False,
    )

    figure: plt.Figure = plot.viz["figure"].item()
    ax = figure.axes[0]
    if mean_labels == "curve":
        density_x = clipped.to_array("component").sel(plot_axis="x").to_numpy()
        density_y = clipped.to_array("component").sel(plot_axis="y").to_numpy()
        label_half_width = 0.05 * (np.nanmax(density_x) - np.nanmin(density_x))
        label_pad = 0.03 * np.nanmax(density_y)
        for name, color in zip(samples, colors, strict=True):
            mean = float(samples[name].mean())
            near_mean = np.abs(density_x - mean) <= label_half_width
            label_height = np.nanmax(np.where(near_mean, density_y, np.nan)) + label_pad
            ax.vlines(x=mean, ymin=0.0, ymax=label_height, color=color, lw=0.8, ls=":")
            ax.text(mean, label_height, f"mean {mean:.3g}", ha="center", va="bottom", color=color, fontsize=9)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")

    return ax
