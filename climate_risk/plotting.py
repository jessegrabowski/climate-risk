from collections.abc import Mapping, Sequence
from typing import Any, Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import AnchoredText
from scipy import stats

REGIONS = ("Asia", "Europe", "Africa", "Oceania", "Americas")


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
    color: str = "tab:blue",
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
        Color of the bands and the median line. Default 'tab:blue'.
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
        axis.plot(observed.index, observed.to_numpy(), color="0.15", lw=1.0, ls="--")

    if divider is not None:
        axis.axvline(divider, color="0.35", ls=":", lw=0.9)

    axis.margins(x=0)


def _plot_single_kde(
    data: pd.Series,
    axis: plt.Axes | None = None,
    bins: int = 30,
    color: str = "tab:blue",
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
        Fill color of the histogram. Default ``"tab:blue"``.
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
    color: str = "tab:blue",
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
        Fill color of the histograms. Default ``"tab:blue"``.
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

        figure = plot_aggregated_series(panel, ["damage"], "year", "sum")
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

        figure = plot_aggregated_series_by_region(panel, ["damage"], "year", "sum")
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
        data["Start_Year"],
        data["predictions"],
        zorder=1000,
        color="tab:red",
        label="Mean Predicted Disaster Count",
    )
    ax.scatter(data["Start_Year"], data["is_disaster"], color="k", label="Actual prob")
    ax.fill_between(
        data["Start_Year"],
        data["higher_y_hat_95"],
        data["lower_y_hat_95"],
        alpha=0.25,
        color="tab:blue",
        label="95% HDI",
    )
    ax.fill_between(
        data["Start_Year"],
        data["lower_y_hat_50"],
        data["higher_y_hat_50"],
        alpha=0.5,
        color="tab:blue",
        label="50% HDI",
    )
    ax.legend(loc="upper left")

    ax.set_xlabel("Start_Year")
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
        data["year"],
        (data[target_variable].astype(float)),
        color="k",
        label=("Real hydrometereological events damage in millions of dollars"),
    )
    ax.fill_between(
        data["year"],
        data["higher_damage_75"],
        data["lower_damage_75"],
        alpha=0.25,
        color="tab:blue",
        label="75% HDI",
    )
    ax.fill_between(
        data["year"],
        data["lower_damage_50"],
        data["higher_damage_50"],
        alpha=0.5,
        color="tab:blue",
        label="50% HDI",
    )
    ax.legend(loc="upper left", fontsize=14)
    ax.set_xlabel("year", fontsize=14)
    ax.set_ylabel("Disaster damages in 2000 USD millions", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    return fig
