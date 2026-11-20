import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnchoredText

from climate_risk.plotting import (
    REGIONS,
    attach_count_predictions,
    attach_damage_predictions,
    configure_plot_style,
    panel_grid,
    plot_aggregated_series,
    plot_aggregated_series_by_region,
    plot_clipped_densities,
    plot_descriptive,
    plot_estimate,
    plot_fan,
    plot_impact_heatmap,
    plot_moment_grid,
    plot_predicted_counts,
    plot_predicted_damages,
    plot_response_grid,
    plot_shock_responses,
    prepare_gridspec_figure,
)


@pytest.fixture(autouse=True)
def matplotlib_defaults():
    """`configure_plot_style` mutates global rcParams, so a test inherits whatever ran before it."""
    plt.rcdefaults()

    yield

    plt.rcdefaults()


OBSERVATIONS = 4
ISO_CODES = ["AAA", "AAA", "BBB", "BBB"]
YEARS = pd.to_datetime(["1990", "1991", "1990", "1991"])


@pytest.fixture
def counts():
    return pd.DataFrame({"ISO": ISO_CODES, "date": YEARS, "is_disaster": [1, 0, 1, 1]})


@pytest.fixture
def damages():
    return pd.DataFrame({"ISO": ISO_CODES, "date": YEARS, "damage_millions": [1.0, 2.0, 3.0, 4.0]})


def _posterior_predictive(name):
    """Draws spread around a per-observation mean, so the HDI bounds are genuinely distinct."""
    rng = np.random.default_rng(0)
    draws = np.arange(OBSERVATIONS, dtype=float) + rng.normal(0, 1, size=(2, 50, OBSERVATIONS))
    variable = xr.DataArray(draws, dims=("chain", "draw", "obs"), coords={"obs": np.arange(OBSERVATIONS)})
    return xr.DataTree.from_dict({"posterior_predictive": xr.Dataset({name: variable})})


@pytest.fixture
def idata():
    return _posterior_predictive("y_hat")


@pytest.fixture
def damages_idata():
    return _posterior_predictive("damage_millions")


def test_predictions_are_attached_in_row_order(idata, counts):
    inputs = attach_count_predictions(idata, counts)

    assert set(counts.columns) <= set(inputs.columns)
    assert inputs["predictions"].is_monotonic_increasing


def test_hdi_bounds_bracket_the_prediction(idata, counts):
    """The bands are drawn as fill_between, so an inverted bound silently plots nothing."""
    inputs = attach_count_predictions(idata, counts)

    assert (inputs["lower_y_hat_95"] < inputs["predictions"]).all()
    assert (inputs["higher_y_hat_95"] > inputs["predictions"]).all()
    assert (inputs["lower_y_hat_50"] >= inputs["lower_y_hat_95"]).all()
    assert (inputs["higher_y_hat_50"] <= inputs["higher_y_hat_95"]).all()


def test_a_frame_of_the_wrong_length_is_rejected(idata, counts):
    """Aligning by position means a mismatched frame would otherwise plot the wrong country."""
    with pytest.raises(ValueError, match="observations but df has"):
        attach_count_predictions(idata, counts.iloc[:2])


def test_plotting_one_country_draws_only_its_observations(idata, counts):
    fig = plot_predicted_counts(idata, counts, "AAA")

    predicted_line = fig.axes[0].lines[0]
    assert len(np.asarray(predicted_line.get_xdata())) == 2


def test_damage_bounds_bracket_the_prediction(damages_idata, damages):
    inputs = attach_damage_predictions(damages_idata, damages)

    assert (inputs["lower_damage_75"] < inputs["predictions"]).all()
    assert (inputs["higher_damage_75"] > inputs["predictions"]).all()
    assert (inputs["lower_damage_50"] >= inputs["lower_damage_75"]).all()
    assert (inputs["higher_damage_50"] <= inputs["higher_damage_75"]).all()


def test_plotting_damages_draws_only_one_country(damages_idata, damages):
    fig = plot_predicted_damages(damages_idata, damages, "AAA", "damage_millions")

    observed_points = fig.axes[0].collections[0]
    assert len(np.asarray(observed_points.get_offsets())) == 2


def test_each_variable_lands_on_its_own_panel():
    """Panels are filled by position; the previous index lookup sent duplicate names to one axis."""
    df = pd.DataFrame({"year": [1990, 1991] * 3, "a": range(6), "b": range(6), "c": range(6)})

    fig = plot_aggregated_series(df, ["a", "b", "c"], index="year", aggregation="mean", title="t")

    drawn = [axis.get_title() for axis in fig.axes if axis.get_title()]
    assert drawn == ["a", "b", "c"]


def test_an_odd_number_of_variables_blanks_the_spare_panel():
    """A grid is two wide, so an odd count leaves an empty axis that would otherwise show as blank
    ticks and a frame."""
    df = pd.DataFrame({"year": [1990, 1991] * 3, "a": range(6), "b": range(6), "c": range(6)})

    fig = plot_aggregated_series(df, ["a", "b", "c"], index="year", aggregation="mean", title="t")

    assert not fig.axes[3].axison


PERIODS = 6
DATES = pd.date_range("2000", periods=PERIODS, freq="YS")


@pytest.fixture
def fan_draws():
    """Draws whose spread grows with time, so the bands are ordered and genuinely distinct."""
    rng = np.random.default_rng(0)
    spread = np.linspace(1.0, 4.0, PERIODS)
    values = np.arange(PERIODS, dtype=float) + rng.normal(0, 1, size=(2, 200, PERIODS)) * spread

    return xr.DataArray(values, dims=("chain", "draw", "time"), coords={"time": DATES})


def test_a_name_without_a_label_titles_itself():
    """Panels are keyed by name, so a missing entry must not silently produce an untitled axis."""
    figure, axes = panel_grid(names=["alpha", "beta"], labels={"alpha": "Alpha"}, units={"beta": "kg"})

    assert [axis.get_title() for axis in axes] == ["Alpha", "beta"]
    assert [axis.get_ylabel() for axis in axes] == ["", "kg"]
    plt.close(figure)


def test_every_name_gets_its_own_axis():
    figure, axes = panel_grid(names=list("abcde"), n_cols=2)

    assert len(axes) == 5
    assert len({id(axis) for axis in axes}) == 5
    plt.close(figure)


def test_the_bands_are_the_credible_intervals_they_claim(fan_draws):
    """The band is drawn as fill_between, which accepts inverted or wrong bounds silently."""
    figure, axis = plt.subplots()

    plot_fan(draws=fan_draws, axis=axis, probs=(0.5, 0.89))

    expected = {prob: az.hdi(fan_draws, prob=prob).transpose("ci_bound", ...).values for prob in (0.5, 0.89)}
    drawn = [np.asarray(collection.get_paths()[0].vertices) for collection in axis.collections]
    assert len(drawn) == 2

    widest = drawn[0][:, 1]
    assert np.isclose(widest.max(), expected[0.89].max())
    assert np.isclose(widest.min(), expected[0.89].min())
    plt.close(figure)


def test_the_widest_band_is_drawn_faintest(fan_draws):
    """Pairing the shades with the masses the other way round buries the narrow band and the fan
    reads as one flat block, so the alpha has to be tied to the width and not to the draw order."""
    figure, axis = plt.subplots()

    plot_fan(draws=fan_draws, axis=axis, probs=(0.5, 0.89), shades=(0.16, 0.34))

    spans = [
        (collection.get_alpha(), np.ptp(np.asarray(collection.get_paths()[0].vertices)[:, 1]))
        for collection in axis.collections
    ]
    faintest = min(spans)
    widest = max(spans, key=lambda entry: entry[1])
    assert faintest == widest
    plt.close(figure)


def test_the_median_line_tracks_the_draws(fan_draws):
    figure, axis = plt.subplots()

    plot_fan(draws=fan_draws, axis=axis)

    median_line = axis.lines[0]
    assert np.allclose(median_line.get_ydata(), fan_draws.median(dim=["chain", "draw"]).values)
    plt.close(figure)


def test_the_observed_series_is_drawn_over_the_bands(fan_draws):
    """Observed and forecast are on different indices; drawing the observed against the forecast
    dates would shift it silently."""
    observed = pd.Series(np.arange(3, dtype=float), index=pd.date_range("1997", periods=3, freq="YS"))
    figure, axis = plt.subplots()

    plot_fan(draws=fan_draws, axis=axis, observed=observed)

    observed_line = axis.lines[1]
    assert list(pd.to_datetime(np.asarray(observed_line.get_xdata()))) == list(observed.index)
    assert np.allclose(observed_line.get_ydata(), observed.to_numpy())
    plt.close(figure)


def test_no_divider_is_drawn_unless_asked(fan_draws):
    figure, axis = plt.subplots()

    plot_fan(draws=fan_draws, axis=axis)
    without = len(axis.lines)
    plot_fan(draws=fan_draws, axis=axis, divider=DATES[3])

    assert len(axis.lines) == 2 * without + 1
    plt.close(figure)


def test_every_region_is_drawn_on_each_panel():
    """One line per region per variable; a mis-scoped loop silently plots only the last region."""
    df = pd.DataFrame(
        {
            "year": [1990, 1991] * len(REGIONS),
            "Region": [region for region in REGIONS for _ in range(2)],
            "a": range(2 * len(REGIONS)),
            "b": range(2 * len(REGIONS)),
        }
    )

    fig = plot_aggregated_series_by_region(df, ["a", "b"], index="year", aggregation="mean", title="t")

    panels = [axis for axis in fig.axes if axis.get_title()]
    assert [axis.get_title() for axis in panels] == ["a", "b"]
    assert all(len(axis.get_lines()) == len(REGIONS) for axis in panels)


def test_the_grid_centers_a_short_last_row():
    """Five plots in three columns leaves two on the bottom row; left-aligning them looks broken."""
    _, locations = prepare_gridspec_figure(n_cols=3, n_plots=5)

    assert len(locations) == 5
    last_row_columns = [columns.start for _, columns in locations[3:]]
    assert last_row_columns == [1, 3], "the short row should be inset, not flush left"


def test_the_grid_is_exact_when_the_plots_fill_it():
    _, locations = prepare_gridspec_figure(n_cols=3, n_plots=6)

    assert len(locations) == 6
    assert [columns.start for _, columns in locations[:3]] == [0, 2, 4]


def test_configuring_the_style_turns_the_grid_on_and_off():
    """Every notebook opens with this call, and a silently ignored argument is invisible."""
    configure_plot_style(add_grid=True)
    assert plt.rcParams["axes.grid"] is True

    configure_plot_style()
    assert plt.rcParams["axes.grid"] is False


def test_the_style_leaves_the_axis_lines_the_ticks_sit_on():
    """Dropping all four spines leaves tick marks pointing at nothing, which reads as unfinished."""
    configure_plot_style()

    assert plt.rcParams["axes.spines.left"] is True
    assert plt.rcParams["axes.spines.bottom"] is True
    assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.right"] is False


def test_the_grid_runs_horizontally_only():
    """A vertical grid competes with the series on a time axis, where x is not a measurement."""
    configure_plot_style(add_grid=True)

    assert plt.rcParams["axes.grid.axis"] == "y"


def test_a_single_column_frame_plots_without_unwrapping_it_first():
    """A one-column DataFrame took the single-panel branch and reached `data.name`, which a frame
    does not have, so every caller with one variable raised AttributeError."""
    df = pd.DataFrame({"only": np.linspace(0.0, 1.0, 40)})

    axis = plot_descriptive(df, add_sum_box=False)

    assert isinstance(axis, plt.Axes), "one column returns the axis, not the figure"
    assert axis.get_title() == "only"


def test_one_panel_is_drawn_per_column():
    df = pd.DataFrame({name: np.linspace(0.0, 1.0, 40) for name in ("a", "b", "c")})

    fig = plot_descriptive(df, add_sum_box=False)

    assert isinstance(fig, plt.Figure), "several columns return the figure, not an axis"
    assert [axis.get_title() for axis in fig.axes] == ["a", "b", "c"]


def test_the_summary_box_reports_the_series_statistics():
    """The box reads six fields off `stats.describe`; taking them positionally put the wrong number
    against every label the moment scipy reordered them."""
    df = pd.DataFrame({"only": np.arange(50, dtype=float)})

    axis = plot_descriptive(df, add_sum_box=True)

    assert isinstance(axis, plt.Axes)
    box = next(child for child in axis.get_children() if isinstance(child, AnchoredText))
    reported = dict(
        (name.strip(), float(value))
        for name, _, value in (line.partition("=") for line in box.txt.get_text().splitlines())
    )
    assert reported["N"] == 50
    assert reported["Min"] == 0.0
    assert reported["Max"] == 49.0
    assert reported["Mean"] == pytest.approx(24.5)
    assert reported["Std"] == pytest.approx(np.arange(50).std(ddof=1), abs=1e-3)


def response_quantile_grid(shocks, variables):
    quantile_levels = [0.03, 0.5, 0.97]
    values = np.broadcast_to(np.array(quantile_levels)[None, :, None, None], (len(shocks), 3, 4, len(variables)))

    return xr.DataArray(
        values,
        dims=["shock", "quantile", "time", "variable"],
        coords={"shock": shocks, "quantile": quantile_levels, "time": np.arange(4), "variable": variables},
    )


def test_the_response_grid_has_a_row_per_shock_and_a_column_per_variable():
    quantiles = response_quantile_grid(shocks=["a", "b"], variables=["GDP", "C", "I"])

    fig = plot_response_grid(
        quantiles=quantiles,
        reference=None,
        shock_labels={"a": "shock a", "b": "shock b"},
        variable_labels={"GDP": "GDP (%)", "C": "consumption (%)", "I": "investment (%)"},
    )

    assert len(fig.axes) == 6
    assert [axis.get_title(loc="left") for axis in fig.axes[:3]] == ["GDP (%)", "consumption (%)", "investment (%)"]
    plt.close(fig)


def test_the_response_legend_names_the_interval_the_quantiles_span():
    """The band spans the outer quantiles it is given, so a 94% label on a 90% band would misreport it."""
    quantiles = response_quantile_grid(shocks=["a"], variables=["GDP"])

    fig = plot_response_grid(
        quantiles=quantiles, reference=None, shock_labels={"a": "shock a"}, variable_labels={"GDP": "GDP (%)"}
    )

    assert [text.get_text() for text in fig.legends[0].get_texts()] == ["median and 94% interval"]
    plt.close(fig)


def test_a_fixed_estimate_is_drawn_hollow_without_an_interval():
    figure, ax = plt.subplots()
    row = pd.Series({"value": 0.5, "lower": np.nan, "upper": np.nan, "status": "fixed"})

    plot_estimate(ax=ax, row=row, y_position=0.0, color="tab:blue")

    assert np.asarray(ax.collections[0].get_facecolor())[0, :3].tolist() == [1.0, 1.0, 1.0]
    assert len(ax.collections) == 1
    plt.close(figure)


def test_the_moment_grid_draws_only_the_lower_triangle():
    names = ["a", "b", "c"]
    rng = np.random.default_rng(0)
    acf = rng.uniform(-1, 1, size=(1, 3, 3))
    data_moments = xr.Dataset(
        {"std": ("variable", [0.01, 0.02, 0.03]), "acf": (("lag", "variable", "variable_aux"), acf)},
        coords={"lag": [0], "variable": names, "variable_aux": names},
    )
    predictive_moments = xr.Dataset(
        {
            "std": (("sample", "variable"), rng.uniform(0.005, 0.04, size=(50, 3))),
            "acf": (("sample", "lag", "variable", "variable_aux"), rng.uniform(-1, 1, size=(50, 1, 3, 3))),
        },
        coords={"lag": [0], "variable": names, "variable_aux": names},
    )

    fig = plot_moment_grid(
        predictive_moments=predictive_moments,
        data_moments=data_moments,
        label="prior predictive",
        short_names={"a": "A", "b": "B", "c": "C"},
    )

    assert sum(axis.axison for axis in fig.axes) == 6
    plt.close(fig)


def test_unused_response_panels_are_hidden():
    """Five variables fill two rows of four, and the three spare panels would otherwise show empty axes."""
    variables = ["GDP", "C", "I", "K", "L"]
    quantiles = response_quantile_grid(shocks=["a"], variables=variables).sel(shock="a")

    fig = plot_shock_responses(quantiles=quantiles, reference=None, variable_labels={name: name for name in variables})

    assert [axis.get_title(loc="left") for axis in fig.axes if axis.axison] == variables
    assert sum(not axis.axison for axis in fig.axes) == 3
    plt.close(fig)


def test_the_impact_heatmap_is_drawn_in_percent_with_each_source_marked():
    """The impact arrives as a share and the shared colorbar is in percent, so an unscaled mesh would read 100 times too
    small against it."""
    x, y = np.array([0.1, 0.5, 0.9]), np.array([1.0, 2.0])
    impact = xr.DataArray(
        [[0.01, 0.0, -0.01], [0.02, 0.0, -0.02]], dims=["chi", "phi_H"], coords={"chi": y, "phi_H": x}
    )
    points = pd.DataFrame({"phi_H": [0.3], "chi": [1.5]}, index=["source A"])
    figure, ax = plt.subplots()

    mesh = plot_impact_heatmap(
        ax=ax,
        impact=impact,
        literature_points=points,
        literature_colors={"source A": "tab:green"},
        xlabel="habit",
        ylabel="inverse Frisch",
        yscale="linear",
        norm=Normalize(vmin=-2, vmax=2),
    )

    assert np.asarray(mesh.get_array()).ravel().tolist() == pytest.approx([1.0, 0.0, -1.0, 2.0, 0.0, -2.0])
    assert ax.get_legend_handles_labels()[1] == ["source A"]
    plt.close(figure)


def test_clipped_density_legend_carries_each_mean_when_asked():
    """Means too close to label on the curve go in the legend, which must name the right mean for each variable."""
    rng = np.random.default_rng(0)
    a_draws, b_draws = rng.normal(1.0, 0.1, 500), rng.normal(2.0, 0.1, 500)
    samples = xr.Dataset({"a": ("sample", a_draws), "b": ("sample", b_draws)})

    ax = plot_clipped_densities(
        samples=samples, colors=["tab:blue", "tab:orange"], xlabel="x", title="t", mean_labels="legend"
    )

    labels = [text.get_text() for text in ax.figure.legends[0].get_texts()]
    assert labels == [f"a, mean {a_draws.mean():.3g}", f"b, mean {b_draws.mean():.3g}"]
    plt.close(ax.get_figure(root=True))
