import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import scipy.optimize

from ptgp.gp import SVGP, init_variational_params
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Poisson
from ptgp.optim.training import compile_scipy_objective, get_trained_params

from climate_risk.exceptions import DataValidationError
from climate_risk.models.aggregation import build_aggregation, build_aggregation_from_overlaps
from climate_risk.models.areal import windowed_poisson_elbo

CELLS_PER_AXIS = 16
INDUCING_PER_AXIS = 4
TRUE_COEFFICIENTS = np.array([0.9, -0.6])
TRUE_INTERCEPT = 4.0


def square_grid(cells_per_axis=CELLS_PER_AXIS):
    """Cell centres and equal weights over the unit square."""
    edges = np.linspace(0.0, 1.0, cells_per_axis + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    x, y = (axis.ravel() for axis in np.meshgrid(centres, centres, indexing="ij"))

    return np.column_stack([x, y]), np.full(x.shape, 1.0 / cells_per_axis**2)


def windows_of(coordinates, units_per_axis):
    """Label each cell with the square block it falls in."""
    column = np.minimum((coordinates[:, 0] * units_per_axis).astype(int), units_per_axis - 1)
    row = np.minimum((coordinates[:, 1] * units_per_axis).astype(int), units_per_axis - 1)

    return [f"u{r:02d}{c:02d}" for r, c in zip(row, column, strict=True)]


def linear_mean_process(covariates, coordinates, inducing, *, lengthscale):
    """A regression mean under a spatial field, with the priors written where a reader can see them."""
    with pm.Model() as model:
        coefficients = pm.Normal("coefficients", 0.0, 1.0, shape=(covariates.shape[1],))
        intercept = pm.Normal("intercept", 0.0, 5.0)
        amplitude = pm.HalfNormal("amplitude", 1.0)
        svgp = SVGP(
            kernel=amplitude**2 * ExpQuad(input_dim=coordinates.shape[1], ls=lengthscale),
            likelihood=Poisson(),
            mean=lambda _: intercept + pt.dot(pt.as_tensor_variable(covariates), coefficients),
            inducing_variable=Points(Z=inducing),
            variational_params=init_variational_params(inducing.shape[0]),
            whiten=True,
        )

    return model, svgp


def test_the_coefficients_come_back_from_simulated_counts():
    """The covariates are what identify the surface inside a window, so a fit that cannot recover
    them from counts it generated itself has nothing to say about sub-window structure.

    The covariates are spatially rough on purpose. A smooth one is a function of position and the
    field can imitate it, which is the confounding a real fit has to live with; a rough one is
    separable and says whether the linear part is being fitted at all.
    """
    rng = np.random.default_rng(0)
    coordinates, areas = square_grid()
    covariates = rng.standard_normal((coordinates.shape[0], 2))

    log_intensity = TRUE_INTERCEPT + covariates @ TRUE_COEFFICIENTS
    labels = windows_of(coordinates, units_per_axis=8)
    aggregation = build_aggregation(unit_of_cell=labels, weights=areas)
    region = build_aggregation(unit_of_cell=["place"] * len(labels), weights=areas)
    counts = rng.poisson(aggregation.aggregate(np.exp(log_intensity))).astype(float)

    axis_points = np.linspace(0.1, 0.9, INDUCING_PER_AXIS)
    inducing = np.column_stack([axis.ravel() for axis in np.meshgrid(axis_points, axis_points, indexing="ij")])
    model, svgp = linear_mean_process(covariates, coordinates, inducing, lengthscale=0.3)

    with model:

        def objective(gp, cell_features, window_counts):
            return windowed_poisson_elbo(gp, cell_features, window_counts, aggregation, region, n_draws=16, seed=1)

        loss_and_grad, start, unpack, shared_params, _ = compile_scipy_objective(
            objective, svgp, pt.matrix("X", shape=(None, 2)), pt.vector("y", shape=(None,)), model=model
        )
        fitted = scipy.optimize.minimize(
            loss_and_grad,
            start,
            args=(coordinates, counts),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 300},
        )
        unpack(fitted.x)
        recovered = get_trained_params(model, shared_params)["coefficients"]

    assert np.allclose(recovered, TRUE_COEFFICIENTS, atol=0.25), f"recovered {recovered} for {TRUE_COEFFICIENTS}"


def test_the_region_term_does_not_depend_on_how_the_windows_are_drawn():
    """The exponential term is the intensity over the place, so it is the same whether the place is
    described by four windows or by four plus a coarse one laid over two of them. Summing the
    windows instead would count the overlap twice and the ground no window covers not at all.
    """
    coordinates, areas = square_grid(cells_per_axis=4)
    region = build_aggregation(unit_of_cell=["place"] * 16, weights=areas)
    inducing = coordinates[::4]

    partitioning = windows_of(coordinates, units_per_axis=2)
    nested = build_aggregation_from_overlaps(
        unit_of_overlap=[*partitioning, *(["coarse"] * 8)],
        cell_of_overlap=np.concatenate([np.arange(16), np.arange(8)]),
        weights=np.concatenate([areas, areas[:8]]),
        n_cells=16,
    )
    flat = build_aggregation(unit_of_cell=partitioning, weights=areas)

    scores = []
    for aggregation in (flat, nested):
        model, svgp = linear_mean_process(np.zeros((16, 1)), coordinates, inducing, lengthscale=0.3)
        counts = np.zeros(aggregation.n_units)
        with model:

            def objective(gp, cell_features, window_counts, aggregation=aggregation):
                return windowed_poisson_elbo(gp, cell_features, window_counts, aggregation, region, seed=1)

            loss_and_grad, start, *_ = compile_scipy_objective(
                objective, svgp, pt.matrix("X", shape=(None, 2)), pt.vector("y", shape=(None,)), model=model
            )
            scores.append(float(loss_and_grad(start, coordinates, counts)[0]))

    assert scores[0] == pytest.approx(scores[1]), "the region term moved when the windows were redrawn"


def test_a_region_operator_spanning_several_rows_is_an_error():
    """The exponential term is the intensity over the whole place once. Several rows would subtract
    it several times and pull every count down without touching the fit's shape."""
    coordinates, areas = square_grid(cells_per_axis=4)
    labels = windows_of(coordinates, units_per_axis=2)
    aggregation = build_aggregation(unit_of_cell=labels, weights=areas)
    split_region = build_aggregation(unit_of_cell=["north"] * 8 + ["south"] * 8, weights=areas)

    model, svgp = linear_mean_process(np.zeros((16, 1)), coordinates, coordinates[::4], lengthscale=0.3)

    with model, pytest.raises(DataValidationError, match="one row"):
        windowed_poisson_elbo(svgp, pt.as_tensor_variable(coordinates), pt.ones(4), aggregation, split_region)


def test_the_region_and_the_windows_must_span_the_same_grid():
    """Both operators index cells by position. Built against different grids they would still
    multiply, and the exponential term would be subtracted over the wrong ground."""
    coordinates, areas = square_grid(cells_per_axis=4)
    aggregation = build_aggregation(unit_of_cell=windows_of(coordinates, units_per_axis=2), weights=areas)
    smaller = build_aggregation(unit_of_cell=["place"] * 9, weights=areas[:9])

    model, svgp = linear_mean_process(np.zeros((16, 1)), coordinates, coordinates[::4], lengthscale=0.3)

    with model, pytest.raises(DataValidationError, match="index the same grid"):
        windowed_poisson_elbo(svgp, pt.as_tensor_variable(coordinates), pt.ones(4), aggregation, smaller)


def test_the_objective_is_the_same_twice_for_one_seed():
    """The log-intensity term is sampled, and L-BFGS assumes the objective is a function of the
    parameters alone. Redrawing per call turns a converged fit into a random walk."""
    coordinates, areas = square_grid(cells_per_axis=4)
    aggregation = build_aggregation(unit_of_cell=windows_of(coordinates, units_per_axis=2), weights=areas)
    region = build_aggregation(unit_of_cell=["place"] * 16, weights=areas)
    counts = np.array([2.0, 0.0, 1.0, 3.0])

    scores = []
    for _ in range(2):
        model, svgp = linear_mean_process(np.zeros((16, 1)), coordinates, coordinates[::4], lengthscale=0.3)
        with model:

            def objective(gp, cell_features, window_counts):
                return windowed_poisson_elbo(gp, cell_features, window_counts, aggregation, region, seed=7)

            loss_and_grad, start, *_ = compile_scipy_objective(
                objective, svgp, pt.matrix("X", shape=(None, 2)), pt.vector("y", shape=(None,)), model=model
            )
            scores.append(float(loss_and_grad(start, coordinates, counts)[0]))

    assert scores[0] == scores[1]
