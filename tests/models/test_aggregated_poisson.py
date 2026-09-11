from typing import NamedTuple

import numpy as np
import ptgp
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.optimize

from ptgp import FitResult
from ptgp.gp import SVGP, init_variational_params
from ptgp.inducing import Points
from ptgp.kernels import ExpQuad
from ptgp.likelihoods import Poisson
from ptgp.optim.training import compile_scipy_objective, get_trained_params
from pytensor.graph.traversal import explicit_graph_inputs
from pytensor.tensor.special import gammaln

from climate_risk.exceptions import DataValidationError
from climate_risk.models.aggregated_poisson import (
    aggregated_poisson_elbo,
    expected_intensity,
    latent_moments,
    sample_log_intensity,
)
from climate_risk.models.aggregation import build_aggregation

N_CELLS, N_INDUCING = 12, 4
CELLS_PER_AXIS, INDUCING_PER_AXIS = 24, 6


def latent_pieces(seed: int):
    """A mean, an independent variance, and a covariance factor, standing in for an SVGP fit."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(0.0, 0.4, size=N_CELLS)
    independent_variance = rng.uniform(0.05, 0.3, size=N_CELLS)
    factor = rng.normal(0.0, 0.3, size=(N_CELLS, N_INDUCING))

    return mean, independent_variance, factor


def brute_force_intensities(mean, independent_variance, factor, weights, unit_of_cell, units, n_draws, seed):
    """Draw the field from the covariance the factors describe and aggregate it with numpy."""
    rng = np.random.default_rng(seed)
    covariance = np.diag(independent_variance) + factor @ factor.T
    chol = np.linalg.cholesky(covariance + 1e-10 * np.eye(len(mean)))
    field = mean[:, None] + chol @ rng.standard_normal((len(mean), n_draws))

    rows = np.array([units.index(unit) for unit in unit_of_cell])
    contributions = weights[:, None] * np.exp(field)

    return np.stack([contributions[rows == row].sum(axis=0) for row in range(len(units))])


def test_the_expected_intensity_matches_sampling_the_field():
    """E[Lambda] is exact and needs only the marginals, so it must agree with sampling the full
    factored covariance. Aggregating before exponentiating, or dropping the variance correction,
    both show up as a bias here.
    """
    mean, independent_variance, factor = latent_pieces(seed=0)
    weights = np.linspace(0.5, 1.5, N_CELLS)
    unit_of_cell = ["a"] * 5 + ["b"] * 7
    aggregation = build_aggregation(unit_of_cell=unit_of_cell, weights=weights)

    closed_form = pytensor.function(
        [], expected_intensity(aggregation, pt.as_tensor_variable(mean), independent_variance, factor)
    )()
    drawn = brute_force_intensities(
        mean, independent_variance, factor, weights, unit_of_cell, ["a", "b"], n_draws=400_000, seed=1
    )

    np.testing.assert_allclose(closed_form, drawn.mean(axis=1), rtol=0.02)


def test_the_sampled_log_intensity_converges_on_the_truth():
    """The log term is the one with no closed form. Its estimator has to be centered on the real
    expectation, and the Jensen gap is the thing it exists to capture: log E[Lambda] is an upper
    bound that a correct estimator must sit strictly below.
    """
    mean, independent_variance, factor = latent_pieces(seed=2)
    weights = np.linspace(0.5, 1.5, N_CELLS)
    unit_of_cell = ["a"] * 5 + ["b"] * 7
    aggregation = build_aggregation(unit_of_cell=unit_of_cell, weights=weights)

    rng = np.random.default_rng(3)
    n_draws = 40_000
    inducing_draws, cell_draws = pt.matrix("inducing_draws"), pt.matrix("cell_draws")
    sampled = pytensor.function(
        [inducing_draws, cell_draws],
        sample_log_intensity(
            aggregation, pt.as_tensor_variable(mean), independent_variance, factor, inducing_draws, cell_draws
        ),
    )(rng.standard_normal((N_INDUCING, n_draws)), rng.standard_normal((N_CELLS, n_draws))).mean(axis=1)

    drawn = brute_force_intensities(
        mean, independent_variance, factor, weights, unit_of_cell, ["a", "b"], n_draws=400_000, seed=4
    )
    truth = np.log(drawn).mean(axis=1)
    jensen = np.log(drawn.mean(axis=1))

    np.testing.assert_allclose(sampled, truth, rtol=0.01)
    assert (sampled < jensen).all(), "the estimator must sit below the Jensen bound it corrects"


def test_the_draws_reproduce_the_within_unit_correlation():
    """A unit's cells are correlated through the shared inducing values, and that correlation is
    what the log term is sensitive to. Sampling each cell independently would leave the mean
    intact and the spread of Lambda far too small.
    """
    mean, independent_variance, factor = latent_pieces(seed=5)
    aggregation = build_aggregation(unit_of_cell=["a"] * N_CELLS, weights=np.ones(N_CELLS))

    rng = np.random.default_rng(6)
    n_draws = 20_000
    values = (rng.standard_normal((N_INDUCING, n_draws)), rng.standard_normal((N_CELLS, n_draws)))
    inducing_draws, cell_draws = pt.matrix("inducing_draws"), pt.matrix("cell_draws")

    def spread(factor_used):
        log_intensity = pytensor.function(
            [inducing_draws, cell_draws],
            sample_log_intensity(
                aggregation,
                pt.as_tensor_variable(mean),
                independent_variance,
                factor_used,
                inducing_draws,
                cell_draws,
            ),
        )(*values)
        return log_intensity.std()

    assert spread(factor) > 1.5 * spread(np.zeros_like(factor))


def test_an_unwhitened_svgp_is_refused():
    """The moment expressions are derived from the whitened conditional. Running them against the
    unwhitened parameterization would silently use the wrong covariance factor.
    """

    class Unwhitened:
        whiten = False

    with pytest.raises(DataValidationError, match="whitened"):
        latent_moments(Unwhitened(), pt.matrix("X"))


def small_svgp(features):
    """A whitened SVGP whose variational parameters are still symbolic, as ptgp leaves them."""
    rng = np.random.default_rng(7)

    return SVGP(
        kernel=ExpQuad(input_dim=features.shape[1], ls=0.5),
        likelihood=Poisson(),
        inducing_variable=Points(Z=rng.uniform(0.0, 1.0, size=(N_INDUCING, features.shape[1]))),
        variational_params=init_variational_params(N_INDUCING),
        whiten=True,
    )


def evaluate_elbo(seed: int, counts=(3.0, 5.0)):
    """The objective and its gradient at one point in the variational parameters."""
    rng = np.random.default_rng(8)
    features = rng.uniform(0.0, 1.0, size=(N_CELLS, 2))
    aggregation = build_aggregation(unit_of_cell=["a"] * 5 + ["b"] * 7, weights=np.ones(N_CELLS))

    elbo = aggregated_poisson_elbo(
        small_svgp(features),
        pt.as_tensor_variable(features),
        pt.as_tensor_variable(np.asarray(counts)),
        aggregation,
        n_draws=8,
        seed=seed,
    )
    parameters = list(explicit_graph_inputs([elbo]))
    # The claim is about the graph, not the backend. Compiling it is the expensive part, and
    # numba spends seconds on a graph this size for no extra coverage.
    gradients = pt.grad(elbo, parameters)
    assert isinstance(gradients, list), "a list of wrt variables gives a list of gradients"
    compiled = pytensor.function(parameters, [elbo, *gradients], mode="FAST_COMPILE")

    return compiled(*[rng.normal(0.0, 0.2, size=parameter.type.shape) for parameter in parameters])


def test_the_objective_and_its_gradient_are_finite_and_informative():
    """An optimizer is handed this, so a detached graph or a NaN is the failure that matters:
    L-BFGS reads a zero gradient as a converged fit rather than as an error, and the run ends
    looking successful.
    """
    value, *gradients = evaluate_elbo(seed=0)

    assert np.isfinite(value)
    assert all(np.all(np.isfinite(gradient)) for gradient in gradients)
    assert all(np.any(gradient != 0.0) for gradient in gradients), "every variational parameter moves the objective"


def test_the_same_seed_gives_the_same_objective():
    """The draws are fixed so the objective is a deterministic function of the parameters. Drawing
    afresh per evaluation would make it jitter and a quasi-Newton line search would fail on it.
    """
    first, *_ = evaluate_elbo(seed=0)
    again, *_ = evaluate_elbo(seed=0)
    other, *_ = evaluate_elbo(seed=1)

    assert first == again
    assert first != other, "a different seed draws a different estimate"


def test_the_objective_is_the_poisson_likelihood_less_the_divergence():
    """The three pieces are checked separately above; this pins how they are combined. Dropping
    the counts, or the divergence, still leaves a finite differentiable scalar that an optimizer
    would minimize happily into the wrong answer.
    """
    rng = np.random.default_rng(8)
    features = rng.uniform(0.0, 1.0, size=(N_CELLS, 2))
    aggregation = build_aggregation(unit_of_cell=["a"] * 5 + ["b"] * 7, weights=np.ones(N_CELLS))
    counts = np.array([3.0, 5.0])
    n_draws, seed = 8, 0

    svgp = small_svgp(features)
    cell_features = pt.as_tensor_variable(features)
    elbo = aggregated_poisson_elbo(
        svgp, cell_features, pt.as_tensor_variable(counts), aggregation, n_draws=n_draws, seed=seed
    )

    # The objective seeds its own draws; the same seed and the same order reproduces them.
    draws = np.random.default_rng(seed)
    inducing_draws = draws.standard_normal((N_INDUCING, n_draws))
    cell_draws = draws.standard_normal((aggregation.n_cells, n_draws))

    mean, independent_variance, factor = latent_moments(svgp, cell_features)
    expected = expected_intensity(aggregation, mean, independent_variance, factor)
    log_intensity = pt.mean(
        sample_log_intensity(aggregation, mean, independent_variance, factor, inducing_draws, cell_draws), axis=1
    )
    assembled = pt.sum(counts * log_intensity - expected - gammaln(counts + 1.0)) - svgp.prior_kl()

    parameters = list(explicit_graph_inputs([elbo, assembled]))
    compiled = pytensor.function(parameters, [elbo, assembled], mode="FAST_COMPILE")
    from_objective, from_pieces = compiled(*[rng.normal(0.0, 0.2, size=p.type.shape) for p in parameters])

    assert from_objective == pytest.approx(from_pieces)


class Recovery(NamedTuple):
    """How much of a known surface a fit got back.

    Parameters
    ----------
    correlation : float
        Correlation of the recovered log-intensity with the truth, over every cell.
    within_unit_correlation : float
        The same, after each cell has its own unit's mean removed. This is the only one that says
        anything about seeing inside a polygon; the other is dominated by between-unit signal.
    flat_correlation : float
        The same as ``correlation``, for a baseline spreading each unit's count evenly over its
        cells.
    flat_within_unit_spread : float
        Standard deviation of the baseline's within-unit residual, which is zero by construction.
    """

    correlation: float
    within_unit_correlation: float
    flat_correlation: float
    flat_within_unit_spread: float


def true_log_intensity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A surface no exponentiated-quadratic kernel can reproduce from its own basis, so recovering
    it is a claim about the fit rather than about the prior."""
    return 8.0 + 1.3 * np.sin(np.pi * x) * np.cos(2.0 / 3.0 * np.pi * y) - 0.8 * x


def recover_known_surface(*, units_per_axis: int, fixed_lengthscale: float | None = None) -> Recovery:
    """Observe a known surface through polygon totals, fit it, and score what comes back."""
    rng = np.random.default_rng(0)
    edges = np.linspace(0.0, 1.0, CELLS_PER_AXIS + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    x, y = (axis.ravel() for axis in np.meshgrid(centers, centers, indexing="ij"))
    areas = np.full(x.shape, 1.0 / CELLS_PER_AXIS**2)

    column = np.minimum((x * units_per_axis).astype(int), units_per_axis - 1)
    row = np.minimum((y * units_per_axis).astype(int), units_per_axis - 1)
    unit_of_cell = [f"u{r:02d}{c:02d}" for r, c in zip(row, column, strict=True)]

    features = np.column_stack([x, y])
    aggregation = build_aggregation(unit_of_cell=unit_of_cell, weights=areas)
    log_truth = true_log_intensity(x, y)
    counts = rng.poisson(aggregation.aggregate(np.exp(log_truth))).astype(float)

    axis_points = np.linspace(0.05, 0.95, INDUCING_PER_AXIS)
    inducing = np.column_stack([axis.ravel() for axis in np.meshgrid(axis_points, axis_points, indexing="ij")])

    with pm.Model() as model:
        lengthscale = (
            pt.constant(fixed_lengthscale, name="lengthscale")
            if fixed_lengthscale is not None
            else pm.HalfNormal("lengthscale", 0.5)
        )
        amplitude = pm.HalfNormal("amplitude", 2.0)
        offset = pm.Normal("offset", 7.0, 3.0)
        svgp = SVGP(
            kernel=amplitude**2 * ExpQuad(input_dim=2, ls=lengthscale),
            likelihood=Poisson(),
            mean=lambda X: pt.full((X.shape[0],), offset),
            inducing_variable=Points(Z=inducing),
            variational_params=init_variational_params(inducing.shape[0]),
            whiten=True,
        )

        def objective(gp, cell_features, unit_counts):
            return aggregated_poisson_elbo(gp, cell_features, unit_counts, aggregation, n_draws=32, seed=1)

        loss_and_grad, start, unpack, shared_params, shared_extras = compile_scipy_objective(
            objective, svgp, pt.matrix("X", shape=(None, 2)), pt.vector("y", shape=(None,)), model=model
        )
        result = scipy.optimize.minimize(
            loss_and_grad, start, args=(features, counts), jac=True, method="L-BFGS-B", options={"maxiter": 200}
        )
        unpack(result.x)
        fitted = FitResult(
            result=result,
            params=get_trained_params(model, shared_params),
            shared_params=shared_params,
            shared_extras=tuple(shared_extras),
            model=model,
        )

    posterior_mean, posterior_variance = ptgp.predict(svgp, features, fitted)
    log_recovered = posterior_mean + posterior_variance / 2.0

    rows = np.array([aggregation.units.index(unit) for unit in unit_of_cell])
    cells_per_unit = (CELLS_PER_AXIS // units_per_axis) ** 2
    log_flat = np.log(counts[rows] / (cells_per_unit * areas))

    labels = np.array(unit_of_cell)

    def within_unit(field: np.ndarray) -> np.ndarray:
        centered = np.empty_like(field)
        for label in set(unit_of_cell):
            inside = labels == label
            centered[inside] = field[inside] - field[inside].mean()

        return centered

    inside_truth = within_unit(log_truth)

    return Recovery(
        correlation=float(np.corrcoef(log_recovered, log_truth)[0, 1]),
        within_unit_correlation=float(np.corrcoef(within_unit(log_recovered), inside_truth)[0, 1]),
        flat_correlation=float(np.corrcoef(log_flat, log_truth)[0, 1]),
        flat_within_unit_spread=float(np.std(within_unit(log_flat))),
    )


@pytest.mark.slow
def test_a_variational_gp_recovers_structure_inside_a_polygon():
    """The premise the whole geographic model rests on. Spreading each unit's count evenly over its
    cells already correlates 0.89 with the truth, entirely from between-unit variation, so the
    aggregate number proves nothing — the baseline carries no within-unit signal at all, by
    construction. Against the residual it cannot see, the fit recovers most of the surface."""
    recovery = recover_known_surface(units_per_axis=12)

    assert recovery.flat_within_unit_spread == pytest.approx(0.0, abs=1e-12), "the baseline is blind inside a unit"
    assert recovery.flat_correlation > 0.85, (
        f"the baseline is a real competitor, not a strawman: was {recovery.flat_correlation:.3f}, 0.892 when written"
    )
    assert recovery.within_unit_correlation > 0.8, f"was {recovery.within_unit_correlation:.3f}, 0.888 when written"
    assert recovery.correlation > recovery.flat_correlation + 0.05


@pytest.mark.slow
def test_a_coarse_unit_learns_a_lengthscale_that_sees_nothing_inside_itself():
    """With 36 cells to a unit the lengthscale is fit from between-unit variation alone and comes
    back far too long, so the field is smooth where the truth is not. The aggregate correlation
    falls below the baseline, which is the honest reading: on coarse units this model is worse than
    spreading the totals, so the frequency model has to fix or bound the lengthscale rather than
    learn it. This test pins a limitation, and a fit that removed it would fail here."""
    recovery = recover_known_surface(units_per_axis=4)

    assert recovery.flat_correlation > 0.85, f"was {recovery.flat_correlation:.3f}, 0.933 when written"
    assert recovery.within_unit_correlation < 0.3, f"was {recovery.within_unit_correlation:.3f}, 0.099 when written"
    assert recovery.correlation < recovery.flat_correlation


@pytest.mark.slow
def test_fixing_the_lengthscale_restores_recovery_in_coarse_units():
    """The coarse-unit failure is the lengthscale being learned badly, not an inability to see
    inside a large polygon. Held at the truth's scale, the same fit on the same data recovers most
    of the within-unit surface."""
    recovery = recover_known_surface(units_per_axis=4, fixed_lengthscale=0.5)

    assert recovery.flat_correlation > 0.85, f"was {recovery.flat_correlation:.3f}, 0.933 when written"
    assert recovery.within_unit_correlation > 0.6, f"was {recovery.within_unit_correlation:.3f}, 0.767 when written"
    assert recovery.correlation > recovery.flat_correlation
