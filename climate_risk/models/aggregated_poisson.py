import numpy as np
import pytensor.tensor as pt

from ptgp.gp import SVGP
from pytensor import shared
from pytensor.tensor import TensorVariable

from climate_risk.exceptions import DataValidationError
from climate_risk.models.aggregation import Aggregation


def latent_moments(svgp: SVGP, cell_features: TensorVariable) -> tuple[TensorVariable, TensorVariable, TensorVariable]:
    r"""
    Posterior mean and covariance factors of the latent field at the raster cells.

    The posterior covariance is :math:`\mathrm{diag}(d) + B B^T`, dropping the off-diagonal of the
    Nyström residual as the marginal path does. ``predict_marginal`` assembles the diagonal and
    discards the factors, so both are recomputed here.

    Parameters
    ----------
    svgp : SVGP
        A whitened ptgp SVGP. The unwhitened parameterization is not supported.
    cell_features : TensorVariable
        Shape ``(n_cells, n_features)``.

    Returns
    -------
    mean : TensorVariable
        Shape ``(n_cells,)``.
    independent_variance : TensorVariable
        Shape ``(n_cells,)``, the prior variance the inducing points do not explain.
    factor : TensorVariable
        Shape ``(n_cells, n_inducing)``, the variational covariance factor.
    """
    if not svgp.whiten:
        raise DataValidationError("The aggregated objective is derived for a whitened SVGP.")

    inducing, kernel = svgp.inducing_variable, svgp.kernel
    cross = inducing.K_uf(kernel, cell_features)
    whitened = inducing.Kuu_sqrt_solve(kernel, cross)

    mean = whitened.T @ svgp.q_mu + svgp.mean(cell_features)
    independent_variance = kernel.diag(cell_features) - pt.sum(whitened**2, axis=0)
    factor = whitened.T @ svgp.q_sqrt

    return mean, independent_variance, factor


def expected_intensity(
    aggregation: Aggregation,
    mean: TensorVariable,
    independent_variance: TensorVariable,
    factor: TensorVariable,
) -> TensorVariable:
    r"""
    :math:`\mathbb{E}_q[\Lambda_A]` for :math:`\Lambda_A = \sum_{i \in A} w_i e^{f_i}`.

    Exact, and needs only the marginal variances: the expectation of a sum does not see the
    correlation between cells.

    Parameters
    ----------
    aggregation : Aggregation
        The cell-to-unit operator.
    mean : TensorVariable
        Shape ``(n_cells,)``.
    independent_variance : TensorVariable
        Shape ``(n_cells,)``.
    factor : TensorVariable
        Shape ``(n_cells, n_inducing)``.

    Returns
    -------
    intensity : TensorVariable
        Shape ``(n_units,)``.

    Examples
    --------
    .. code-block:: python

        from climate_risk.models.aggregated_poisson import expected_intensity

        intensity = expected_intensity(aggregation, mean, independent_variance, factor)
    """
    variance = independent_variance + pt.sum(factor**2, axis=1)
    log_scaled = mean + variance / 2.0

    shift = pt.max(log_scaled)

    expected: TensorVariable = aggregation.aggregate_symbolic(pt.exp(log_scaled - shift)) * pt.exp(shift)

    return expected


def sample_log_intensity(
    aggregation: Aggregation,
    mean: TensorVariable,
    independent_variance: TensorVariable,
    factor: TensorVariable,
    inducing_draws: np.ndarray | TensorVariable,
    cell_draws: np.ndarray | TensorVariable,
) -> TensorVariable:
    r"""
    :math:`\log \Lambda_A` at each of a fixed set of draws from the variational posterior.

    :math:`\mathbb{E}_q[\log \Lambda_A]` has no closed form, and a sum over cells is not a function
    of any one marginal, so it is estimated by sampling the field. The draws are reparameterized
    through :math:`f = m + B \varepsilon + \sqrt{d}\,\eta`, which is exact for the factored
    covariance and puts the sampling noise outside the gradient.

    Parameters
    ----------
    aggregation : Aggregation
        The cell-to-unit operator.
    mean : TensorVariable
        Shape ``(n_cells,)``.
    independent_variance : TensorVariable
        Shape ``(n_cells,)``.
    factor : TensorVariable
        Shape ``(n_cells, n_inducing)``.
    inducing_draws : ndarray or TensorVariable
        Shape ``(n_inducing, n_draws)``, standard normal.
    cell_draws : ndarray or TensorVariable
        Shape ``(n_cells, n_draws)``, standard normal.

    Returns
    -------
    log_intensity : TensorVariable
        Shape ``(n_units, n_draws)``.
    """
    spread = pt.sqrt(pt.clip(independent_variance, 0.0, np.inf))
    field = (
        mean[:, None]
        + factor @ pt.as_tensor_variable(inducing_draws)
        + spread[:, None] * pt.as_tensor_variable(cell_draws)
    )

    # Shifted before exponentiating: the optimizer visits fields far from the data early on, and a
    # unit whose cells all underflow would otherwise take log of zero and poison the gradient.
    shift = pt.max(field, axis=0)
    totals = aggregation.aggregate_symbolic(pt.exp(field - shift))

    log_totals: TensorVariable = pt.log(totals) + shift

    return log_totals


def mean_log_intensity(
    svgp: SVGP,
    aggregation: Aggregation,
    moments: tuple[TensorVariable, TensorVariable, TensorVariable],
    *,
    n_draws: int = 16,
    seed: int = 0,
) -> TensorVariable:
    r"""
    :math:`\mathbb{E}_q[\log \Lambda_A]` for each row of the operator, estimated on fixed draws.

    The draws are fixed once so the objective stays deterministic and a quasi-Newton optimizer can
    be used on it.

    Parameters
    ----------
    svgp : SVGP
        A whitened ptgp SVGP over the cell features.
    aggregation : Aggregation
        The cell-to-row operator whose totals are being logged.
    moments : tuple of TensorVariable
        The mean, independent variance and factor from :func:`latent_moments`.
    n_draws : int, optional
        Draws behind the estimate. Default 16.
    seed : int, optional
        Seed for the fixed draw matrices. Default 0.

    Returns
    -------
    TensorVariable
        Shape ``(aggregation.n_units,)``.
    """
    rng = np.random.default_rng(seed)
    # Shared rather than constant: at raster scale the cell draws are far too large to fold into
    # the graph, and numba refuses to cache a function carrying one.
    inducing_draws = shared(rng.standard_normal((svgp.inducing_variable.num_inducing, n_draws)), name="inducing_draws")
    cell_draws = shared(rng.standard_normal((aggregation.n_cells, n_draws)), name="cell_draws")

    estimate: TensorVariable = pt.mean(
        sample_log_intensity(aggregation, *moments, inducing_draws, cell_draws),
        axis=1,
    )

    return estimate


def aggregated_poisson_elbo(
    svgp: SVGP,
    cell_features: TensorVariable,
    unit_counts: TensorVariable,
    aggregation: Aggregation,
    *,
    n_draws: int = 16,
    seed: int = 0,
) -> TensorVariable:
    r"""
    The variational objective for counts observed as polygon totals of a cell-level intensity.

    Replaces ptgp's own ELBO rather than supplying a ``Likelihood``: that interface pairs
    observation :math:`i` with latent point :math:`i` and scales by the number of latent points,
    and an aggregated observation satisfies neither.

    :math:`\mathbb{E}_q[\Lambda_A]` is closed form and only :math:`\mathbb{E}_q[\log \Lambda_A]` is
    sampled, by :func:`mean_log_intensity`.

    Parameters
    ----------
    svgp : SVGP
        A whitened ptgp SVGP over the cell features.
    cell_features : TensorVariable
        Shape ``(n_cells, n_features)``.
    unit_counts : TensorVariable
        Shape ``(n_units,)``, in the row order of ``aggregation.units``.
    aggregation : Aggregation
        The cell-to-unit operator.
    n_draws : int, optional
        Draws behind the log-intensity term. Default 16.
    seed : int, optional
        Seed for the fixed draw matrices. Default 0.

    Returns
    -------
    elbo : TensorVariable
        Scalar.

    Examples
    --------
    Fit counts observed per unit against a latent field on the cell grid:

    .. code-block:: python

        import pymc as pm

        from climate_risk.models.aggregated_poisson import aggregated_poisson_elbo

        with pm.Model() as model:
            elbo = aggregated_poisson_elbo(svgp, cell_features, unit_counts, aggregation)
            pm.Potential("elbo", elbo)
    """
    moments = latent_moments(svgp, cell_features)
    expected = expected_intensity(aggregation, *moments)
    log_intensity = mean_log_intensity(svgp, aggregation, moments, n_draws=n_draws, seed=seed)

    log_likelihood = unit_counts * log_intensity - expected - pt.gammaln(unit_counts + 1.0)

    elbo: TensorVariable = pt.sum(log_likelihood) - svgp.prior_kl()

    return elbo
