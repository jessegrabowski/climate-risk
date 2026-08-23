import numpy as np
import pytensor.tensor as pt

from ptgp.gp import SVGP
from pytensor.compile import shared
from pytensor.tensor import TensorVariable

from climate_risk.exceptions import DataValidationError
from climate_risk.models.aggregated_poisson import expected_intensity, latent_moments, sample_log_intensity
from climate_risk.models.aggregation import Aggregation


def windowed_poisson_elbo(
    svgp: SVGP,
    cell_features: TensorVariable,
    window_counts: TensorVariable,
    aggregation: Aggregation,
    region: Aggregation,
    *,
    n_draws: int = 16,
    seed: int = 0,
) -> TensorVariable:
    r"""
    The variational objective for events observed through windows that need not partition the place.

    Events are a point process whose locations are known only to lie somewhere in the window they
    were reported through, which gives

    .. math::

        \log L = -\Lambda_R + \sum_A y_A \log \Lambda_A

    one exponential term for the whole region and one log term per window. Windows may nest or
    overlap, which is what lets a province-level report sit beside a district-level one. Where they
    do partition the place this is the binned Poisson objective, up to the count factorials.

    Parameters
    ----------
    svgp : SVGP
        A whitened ptgp SVGP over the cell features.
    cell_features : TensorVariable
        Shape ``(n_cells, n_features)``.
    window_counts : TensorVariable
        Events observed through each window, in the row order of ``aggregation.units``.
    aggregation : Aggregation
        The cell-to-window operator.
    region : Aggregation
        A single-row operator over every cell the place holds, giving the term for the ground where
        nothing was recorded.
    n_draws : int, optional
        Draws behind the log-intensity term. Default 16.
    seed : int, optional
        Seed for the fixed draw matrices. Default 0.

    Returns
    -------
    TensorVariable
        Scalar.
    """
    if region.n_units != 1:
        raise DataValidationError(f"The region operator covers the place in one row, not {region.n_units}.")
    if region.n_cells != aggregation.n_cells:
        raise DataValidationError(
            f"The region spans {region.n_cells} cells and the windows {aggregation.n_cells}; they index the same grid."
        )

    mean, independent_variance, factor = latent_moments(svgp, cell_features)

    rng = np.random.default_rng(seed)
    # Shared rather than constant: at raster scale the cell draws are far too large to fold into
    # the graph, and numba refuses to cache a function carrying one.
    inducing_draws = shared(rng.standard_normal((svgp.inducing_variable.num_inducing, n_draws)), name="inducing_draws")
    cell_draws = shared(rng.standard_normal((aggregation.n_cells, n_draws)), name="cell_draws")

    log_intensity = pt.mean(
        sample_log_intensity(aggregation, mean, independent_variance, factor, inducing_draws, cell_draws),
        axis=1,
    )
    over_the_region = expected_intensity(region, mean, independent_variance, factor)

    elbo: TensorVariable = pt.sum(window_counts * log_intensity) - pt.sum(over_the_region) - svgp.prior_kl()

    return elbo
