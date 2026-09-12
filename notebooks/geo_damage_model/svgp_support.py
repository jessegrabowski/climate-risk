import json

from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import pytensor
import pytensor.tensor as pt

from better_optimize import minimize
from matplotlib.path import Path as MplPath
from ptgp import FitResult
from ptgp.gp import SVGP, init_variational_params
from ptgp.inducing import Points, greedy_variance_init
from ptgp.kernels import Matern32
from ptgp.likelihoods import Poisson
from ptgp.optim.training import compile_scipy_objective, compile_training_step, get_trained_params
from pytensor import shared
from pytensor_ml.optim import adam as adam_rule
from pytensor_ml.optim import apply_if_finite, cosine_schedule, large_step, skip_if
from sklearn.cluster import MiniBatchKMeans
from tqdm.auto import trange

from climate_risk.models.aggregated_poisson import VARIANCE_FLOOR, latent_moments
from climate_risk.models.aggregation import Aggregation
from climate_risk.models.areal import windowed_poisson_elbo

# Longitude, latitude and year. The covariates, if the kernel acts over them at all, follow.
SPATIAL_TEMPORAL_DIMS = 3


def standardized(values: np.ndarray) -> np.ndarray:
    """Centre and scale to unit variance, so a lengthscale reads in standard deviations."""
    return (values - values.mean()) / values.std()


def poisson_intercept(n_events: float, exposure: float) -> float:
    """
    The log rate a homogeneous Poisson would use, as a starting level for the field.

    The intercept and the variational mean are not separately identified: a constant shift of the
    field can sit in either. The intercept is nearly free, costing only a wide normal prior, while
    the same shift held in ``q_mu`` is taxed by the KL term. A quasi-Newton method has the
    curvature to slide along that ridge and park the level in the free parameter; a first-order
    method does not, and leaves it where it started. Starting the intercept where the data puts it
    removes the journey.

    Parameters
    ----------
    n_events : float
        Events observed anywhere in the lattice.
    exposure : float
        Total weight of the lattice, in the units the operator weights by.

    Returns
    -------
    float
    """
    return float(np.log(n_events / exposure))


def place_inducing(coordinates: np.ndarray, weights: np.ndarray, n_inducing: int, *, seed: int = 0) -> np.ndarray:
    """
    Inducing points at population-weighted cluster centres of the lattice.

    Parameters
    ----------
    coordinates : ndarray
        Shape ``(n_points, n_kernel_dims)``, the axes the kernel acts over.
    weights : ndarray
        Exposure of each point, which draws the centres towards where the people are.
    n_inducing : int
        Centres to place. The dense work is ``(n_points, n_inducing)``, so this sets the memory the
        fit needs, and centres closer together than the field resolves drive ``K(Z, Z)`` towards
        singular.
    seed : int, optional
        Seed for the clustering. Default 0.

    Returns
    -------
    ndarray
        Shape ``(n_inducing, n_kernel_dims)``.
    """
    # KMeans minimizes plain Euclidean distance and the axes span wildly different ranges, so
    # clustering raw coordinates partitions the widest one and leaves every centre at one place.
    axis_scale = coordinates.std(axis=0)

    placement = MiniBatchKMeans(n_clusters=n_inducing, random_state=seed, n_init=3, batch_size=8192).fit(
        coordinates / axis_scale, sample_weight=weights + 1.0
    )

    return placement.cluster_centers_ * axis_scale


def select_inducing(features, n_inducing, kernel, *, sample_size=20_000, seed=0):
    """
    Inducing points chosen for what the set as a whole leaves unexplained.

    Greedy conditional-variance selection takes, at each step, the lattice point the current set
    explains least: a partial pivoted Cholesky under the max-diagonal pivot rule, from Burt et al.
    (2020). Where clustering puts centres wherever the exposure is and so repeats itself in the
    cities, this returns points that are non-redundant by construction. ``K(Z, Z)`` inherits that:
    at a thousand points and above, the clustered set is indefinite in single precision and no
    diagonal ridge small enough to leave the model alone will recover it, while this one factors.

    The points are a subset of the lattice rather than free positions, and are meant to be frozen:
    Burt et al. find a good greedy set is within noise of one optimized jointly with the
    hyperparameters, at a fraction of the cost.

    Parameters
    ----------
    features : ndarray
        Shape ``(n_points, n_features)``, the lattice to select from.
    n_inducing : int
        Points to select.
    kernel : Kernel
        The kernel to judge redundancy under, at whatever hyperparameters are believed going in.
    sample_size : int, optional
        Rows to select from. The selection is O(sample_size * n_inducing^2), and a sample of the
        lattice spans the same ground. Default 20000.
    seed : int, optional
        Seed for the sample. Default 0.

    Returns
    -------
    ndarray
        Shape ``(n_inducing, n_features)``.
    """
    floatx = pytensor.config.floatX
    rng = np.random.default_rng(seed)
    sample = features[rng.choice(len(features), size=min(sample_size, len(features)), replace=False)]

    points, _ = greedy_variance_init(sample, n_inducing, kernel)
    selected = points.Z.eval() if hasattr(points.Z, "eval") else points.Z

    return np.ascontiguousarray(selected, dtype=floatx)


def maxent_prior(distribution, name: str, *, lower: float, upper: float, mass: float = 0.9, **kwargs):
    """
    A PyMC prior from the maximum-entropy fit of ``distribution`` to an interval.

    The fitted parameters come back as double, and handing those to PyMC builds a distribution
    whose log-density is double whatever ``floatX`` is set to. That reaches the objective through
    the prior term and makes the whole loss double, which an optimizer then cannot write back into
    single-precision parameters. Casting the parameters keeps the model at one dtype throughout.

    Parameters
    ----------
    distribution : preliz distribution
        An unparameterized instance, such as ``pz.Gamma()``.
    name : str
        Name of the PyMC variable.
    lower, upper : float
        Bounds the fit puts ``mass`` of the distribution between.
    mass : float, optional
        Probability to place inside the interval. Default 0.9.
    **kwargs
        Passed to the PyMC distribution, such as ``shape`` or ``initval``.

    Returns
    -------
    TensorVariable
        The PyMC random variable, in the open model context.
    """
    floatx = pytensor.config.floatX
    fitted = pz.maxent(distribution, lower=lower, upper=upper, mass=mass)
    parameters = {key: np.asarray(value, dtype=floatx) for key, value in fitted.params_dict.items()}

    return getattr(pm, type(fitted).__name__)(name, **parameters, **kwargs)


def build_model(
    features: np.ndarray,
    inducing: np.ndarray,
    *,
    intercept_init: float = -10.0,
    spatial_bounds: tuple[float, float] = (0.5, 8.0),
    temporal_bounds: tuple[float, float] = (2.0, 20.0),
    covariate_bounds: tuple[float, float] = (0.5, 5.0),
    amplitude_bounds: tuple[float, float] = (0.2, 1.0),
) -> tuple[pm.Model, SVGP]:
    """
    The areal GP: a Matern 3/2 field over the axes the inducing points span.

    The width of ``inducing`` chooses how the covariates enter. Three columns leaves them outside
    the kernel, acting on the mean through a linear predictor. As many columns as ``features``
    gives them a second Matern 3/2 of their own, added to the spatial one, so the field is a
    geographic surface plus a surface over the covariates. The two terms are added rather than
    multiplied into one kernel over every axis: a single kernel can switch geography off by
    lengthening one lengthscale, and an additive term still contributes when the other is flat.

    Every prior is a maximum-entropy fit to the interval it is given, so a bound is a statement
    about the scale the process runs on rather than a distribution to be chosen.

    Parameters
    ----------
    features : ndarray
        Shape ``(n_points, n_kernel_dims + n_covariates)``, kernel axes first.
    inducing : ndarray
        Shape ``(n_inducing, n_kernel_dims)``.
    intercept_init : float, optional
        Where the field's level starts. See :func:`poisson_intercept`. Default -10.0, the prior's
        own median.
    spatial_bounds : tuple of float, optional
        Degrees the spatial lengthscale is expected between. The inducing points have to resolve
        whatever this allows, so a short lengthscale needs many of them. Default (0.5, 8.0).
    temporal_bounds : tuple of float, optional
        Years the temporal lengthscale is expected between. Default (2.0, 20.0).
    covariate_bounds : tuple of float, optional
        Standard deviations the covariate lengthscales are expected between. Under one is finer
        than the spread of the data resolves, over a few is a straight line. Default (0.5, 5.0).
    amplitude_bounds : tuple of float, optional
        The field is a log-scale multiplier on the rate, so the default is a residual swing
        between about 1.2x and 2.7x either way. Default (0.2, 1.0).

    Returns
    -------
    model : Model
        The PyMC model holding the hyperparameter priors.
    svgp : SVGP
        The sparse variational field defined over it.
    """
    floatx = pytensor.config.floatX
    n_kernel_dims = inducing.shape[1]
    n_covariates = features.shape[1] - n_kernel_dims
    n_kernel_covariates = n_kernel_dims - SPATIAL_TEMPORAL_DIMS

    # active_dims slices the inducing points too, so any padding here is never read.
    inducing_features = np.column_stack([inducing, np.zeros((len(inducing), n_covariates))]).astype(floatx)

    with pm.Model() as model:
        intercept = pm.Normal(
            "intercept", mu=np.asarray(-10.0, floatx), sigma=np.asarray(5.0, floatx), initval=intercept_init
        )
        amplitude = maxent_prior(pz.Gamma(), "amplitude", lower=amplitude_bounds[0], upper=amplitude_bounds[1])

        # Lengthscales are lognormal: the likelihood is nearly flat in them, so the shape of the
        # tail is what decides where they settle, and a tail quadratic in the log holds where one
        # linear in it does not.
        spatial = maxent_prior(pz.LogNormal(), "spatial_lengthscale", lower=spatial_bounds[0], upper=spatial_bounds[1])
        temporal = maxent_prior(
            pz.LogNormal(), "temporal_lengthscale", lower=temporal_bounds[0], upper=temporal_bounds[1]
        )

        kernel = amplitude**2 * Matern32(
            input_dim=features.shape[1],
            ls=pt.stack([spatial, spatial, temporal]),
            active_dims=list(range(SPATIAL_TEMPORAL_DIMS)),
        )

        if n_kernel_covariates:
            covariate_amplitude = maxent_prior(
                pz.Gamma(), "covariate_amplitude", lower=amplitude_bounds[0], upper=amplitude_bounds[1]
            )
            covariate_lengthscale = maxent_prior(
                pz.LogNormal(),
                "covariate_lengthscale",
                lower=covariate_bounds[0],
                upper=covariate_bounds[1],
                shape=(n_kernel_covariates,),
            )
            kernel = kernel + covariate_amplitude**2 * Matern32(
                input_dim=features.shape[1],
                ls=covariate_lengthscale,
                active_dims=list(range(SPATIAL_TEMPORAL_DIMS, n_kernel_dims)),
            )

        if n_covariates:
            coefficients = pm.Normal(
                "coefficients",
                mu=np.asarray(0.0, floatx),
                sigma=np.asarray(1.0, floatx),
                shape=(n_covariates,),
            )
            mean = lambda batch: intercept + pt.dot(batch[:, n_kernel_dims:], coefficients)  # noqa: E731
        else:
            mean = lambda batch: pt.full((batch.shape[0],), intercept)  # noqa: E731

        svgp = SVGP(
            kernel=kernel,
            likelihood=Poisson(),
            mean=mean,
            inducing_variable=Points(Z=inducing_features),
            variational_params=init_variational_params(len(inducing)),
            whiten=True,
        )

    return model, svgp


# Column layout build_model_2 reads. Time is shared between the two Matern blocks rather than
# carried by one, so each surface bends in its own way over the record. Country is absent: it is a
# set of offsets, so it rides in the linear predictor as indicator columns rather than as a
# categorical kernel, whose gradient is a scatter-add that the MLX backend dispatches wrongly.
YEAR, TO_RIVER, TO_COAST, RIVER_FLOW = 0, 1, 2, 3

# Shape of the inverse gamma the lengthscales carry. Higher concentrates it harder on the mode.
LENGTHSCALE_CONCENTRATION = 3.0


def build_model_2(
    features: np.ndarray,
    inducing: np.ndarray,
    *,
    intercept_init: float = -10.0,
    temporal_bounds: tuple[float, float] = (2.0, 20.0),
    covariate_bounds: tuple[float, float] = (0.5, 5.0),
    amplitude_bounds: tuple[float, float] = (0.2, 1.0),
    sqrt_method: str = "cholesky",
) -> tuple[pm.Model, SVGP]:
    """
    Hazard as a response to terrain, with no geographic surface under it.

    Two Matern 3/2 surfaces, one over distance to river against the river's own discharge, one over
    distance to coast, each crossed with time. Geography enters only as a per-country offset in
    the linear predictor, which is the resolution EM-DAT's reporting varies at, and which cannot stand in for a smooth covariate
    gradient the way a surface over latitude and longitude can.

    Population leaves the exposure weight and enters the linear predictor, so the rate reads as
    ``area * people ** beta * exp(f)``. The exponent is fitted rather than assumed to be one, and
    held inside the unit interval, which is the range over which it has a meaning.

    Parameters
    ----------
    features : ndarray
        Shape ``(n_points, 4 + n_covariates)``, ordered by the module's column constants. The
        trailing columns are the linear predictor and their order is load-bearing: the country
        indicators, then centered log population as the last column.
    inducing : ndarray
        Shape ``(n_inducing, 4)``.
    intercept_init : float, optional
        Where the field's level starts. See :func:`poisson_intercept`. Default -10.0.
    temporal_bounds : tuple of float, optional
        Years each block's temporal lengthscale is expected between. Default (2.0, 20.0).
    covariate_bounds : tuple of float, optional
        Standard deviations the covariate lengthscales are expected between. Default (0.5, 5.0).
    amplitude_bounds : tuple of float, optional
        Log-scale swing each surface is expected to carry. Default (0.2, 1.0).
    sqrt_method : str, optional
        How ``K(Z, Z)`` is factored, 'cholesky' or 'eigh'. The two define different whitened
        parameterizations, so variational parameters do not carry across a change of it. Default
        'cholesky'.

    Returns
    -------
    model : Model
        The PyMC model holding the hyperparameter priors.
    svgp : SVGP
        The sparse variational field defined over it.
    """
    floatx = pytensor.config.floatX
    n_kernel_dims = inducing.shape[1]
    n_covariates = features.shape[1] - n_kernel_dims

    inducing_features = np.column_stack([inducing, np.zeros((len(inducing), n_covariates))]).astype(floatx)

    def lengthscale(name, lower, upper):
        """An inverse gamma over the interval, placing its mode at the interval's geometric centre.

        Inverse gamma, not a maximum-entropy fit to the interval: the fit puts most of its mass
        inside and leaves tails the optimizer can walk out of, which is how a lengthscale ends up
        below the floor and another above the ceiling in the same fit. This has zero density at both
        zero and infinity, so neither runaway direction is reachable rather than merely unlikely.
        """
        mode = float(np.sqrt(lower * upper))

        return pm.InverseGamma(
            name,
            alpha=np.asarray(LENGTHSCALE_CONCENTRATION, floatx),
            beta=np.asarray(mode * (LENGTHSCALE_CONCENTRATION + 1.0), floatx),
        )

    def matern_block(name, dims, bounds):
        """One additive surface, with a lengthscale per axis on that axis's own scale."""
        # Exponential, with its mean at the geometric centre of the interval. The maximum-entropy
        # gamma it replaces left a tail heavy enough for one surface to inflate to twice the top of
        # its own interval while the other sat on the floor. This decays exponentially above the
        # mean, so inflation costs; it does nothing against a surface switching itself off, which
        # its mode at zero makes cheaper rather than dearer.
        amplitude_mean = float(np.sqrt(amplitude_bounds[0] * amplitude_bounds[1]))
        amplitude = pm.Exponential(f"{name}_amplitude", lam=np.asarray(1.0 / amplitude_mean, floatx))

        lengthscales = pt.stack(
            [lengthscale(f"{name}_lengthscale_{axis}", lower, upper) for axis, (lower, upper) in bounds.items()]
        )

        return amplitude**2 * Matern32(input_dim=features.shape[1], ls=lengthscales, active_dims=list(dims))

    with pm.Model() as model:
        intercept = pm.Normal(
            "intercept", mu=np.asarray(-10.0, floatx), sigma=np.asarray(5.0, floatx), initval=intercept_init
        )

        kernel = matern_block(
            "hydro",
            (TO_RIVER, RIVER_FLOW, YEAR),
            {"to_river": covariate_bounds, "river_flow": covariate_bounds, "year": temporal_bounds},
        ) + matern_block("coast", (TO_COAST, YEAR), {"to_coast": covariate_bounds, "year": temporal_bounds})

        # The exponent is bounded to the unit interval, which is the whole of what it can mean: zero
        # is hazard over ground alone and one is strict proportionality to people. It is not a free
        # coefficient. Left unbounded it wanders negative -- fewer people, more recorded disasters --
        # and under an area weight that is the one direction that detonates, since an atom with
        # almost nobody on it sits eleven logs below the mean and a negative exponent turns that
        # into an exponential. It is also what caps the learning rate: the divergence at every rate
        # above 5e-3 ran through this sign.
        population_exponent = maxent_prior(pz.Beta(), "population_exponent", lower=0.1, upper=0.95)

        # Trailing columns are the country indicators, then log population last.
        country_offsets = pm.Normal(
            "country_offsets", mu=np.asarray(0.0, floatx), sigma=np.asarray(1.0, floatx), shape=(n_covariates - 1,)
        )

        def mean(batch):
            country = pt.dot(batch[:, n_kernel_dims:-1], country_offsets)

            return intercept + country + population_exponent * batch[:, -1]

        inducing_variable = Points(Z=inducing_features)
        inducing_variable.sqrt_method = sqrt_method

        svgp = SVGP(
            kernel=kernel,
            likelihood=Poisson(),
            mean=mean,
            inducing_variable=inducing_variable,
            variational_params=init_variational_params(len(inducing)),
            whiten=True,
        )

    return model, svgp


def flatten_params(fit: FitResult) -> np.ndarray:
    """
    A fit's parameters as the flat vector the scipy objective is written in.

    The layout is the one ``compile_scipy_objective`` builds: the model's value variables in the
    order it created them, then the variational extras. A fit over the same lattice and the same
    inducing points yields the same layout, which is what makes one usable as another's start.

    Parameters
    ----------
    fit : FitResult
        Any fit of this model, from either optimizer.

    Returns
    -------
    ndarray
        Shape ``(n_parameters,)``.
    """
    ordered = [*fit.shared_params.values(), *fit.shared_extras]

    return np.concatenate([np.asarray(variable.get_value(), dtype=np.float64).ravel() for variable in ordered])


def cap_mlx_memory(limit_gb: int = 20, cache_gb: int = 6) -> None:
    """
    Cap MLX's allocator, so an oversized fit raises rather than taking the machine down.

    Parameters
    ----------
    limit_gb : int, optional
        Hard allocation ceiling. Default 20.
    cache_gb : int, optional
        How much freed memory MLX keeps for reuse. Default 6.
    """
    import mlx.core as mx

    mx.set_cache_limit(cache_gb * 1024**3)
    mx.set_memory_limit(limit_gb * 1024**3)


def annealed_adam(
    n_steps: int, *, learning_rate: float = 3e-3, n_parameters: int | None = None, skip_multiple: float = 5.0
):
    """
    Adam under a cosine schedule, with a divergent step thrown away rather than shrunk.

    Annealing to zero is what removes the noise floor: at a fixed step size the iterates orbit the
    optimum at a radius set by the step times the gradient noise, and shrinking the noise only
    shrinks the orbit.

    The guard discards a step rather than rescaling it, which leaves an ordinary step untouched at
    any scale. Rescaling cannot be: Adam moves about the learning rate per coordinate, so a step's
    norm runs near ``learning_rate * sqrt(n_parameters)``, and a bound below that shrinks every step
    and flattens the schedule into a constant one. A bound set too low here rejects every step and
    raises within a few instead. Rescaling also cannot survive an infinity, which returns
    ``inf * (max_norm / inf)`` and writes a NaN into the parameters.

    Parameters
    ----------
    n_steps : int
        Steps the schedule anneals over.
    learning_rate : float, optional
        Step size at the first step, decaying to zero at the last. Default 3e-3.
    n_parameters : int, optional
        Free parameters the rule updates, which sets the scale a step is judged against. Only
        non-finite steps are rejected when it is unknown. Default None.
    skip_multiple : float, optional
        Multiples of the expected step norm a step may reach before it is thrown away. Default 5.0.

    Returns
    -------
    UpdateRule
        Takes ``(loss, parameters)`` and returns the updates, which is what ptgp's training step
        asks of an ``optimizer_fn``.
    """
    rule = adam_rule(learning_rate=cosine_schedule(learning_rate, total_steps=n_steps))

    if n_parameters is None:
        return apply_if_finite(rule)

    return skip_if(rule, large_step(skip_multiple * learning_rate * np.sqrt(n_parameters)))


def fit_field(
    features,
    counts,
    observations,
    region,
    inducing,
    *,
    start=None,
    mode="MLX",
    maxiter=2000,
    tol=1e-16,
    max_passes=3,
    pass_tol=1.0,
    n_draws=16,
    seed=0,
    model_fn=None,
    capture=None,
    **model_kwargs,
):
    """
    Fit the areal GP over the whole lattice at once.

    Parameters
    ----------
    features : ndarray
        Shape ``(n_points, n_kernel_dims + n_covariates)``.
    counts : ndarray
        Events observed through each row of ``observations``.
    observations : Aggregation
        The event-window operator over the lattice.
    region : Aggregation
        The whole lattice in one row, which the objective subtracts once.
    inducing : ndarray
        Shape ``(n_inducing, n_kernel_dims)``. Its width chooses whether the covariates enter
        through a kernel of their own or through the mean; see :func:`build_model`.
    start : FitResult, optional
        A fit over the same lattice and inducing points to begin from. Default None, which starts
        from the prior.
    mode : str, optional
        PyTensor compile mode. 'MLX' runs on Apple silicon, 'FAST_RUN' on the CPU. A singular
        ``K(Z, Z)`` comes back as NaN under numba, which reaches the loss and gets the step
        rejected; MLX aborts the interpreter. Default 'MLX'.
    maxiter : int, optional
        L-BFGS-B iterations within one pass. Default 2000.
    tol : float, optional
        L-BFGS-B convergence tolerance. Default 1e-16.
    max_passes : int, optional
        Times to run L-BFGS-B. Default 3.
    pass_tol : float, optional
        A pass gaining less than this ends the fit. Default 1.0.
    n_draws : int, optional
        Draws behind the log-intensity term. Default 16.
    seed : int, optional
        Seed for those draws. Default 0.
    **model_kwargs
        Passed to :func:`build_model`, such as the prior bounds.

    Returns
    -------
    svgp : SVGP
        The fitted field.
    result : FitResult
        Trained parameters and the optimizer's own result, ready for :func:`ptgp.predict`.
    """
    if mode == "MLX":
        cap_mlx_memory()

    model, svgp = (model_fn or build_model)(
        features,
        inducing,
        intercept_init=poisson_intercept(float(counts.sum()), float(region.weights.sum())),
        **model_kwargs,
    )

    with model:

        def objective(gp, batch_features, window_counts):
            return windowed_poisson_elbo(
                gp, batch_features, window_counts, observations, region, n_draws=n_draws, seed=seed
            )

        loss_and_grad, from_prior, unpack, shared_params, shared_extras = compile_scipy_objective(
            objective,
            svgp,
            pt.matrix("X", shape=(None, features.shape[1])),
            pt.vector("y", shape=(None,)),
            model=model,
            compile_kwargs={"mode": mode},
        )

        position = from_prior if start is None else flatten_params(start)
        if position.size != from_prior.size:
            raise ValueError(
                f"The start carries {position.size} parameters and this fit takes {from_prior.size}. They are "
                f"the same model only over the same lattice and the same inducing points."
            )

        # Scipy needs numpy; the MLX linker hands back its own array type.
        def loss_and_grad_numpy(*args):
            loss, gradient = loss_and_grad(*args)
            return np.asarray(loss, dtype=np.float64), np.asarray(gradient, dtype=np.float64)

        def descend(theta):
            return minimize(
                loss_and_grad_numpy,
                theta,
                args=(features, counts),
                method="L-BFGS-B",
                tol=tol,
                maxiter=maxiter,
            )

        # L-BFGS-B stops on its relative-reduction test with the gradient still far from zero, and
        # running it again from where it stopped resets the curvature memory. What a pass gains
        # falls away quickly, so this bounds the passes and ends on one that gains nothing. The
        # iteration count says nothing about whether a pass helped, and against a noisy objective
        # it is never zero.
        settled = descend(position)
        for _ in range(max_passes - 1):
            candidate = descend(settled.x)
            gained = settled.fun - candidate.fun
            if candidate.fun < settled.fun:
                settled = candidate
            if gained < pass_tol:
                break

        unpack(settled.x)

        trained = get_trained_params(model, shared_params)

    return svgp, FitResult(
        result=settled,
        params=trained,
        shared_params=shared_params,
        shared_extras=tuple(shared_extras),
        model=model,
    )


MinibatchTerms = namedtuple(
    "MinibatchTerms",
    "elbo window_term region_term kl smallest_window_total windows_at_zero smallest_log_intensity "
    "largest_absolute_mean smallest_independent_variance largest_independent_variance largest_absolute_factor",
)


def build_minibatch_objective(observations, features, weights, n_inducing, *, batch_size, n_draws=16):
    """
    A batch sampler and the objective over whatever batch it last wrote.

    Everything the batch stands for other than its own features -- its operator, its scale factor
    and its noise -- travels in shared variables, because the compiled step takes only the features
    and the counts as arguments.

    Parameters
    ----------
    observations : Aggregation
        The event-window operator over the lattice.
    features : ndarray
        Shape ``(n_points, n_kernel_dims + n_covariates)``, the whole lattice.
    weights : ndarray
        Exposure of each lattice point.
    n_inducing : int
        Inducing points the field is defined over.
    batch_size : int
        Lattice points per batch. Fixed, so the backend traces one graph. It has to clear what the
        sampled windows bring on their own, or the complement falls back on its floor and the
        region term is left standing on a handful of cells at a large scale.
    n_draws : int, optional
        Draws behind the log-intensity term. Default 16.

    Returns
    -------
    load_batch : callable
        Takes ``(rng, n_windows)``, writes the shared variables, returns the step's arguments.
    minibatch_elbo : callable
        The objective, in the signature ptgp's training step expects.
    """
    floatx = pytensor.config.floatX
    n_quadrature = observations.n_cells
    window_operator = observations.matrix.tocsr()

    batch_rows = shared(np.zeros(1, dtype="int64"), name="batch_rows")
    batch_columns = shared(np.zeros(1, dtype="int64"), name="batch_columns")
    batch_window_weights = shared(np.zeros(1, dtype=floatx), name="batch_window_weights")
    batch_region_weights = shared(np.zeros(batch_size, dtype=floatx), name="batch_region_weights")
    window_scale = shared(np.array(1.0, dtype=floatx), name="window_scale")
    inducing_draws = shared(np.zeros((n_inducing, n_draws), dtype=floatx), name="inducing_draws")
    atom_draws = shared(np.zeros((batch_size, n_draws), dtype=floatx), name="atom_draws")

    def load_batch(rng, n_windows, *, min_sampled=2_000):
        rows = rng.choice(observations.n_units, size=n_windows, replace=False)
        overlaps = window_operator[rows].tocoo()
        window_columns = np.unique(overlaps.col)

        # The batch grows past its budget only when the sampled windows alone exceed it, which
        # takes a draw of country-tier events. Growing rather than truncating keeps it unbiased.
        n_sampled = max(batch_size - window_columns.size, min_sampled)

        # Drawing this many guarantees `n_sampled` survive even if every window atom comes up.
        drawn = rng.choice(n_quadrature, size=n_sampled + window_columns.size, replace=False)
        sampled = drawn[~np.isin(drawn, window_columns, assume_unique=True)][:n_sampled]

        # Window atoms first and sorted, so an overlap's column is its position by binary search.
        columns = np.concatenate([window_columns, sampled])
        region_weights = weights[columns].astype(floatx)
        region_weights[window_columns.size :] *= (n_quadrature - window_columns.size) / n_sampled

        batch_rows.set_value(overlaps.row.astype("int64"))
        batch_columns.set_value(np.searchsorted(window_columns, overlaps.col).astype("int64"))
        batch_window_weights.set_value(overlaps.data.astype(floatx))
        batch_region_weights.set_value(region_weights)
        window_scale.set_value(np.array(observations.n_units / n_windows, dtype=floatx))

        inducing_draws.set_value(rng.standard_normal((n_inducing, n_draws), dtype=floatx))
        atom_draws.set_value(rng.standard_normal((columns.size, n_draws), dtype=floatx))

        return features[columns], np.ones(n_windows, dtype=floatx)

    def minibatch_elbo(gp, batch_features, batch_counts):
        mean, independent_variance, factor = latent_moments(gp, batch_features)

        spread = pt.sqrt(pt.clip(independent_variance, VARIANCE_FLOOR, np.inf))
        field = mean[:, None] + factor @ inducing_draws + spread[:, None] * atom_draws

        # Shifted before exponentiating, so a window whose atoms all underflow early in training
        # does not take the log of zero.
        shift = pt.max(field, axis=0)
        contributions = batch_window_weights[:, None] * pt.exp(field - shift)[batch_columns]
        totals = pt.inc_subtensor(pt.zeros((batch_counts.shape[0], n_draws))[batch_rows], contributions)
        log_intensity = pt.mean(pt.log(totals) + shift, axis=1)

        # Every atom of the batch enters the region term; the weights carry the scale that makes
        # the sampled ones stand for the ground no window covers.
        log_scaled = mean + (independent_variance + pt.sum(factor**2, axis=1)) / 2.0
        region_shift = pt.max(log_scaled)
        over_region = pt.dot(batch_region_weights, pt.exp(log_scaled - region_shift)) * pt.exp(region_shift)

        return window_scale * pt.sum(batch_counts * log_intensity) - over_region - gp.prior_kl()

    def minibatch_terms(gp, batch_features, batch_counts):
        """The same objective, opened up, so a non-finite loss can be traced to the term carrying it."""
        mean, independent_variance, factor = latent_moments(gp, batch_features)

        field = (
            mean[:, None]
            + factor @ inducing_draws
            + pt.sqrt(pt.clip(independent_variance, VARIANCE_FLOOR, np.inf))[:, None] * atom_draws
        )
        shift = pt.max(field, axis=0)
        contributions = batch_window_weights[:, None] * pt.exp(field - shift)[batch_columns]
        totals = pt.inc_subtensor(pt.zeros((batch_counts.shape[0], n_draws))[batch_rows], contributions)
        log_intensity = pt.mean(pt.log(totals) + shift, axis=1)

        log_scaled = mean + (independent_variance + pt.sum(factor**2, axis=1)) / 2.0
        region_shift = pt.max(log_scaled)
        over_region = pt.dot(batch_region_weights, pt.exp(log_scaled - region_shift)) * pt.exp(region_shift)

        window_term = window_scale * pt.sum(batch_counts * log_intensity)
        kl = gp.prior_kl()

        return MinibatchTerms(
            elbo=window_term - over_region - kl,
            window_term=window_term,
            region_term=over_region,
            kl=kl,
            smallest_window_total=pt.min(totals),
            windows_at_zero=pt.sum(totals <= 0.0).astype(mean.dtype),
            smallest_log_intensity=pt.min(log_intensity),
            largest_absolute_mean=pt.max(pt.abs(mean)),
            smallest_independent_variance=pt.min(independent_variance),
            largest_independent_variance=pt.max(independent_variance),
            largest_absolute_factor=pt.max(pt.abs(factor)),
        )

    # The batch itself lives in these, so a failing step can be reproduced only by carrying their
    # values across to another copy of the objective.
    batch_state = {
        "batch_rows": batch_rows,
        "batch_columns": batch_columns,
        "batch_window_weights": batch_window_weights,
        "batch_region_weights": batch_region_weights,
        "window_scale": window_scale,
        "inducing_draws": inducing_draws,
        "atom_draws": atom_draws,
    }

    return load_batch, minibatch_elbo, minibatch_terms, batch_state


def fit_minibatch(
    features,
    observations,
    weights,
    inducing,
    *,
    mode="MLX",
    n_steps=1000,
    n_windows=16,
    batch_size=24_000,
    optimizer_fn=None,
    n_draws=16,
    seed=0,
    model_fn=None,
    capture=None,
    **model_kwargs,
):
    """
    Fit the areal GP with Adam on sampled batches.

    Parameters
    ----------
    features : ndarray
        Shape ``(n_points, n_kernel_dims + n_covariates)``.
    observations : Aggregation
        The event-window operator over the lattice.
    weights : ndarray
        Exposure of each lattice point.
    inducing : ndarray
        Shape ``(n_inducing, n_kernel_dims)``.
    mode : str, optional
        PyTensor compile mode. 'MLX' runs on Apple silicon, 'FAST_RUN' on the CPU. Default 'MLX'.
    n_steps : int, optional
        Optimizer steps. Default 1000.
    n_windows : int, optional
        Event windows per batch. The window term is scaled by the events over the events sampled,
        so few windows put a large multiplier on a noisy sum; windows are small, so many of them
        fit inside the same budget. Default 16.
    batch_size : int, optional
        Lattice points per batch. Default 24000.
    optimizer_fn : callable, optional
        Takes ``(loss, parameters)`` and returns the updates. Default None, which is
        :func:`annealed_adam` over ``n_steps``.
    n_draws : int, optional
        Draws behind the log-intensity term. Default 16.
    seed : int, optional
        Seed for the batch sampler and the step noise. Default 0.
    **model_kwargs
        Passed to :func:`build_model`, such as the prior bounds.

    Returns
    -------
    svgp : SVGP
        The fitted field.
    result : FitResult
        Trained parameters, carrying the per-step loss on ``result.result``.
    """
    if mode == "MLX":
        cap_mlx_memory()

    model, svgp = (model_fn or build_model)(
        features,
        inducing,
        intercept_init=poisson_intercept(observations.n_units, float(weights.astype(np.float64).sum())),
        **model_kwargs,
    )
    load_batch, minibatch_elbo, _, batch_state = build_minibatch_objective(
        observations, features, weights, len(inducing), batch_size=batch_size, n_draws=n_draws
    )
    rng = np.random.default_rng(seed)

    with model:
        step, shared_params, shared_extras = compile_training_step(
            minibatch_elbo,
            svgp,
            pt.matrix("X", shape=(None, features.shape[1])),
            pt.vector("y", shape=(n_windows,)),
            model=model,
            optimizer_fn=annealed_adam(n_steps) if optimizer_fn is None else optimizer_fn,
            compile_kwargs={"mode": mode},
        )

        ordered = [*shared_params.values(), *shared_extras]
        recording = capture is not None
        radius = capture.get("radius", 5) if recording else 0
        history = deque(maxlen=radius + 1)
        window = None

        losses = np.empty(n_steps)
        progress = trange(n_steps, desc="adam")
        for position in progress:
            if recording:
                # The parameters a step's loss is reported at are the ones before its own update.
                before = np.concatenate([np.asarray(v.get_value(), dtype=np.float64).ravel() for v in ordered])

            batch = load_batch(rng, n_windows)
            losses[position] = step(*batch)

            if recording:
                history.append((position, before, float(losses[position])))

                if window is None and not np.isfinite(losses[position]):
                    # Read before the next load_batch overwrites them, so these are the failing batch.
                    capture.update(
                        step=position,
                        theta=before,
                        batch=batch,
                        loss=float(losses[position]),
                        batch_state={name: variable.get_value() for name, variable in batch_state.items()},
                    )
                    window = list(history)
                elif window is not None:
                    window.append((position, before, float(losses[position])))

                if window is not None and len(window) >= 2 * radius + 1:
                    capture["window"] = window
                    recording = False

            if position % 25 == 0:
                # A window mean rather than the step's own loss. Every step draws a fresh batch, so
                # one value carries the whole batch-to-batch spread and reads as movement.
                recent = losses[max(0, position - 99) : position + 1]
                progress.set_postfix(loss=f"{recent.mean():.1f}")

        if capture is not None and "window" not in capture and window is not None:
            capture["window"] = window

        trained = get_trained_params(model, shared_params)

    return svgp, FitResult(
        result=losses,
        params=trained,
        shared_params=shared_params,
        shared_extras=tuple(shared_extras),
        model=model,
    )


def save_fit(fit: FitResult, name: str, directory: Path, **metadata) -> Path:
    """
    Write a fit somewhere a kernel restart cannot reach.

    ``theta`` is the flat parameter vector, so a saved fit can start another one; the constrained
    parameters and whatever metadata the caller passes ride alongside for reading.

    Parameters
    ----------
    fit : FitResult
        The fit to store.
    name : str
        File stem.
    directory : Path
        Where to put it. Created if it does not exist.
    **metadata
        Scalars recorded beside the parameters, such as the objective and the configuration that
        produced them.

    Returns
    -------
    Path
        Where it landed.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{name}.npz"

    arrays = {
        "theta": flatten_params(fit),
        "metadata": json.dumps(metadata, default=float),
        **{f"param_{key}": np.asarray(value) for key, value in fit.params.items()},
    }

    # Adam returns its loss trace; the quasi-Newton path returns a scipy result, which carries no
    # per-step history to keep.
    if isinstance(fit.result, np.ndarray):
        arrays["trace"] = fit.result

    np.savez(destination, **arrays)

    return destination


def load_fit(name: str, directory: Path) -> tuple[np.ndarray, dict, dict]:
    """
    Read back what :func:`save_fit` wrote.

    Parameters
    ----------
    name : str
        File stem.
    directory : Path
        Where :func:`save_fit` put it.

    Returns
    -------
    theta : ndarray
        The flat parameter vector, in the layout :func:`flatten_params` produces.
    params : dict
        Constrained parameters, keyed as the model names them.
    metadata : dict
        Whatever was recorded alongside.
    """
    stored = np.load(directory / f"{name}.npz", allow_pickle=False)
    params = {key.removeprefix("param_"): stored[key] for key in stored.files if key.startswith("param_")}

    return stored["theta"], params, json.loads(str(stored["metadata"]))


@dataclass(frozen=True)
class Lattice:
    """
    The quadrature lattice a fit is defined over, and what a map needs to draw it.

    A fitted field is a flat vector over atoms varying fastest within years, which means nothing
    without the atoms, the years and the operator that produced it. Carrying them together keeps a
    plotting call from needing eight arguments that must agree with each other.

    Parameters
    ----------
    atoms : DataFrame
        One row per quadrature atom, carrying ``lon``, ``lat`` and the covariates.
    years : ndarray
        The years the lattice spans, ascending.
    observations : Aggregation
        The event-window operator, whose rows are event ids.
    units : GeoDataFrame
        The administrative units the atoms come from, keyed by ``gid``.
    boundary : GeoDataFrame
        Country polygons, carrying ``ISO_A3``.
    iso_of_event : dict
        Country of each event, keyed by event id.
    year_of_event : dict
        Year of each event, keyed by event id.
    """

    atoms: pd.DataFrame
    years: np.ndarray
    observations: Aggregation
    units: gpd.GeoDataFrame
    boundary: gpd.GeoDataFrame
    iso_of_event: dict
    year_of_event: dict

    @property
    def n_atoms(self) -> int:
        """Atoms the field is defined at."""
        return len(self.atoms)

    @property
    def n_years(self) -> int:
        """Years the lattice spans."""
        return len(self.years)

    @property
    def year_index(self) -> dict:
        """Position of each year in the lattice."""
        return {year: position for position, year in enumerate(self.years)}

    @property
    def iso_of_atom(self) -> np.ndarray:
        """Country of each atom, from the leading segment of its id."""
        return np.array([gid.split(".")[0] for gid in self.atoms["gid"]])

    @property
    def geometry_of_atom(self) -> gpd.GeoSeries:
        """The polygon each atom stands for, in atom order."""
        return self.units.set_index("gid").geometry.reindex(self.atoms["gid"]).reset_index(drop=True)


def boundary_path(geometry) -> MplPath:
    """A matplotlib path over a (multi)polygon's rings, for clipping a filled contour to the place."""
    vertices, codes = [], []

    for polygon in getattr(geometry, "geoms", [geometry]):
        for ring in [polygon.exterior, *polygon.interiors]:
            ring_coordinates = np.asarray(ring.coords)
            vertices.append(ring_coordinates)
            codes.append([MplPath.MOVETO, *[MplPath.LINETO] * (len(ring_coordinates) - 2), MplPath.CLOSEPOLY])

    return MplPath(np.concatenate(vertices), np.concatenate(codes))


def plot_country(
    lattice,
    iso,
    fitted_intensity,
    inducing=None,
    trained=None,
    *,
    panel_years=None,
    figsize=(15, 7.5),
    exposure="person",
):
    """
    One country's fitted field at three years, with its units, its events and the inducing points.

    The region is too wide to read at once, and a country is the scale a reader checks a result
    against. The surface is contoured over that country's atoms and clipped to its border; every
    event recorded there in that year is outlined at the union of the units it was reported
    through, which is the ground the likelihood actually integrated over.

    Parameters
    ----------
    lattice : Lattice
        The lattice the field is defined over.
    iso : str
        Three-letter country code, as the atom ids carry it.
    fitted_intensity : ndarray
        Shape ``(n_atoms * n_years,)``, the rate per unit of exposure.
    inducing : ndarray, optional
        Shape ``(n_inducing, n_kernel_dims)``, longitude, latitude and year. Drawn only when the
        kernel is over geography, since points in a covariate space have nowhere to sit on a map.
        Default None, which draws none.
    trained : dict, optional
        Fitted parameters, read for the temporal lengthscale that sizes the inducing markers.
        Required only alongside ``inducing``. Default None.
    panel_years : sequence of int, optional
        Years to draw. Default None, which takes the first, middle and last of the record.
    figsize : tuple, optional
        Figure size. Default (15, 7.5).
    exposure : str, optional
        The unit the rate is quoted against, which is whatever the operator weights by. Default
        'person'.
    """
    iso_of_atom = lattice.iso_of_atom
    here = iso_of_atom == iso
    if not here.any():
        raise ValueError(f"No atoms carry the country code {iso!r}; the place holds {sorted(set(iso_of_atom))}.")

    years, year_index = lattice.years, lattice.year_index
    panel_years = (
        list(panel_years)
        if panel_years is not None
        else [int(years[0]), int(years[lattice.n_years // 2]), int(years[-1])]
    )

    # Contoured on the log scale, which is where the field is smooth and where the prior lives.
    field = np.log(fitted_intensity).reshape(lattice.n_years, lattice.n_atoms)[:, here]
    shown = field[[year_index[year] for year in panel_years]]
    levels = np.linspace(shown.min(), shown.max(), 21)

    country = lattice.boundary.loc[lattice.boundary["ISO_A3"] == iso].geometry.union_all()
    country_units = lattice.units.loc[lattice.units["gid"].str.startswith(f"{iso}.")]
    clip = boundary_path(country)
    west, south, east, north = country.bounds

    country_lon = lattice.atoms["lon"].to_numpy()[here]
    country_lat = lattice.atoms["lat"].to_numpy()[here]
    window_operator = lattice.observations.matrix.tocsr()
    geometry_of_atom = lattice.geometry_of_atom

    figure, axes = plt.subplots(1, len(panel_years), figsize=figsize, sharex=True, sharey=True)

    for ax, year in zip(np.atleast_1d(axes), panel_years, strict=True):
        surface = ax.tricontourf(
            country_lon, country_lat, field[year_index[year]], levels=levels, cmap="magma", extend="both"
        )
        surface.set_clip_path(clip, transform=ax.transData)
        country_units.boundary.plot(ax=ax, color="0.35", linewidth=0.4)

        observed = [
            row
            for row, disno in enumerate(lattice.observations.units)
            if lattice.iso_of_event[disno] == iso and lattice.year_of_event[disno] == year
        ]
        for row in observed:
            reached = np.unique(window_operator[[row]].tocoo().col % lattice.n_atoms)
            gpd.GeoSeries([geometry_of_atom.iloc[reached].union_all()], crs=lattice.units.crs).boundary.plot(
                ax=ax, color="#39ff88", linewidth=1.4
            )

        if inducing is not None:
            # An inducing point sits at a year as well as a place, so it informs this slice by how
            # near it is, at the temporal lengthscale the fit settled on.
            informs = np.exp(-0.5 * ((inducing[:, 2] - year) / trained["temporal_lengthscale"]) ** 2)
            inside = (
                (inducing[:, 0] > west) & (inducing[:, 0] < east) & (inducing[:, 1] > south) & (inducing[:, 1] < north)
            )
            ax.scatter(
                inducing[inside, 0],
                inducing[inside, 1],
                s=6 + 90 * informs[inside],
                marker="D",
                facecolor="none",
                edgecolor="white",
                linewidth=0.9,
            )

        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{year}   ({len(observed)} events)", loc="left")

    figure.suptitle(f"{iso}: log rate per {exposure}-year", x=0.125, ha="left")
    figure.colorbar(surface, ax=axes, label=f"log rate per {exposure}-year", shrink=0.55, pad=0.02)
    plt.show()
