import os
import pathlib

from copy import deepcopy

import pymc as pm
import xarray as xr


def drop_transformed(idata: xr.DataTree, model: pm.Model | None = None) -> xr.DataTree:
    """
    Drop a model's transformed value variables from a posterior in place.

    Constrained parameters are sampled on an unconstrained scale, and PyMC keeps both. The
    transformed copies carry ``__``-suffixed names and their own dimensions, and are noise in
    anything that iterates the posterior.

    Parameters
    ----------
    idata : DataTree
        Inference data whose ``posterior`` group is edited.
    model : Model, optional
        Model the variables belong to. Defaults to the model on the context stack.

    Returns
    -------
    idata : DataTree
        The same object, with the transformed variables and their dimensions removed.

    Examples
    --------
    .. code-block:: python

        import pymc as pm

        from climate_risk.sample import drop_transformed

        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma")
            idata = pm.sample()

        idata = drop_transformed(idata, model)
    """
    model = pm.modelcontext(model)

    value_var_names = [x.name for x in model.value_vars if x.name.endswith("__")]
    value_var_dims = [x for x in idata.posterior.indexes.keys() if "__dim" in x]

    idata.posterior = idata.posterior.drop_vars(value_var_names).drop_dims(value_var_dims)

    return idata


def sample_or_load(
    fp: str,
    *,
    model: pm.Model | None = None,
    force_resample: bool = False,
    sample_kwargs: dict | None = None,
    compile_kwargs: dict | None = None,
    save_results: bool = True,
) -> xr.DataTree:
    """Sample the model or load the model from disk.

    Where ``fp`` already holds inference data and no resample is asked for, it is read back. Otherwise
    the model is sampled, its posterior predictive drawn, and the result written to ``fp``.

    Parameters
    ----------
    fp : str
        Path the inference data is saved to and loaded from.
    model : Model, optional
        Model to sample. Defaults to the model on the context stack.
    force_resample : bool, optional
        Sample again even when data is found at ``fp``. Default False.
    sample_kwargs : dict, optional
        Extra keyword arguments for :func:`pymc.sample`.
    compile_kwargs : dict, optional
        PyTensor compilation arguments, passed to :func:`pymc.sample_posterior_predictive` and
        :func:`pymc.compute_log_likelihood`.
    save_results : bool, optional
        Write the result to ``fp``. Default True.

    Returns
    -------
    idata : DataTree
        Posterior, posterior predictive and log likelihood.

    Examples
    --------
    Sample once, then reuse the saved draws on every later run:

    .. code-block:: python

        import pymc as pm

        from climate_risk.sample import sample_or_load

        with pm.Model() as model:
            mu = pm.Normal("mu")
            obs = pm.Normal("obs", mu=mu, observed=[0.1, 0.3, -0.2])
            idata = sample_or_load("fits/demo.nc")
    """
    _fp = pathlib.Path(fp)

    sample_kwargs = {} if sample_kwargs is None else sample_kwargs
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

    # Create directory structure if necessary
    os.makedirs(_fp.parent, exist_ok=True)

    # Declared up front: PyMC is untyped, so the sampling branch below infers Any without it.
    idata: xr.DataTree

    if _fp.exists() and not force_resample:
        idata = xr.open_datatree(_fp)
        idata.load()  # Force load to avoid mismatch if the memory is overwritten before idata is used
        return idata

    with pm.modelcontext(model):
        idata = pm.sample(**sample_kwargs)
        if sample_kwargs.get("nuts_sampler", "pymc") == "nutpie":
            idata = drop_transformed(idata)

        idata = pm.sample_posterior_predictive(
            idata, extend_inferencedata=True, compile_kwargs=deepcopy(compile_kwargs)
        )
        idata = pm.compute_log_likelihood(idata, extend_inferencedata=True, compile_kwargs=deepcopy(compile_kwargs))

        if save_results:
            if _fp.exists():
                os.remove(_fp)

            idata.to_netcdf(_fp)

    return idata
