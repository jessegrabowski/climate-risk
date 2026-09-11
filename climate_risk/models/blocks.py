from collections.abc import Sequence

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pytensor.tensor import TensorVariable


def add_hierarchical_effect(
    name: str = "country",
    loc_mu: float = 0.0,
    loc_sigma: float = 1.0,
    scale_alpha: float = 2.0,
    scale_beta: float = 1.0,
    use_zerosum_offset: bool = False,
    group_dim: str | None = None,
) -> tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
    """
    Add a hierarchical effect to the active PyMC model.

    Parameters
    ----------
    name : str, optional
        Name of the effect, prefixing every variable created. Default 'country'.
    loc_mu : float, optional
        Mean of the Normal location parameter. Default 0.0.
    loc_sigma : float, optional
        Standard deviation of the Normal location parameter. Default 1.0.
    scale_alpha : float, optional
        Alpha parameter of the Gamma scale parameter. Default 2.0.
    scale_beta : float, optional
        Beta parameter of the Gamma scale parameter. Default 1.0.
    use_zerosum_offset : bool, optional
        Model the offset as a ZeroSumNormal, which identifies it against the location. Default
        False, which uses a Normal.
    group_dim : str
        Dimension the effect varies over, such as 'ISO' for countries.

    Returns
    -------
    effect : TensorVariable
        The effect itself, the location plus the scaled offset.
    effect_loc : TensorVariable
        Location of the distribution over effects.
    effect_scale : TensorVariable
        Scale of the distribution over effects.
    effect_offset : TensorVariable
        Offset of each group from the location.

    Examples
    --------
    .. code-block:: python

        import pymc as pm

        from climate_risk.models.blocks import add_hierarchical_effect

        with pm.Model(coords={"country": ["LAO", "THA"]}) as model:
            effect, loc, scale, offset = add_hierarchical_effect("country", group_dim="country")
    """
    if group_dim is None:
        raise ValueError("group_dim must be provided")

    with pm.modelcontext(None):
        country_effect_loc = pm.Normal(f"{name}_effect_loc", mu=loc_mu, sigma=loc_sigma)
        country_effect_scale = pm.Gamma(f"{name}_effect_scale", alpha=scale_alpha, beta=scale_beta)

        if use_zerosum_offset:
            country_effect_offset = pm.ZeroSumNormal(f"{name}_effect_offset", sigma=1, dims=group_dim)
        else:
            country_effect_offset = pm.Normal(f"{name}_effect_offset", sigma=1, dims=group_dim)
        country_effect = pm.Deterministic(
            f"{name}_effect",
            country_effect_loc + country_effect_scale * country_effect_offset,
            dims=group_dim,
        )

    return (
        country_effect,
        country_effect_loc,
        country_effect_scale,
        country_effect_offset,
    )


def add_data(
    features: list[str],
    df: pd.DataFrame,
    target: str | None = None,
    name: str | None = None,
    dims: Sequence[str] | None = None,
) -> TensorVariable | tuple[TensorVariable, TensorVariable]:
    """Add data to the active PyMC model.

    Parameters
    ----------
    features : list of str
        Features to include in the model. Each entry must be a column of ``df``.
    df : DataFrame
        Data to add.
    target : str, optional
        Column to use for the targets. Default None, which returns no target data.
    name : str, optional
        Suffix for the variable names, giving ``X_name`` and ``Y_name``. Default None, which names
        them ``X`` and ``Y``.
    dims : sequence of str, optional
        Named dimensions to include on the data. If targets are requested, only the first dimension is used for the
        targets, which is assumed to be the batch dimension.

    Returns
    -------
    X : TensorVariable
        The data tensor.
    Y : TensorVariable
        The target tensor, returned only when ``target`` is given.

    Examples
    --------
    .. code-block:: python

        import pymc as pm

        from climate_risk.models.blocks import add_data

        with pm.Model() as model:
            features, target = add_data(["co2", "precip"], panel, target="counts")
    """
    X_name = "X" if name is None else f"X_{name}"
    Y_name = "Y" if name is None else f"Y_{name}"

    with pm.modelcontext(None):
        X: TensorVariable = pm.Data(X_name, df[features], dims=dims)

        if target is not None:
            Y: TensorVariable = pm.Data(
                Y_name,
                df[target],
                dims=dims[0] if dims is not None else dims,
            )
            return X, Y

    return X


def compute_center(X: TensorVariable | np.ndarray) -> np.ndarray:
    """
    Return the midpoint of each column's range.

    Centering on the midpoint of the bounding box rather than the mean keeps the centering
    independent of how the points are distributed inside it.

    Parameters
    ----------
    X : TensorVariable or ndarray
        Coordinates, one row per point.

    Returns
    -------
    center : ndarray
        One midpoint per column.

    Examples
    --------
    .. code-block:: python

        import numpy as np

        from climate_risk.models.blocks import compute_center

        coordinates = np.array([[0.0, 10.0], [4.0, 20.0]])
        print(compute_center(coordinates))
    """
    center: np.ndarray = (pt.max(X, axis=0) + pt.min(X, axis=0)).eval() / 2

    return center


def set_plotting_data(df: pd.DataFrame, features: list[str], ISO_list: list[str]) -> None:
    """
    Swap the data on the model currently on the context stack for a plotting grid.

    Parameters
    ----------
    df : DataFrame
        Frame holding the feature columns to set.
    features : list of str
        Feature columns to replace on the model.
    ISO_list : list of str
        Country codes the rows belong to, used to reset the country coordinate.
    """
    iso_idx = df["ISO"].apply(lambda x: ISO_list.index(x))

    pm.set_data(
        {
            "X_gp": df[["lat", "lon"]],
            "Y": np.full(df.shape[0], 0),
            "ISO_idx": iso_idx,
            "X": df[features],
            "is_island": df["is_island"],
        },
        coords={"obs_idx": df.index.values},
    )
