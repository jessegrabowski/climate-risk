import pandas as pd

from sklearn.preprocessing import StandardScaler


def standardize(
    df: pd.DataFrame, columns: list[str], transformer_fitted: StandardScaler | None = None
) -> tuple[StandardScaler, pd.DataFrame]:
    """
    Append centered and scaled copies of ``columns``, suffixed ``__standardized``.

    Parameters
    ----------
    df : DataFrame
        Data to standardize. The original columns survive alongside the new ones.
    columns : list of str
        Columns to standardize.
    transformer_fitted : StandardScaler, optional
        A scaler already fitted to training data, so held-out data reuses the training mean and
        scale. Default None, which fits a new scaler on ``df``.

    Returns
    -------
    scaler : StandardScaler
        The fitted scaler, to pass back for held-out data.
    standardized : DataFrame
        ``df`` with one ``__standardized`` column per entry in ``columns``.

    Examples
    --------
    Fit on the training frame, then reuse the scaler on held-out data:

    .. code-block:: python

        import numpy as np
        import pandas as pd

        from climate_risk.stats.scaling import standardize

        rng = np.random.default_rng(0)
        train = pd.DataFrame({"gdp": rng.normal(10, 2, 100)})
        held_out = pd.DataFrame({"gdp": rng.normal(10, 2, 20)})

        scaler, train_scaled = standardize(train, ["gdp"])
        _, held_out_scaled = standardize(held_out, ["gdp"], transformer_fitted=scaler)
    """
    if transformer_fitted is None:
        transformer_fitted = StandardScaler().fit(df[columns])

    transformer_fitted.set_output(transform="pandas")
    standardized = transformer_fitted.transform(df[columns]).add_suffix("__standardized")

    return transformer_fitted, pd.concat([df, standardized], axis=1)
