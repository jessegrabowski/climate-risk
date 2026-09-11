import pandas as pd


def descriptive_stats_function(df: pd.DataFrame, varlist: list[str]) -> pd.DataFrame:
    """
    Summarize columns with the usual descriptive statistics, plus kurtosis and skewness.

    Parameters
    ----------
    df : DataFrame
        Frame holding the columns to summarize.
    varlist : list of str
        Columns to summarize.

    Returns
    -------
    summary : DataFrame
        The output of :meth:`pandas.DataFrame.describe`, with ``kurtosis`` and ``skewness`` rows
        appended.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pandas as pd

        from climate_risk.stats.descriptive import descriptive_stats_function

        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"damage": rng.lognormal(size=200), "deaths": rng.poisson(3, size=200)})

        print(descriptive_stats_function(frame, ["damage", "deaths"]))
    """
    summary = df[varlist].describe()
    kurtosis = pd.Series({column: df[column].kurt() for column in varlist})
    skewness = pd.Series({column: df[column].skew() for column in varlist})

    return pd.concat(
        [
            summary,
            kurtosis.to_frame().rename(columns={0: "kurtosis"}).transpose(),
            skewness.to_frame().rename(columns={0: "skewness"}).transpose(),
        ]
    )
