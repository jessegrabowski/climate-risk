import geopandas as gpd
import pandas as pd
import xarray as xr

from climate_risk.geo.crs import GEOGRAPHIC_CRS


def prediction_to_gpd_df(
    prediction_idata: xr.DataTree, variables: list[str], points: pd.DataFrame
) -> dict[str, gpd.GeoDataFrame]:
    """
    Attach posterior predictions to the points they were made at.

    Parameters
    ----------
    prediction_idata : DataTree
        Inference data holding the posterior predictive draws.
    variables : list of str
        Names of the predicted variables to extract.
    points : DataFrame
        Points the predictions correspond to, in the order the model saw them.

    Returns
    -------
    predictions : dict mapping str to GeoDataFrame
        One frame per requested variable, each carrying the point geometry and the draws.

    Examples
    --------
    .. code-block:: python

        from climate_risk.models.predictions import prediction_to_gpd_df

        frames = prediction_to_gpd_df(idata, ["intensity"], grid_points)
    """
    predictions_dict = {}
    predictions_dict_geo = {}

    for variable in variables:
        # Tranform predictions to DF
        predictions_dict[variable] = (
            prediction_idata.posterior_predictive.mean(dim=("chain", "draw"))[variable].to_dataframe().reset_index()
        )
        predictions_dict[variable] = pd.merge(
            predictions_dict[variable],
            points,
            left_index=True,
            right_index=True,
            how="left",
        )

        # Transform into geo Data Frame
        predictions_dict_geo[variable] = gpd.GeoDataFrame(
            predictions_dict[variable],
            geometry=gpd.points_from_xy(predictions_dict[variable]["lon"], predictions_dict[variable]["lat"]),
            crs=GEOGRAPHIC_CRS,
        )

    return predictions_dict_geo
