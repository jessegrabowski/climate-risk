import numpy as np
import pandas as pd

# Lat/lon as the upstream shapefiles and gridded products supply it.
GEOGRAPHIC_CRS = "EPSG:4326"

# World Mercator. Distances measured in it come back in meters, which every caller assumes.
PROJECTED_CRS = "EPSG:3395"

METERS_PER_KM = 1000


def to_km[Distance: (float, np.ndarray, pd.Series)](meters: Distance) -> Distance:
    """
    Convert a distance measured in ``PROJECTED_CRS`` to kilometers.

    Parameters
    ----------
    meters : float, ndarray or Series
        Distance in meters.

    Returns
    -------
    distance : float, ndarray or Series
        The same distance in kilometers, matching the input type.

    Examples
    --------
    .. code-block:: python

        import numpy as np

        from climate_risk.geo.crs import to_km

        print(to_km(np.array([1000.0, 2500.0])))
    """
    return meters / METERS_PER_KM
