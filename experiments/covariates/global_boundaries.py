"""
Cut the GADM 4.1 file into one shape per country and one per first-level region (admin-1).

GADM stores every country at its finest level, so a country is the union of its rows. This reads the
file in chunks, merges each chunk's rows, then merges the partial pieces. Shapes are simplified by about
200 m, far below the covariate pixel size (5 to 25 km), which makes the later pixel-overlap step fast.
Results: data/processed/covariates/global/boundaries_{country,admin1}.gpkg (git-ignored; GADM is
non-commercial and may not be redistributed).

    pixi run python experiments/covariates/global_boundaries.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio

GADM = Path("data/gadm/gadm_410.gpkg")
OUT = Path("data/processed/covariates/global")
CHUNK = 15000
SIMPLIFY_DEGREES = 0.002


def main() -> None:
    total = pyogrio.read_info(GADM)["features"]
    pieces = []
    for start in range(0, total, CHUNK):
        rows = pyogrio.read_dataframe(
            GADM, columns=["GID_0", "NAME_0", "GID_1", "NAME_1"], skip_features=start, max_features=CHUNK
        )
        rows["GID_1"] = rows["GID_1"].fillna(rows["GID_0"])  # countries without admin-1 stay as one unit
        rows["geometry"] = rows.geometry.simplify(SIMPLIFY_DEGREES, preserve_topology=True)
        pieces.append(rows.dissolve(by=["GID_0", "NAME_0", "GID_1", "NAME_1"]).reset_index())
        print(f"{min(start + CHUNK, total)} / {total}", flush=True)
    admin1 = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs="EPSG:4326")
    admin1 = admin1.dissolve(by=["GID_0", "NAME_0", "GID_1", "NAME_1"]).reset_index()
    country = admin1.dissolve(by=["GID_0", "NAME_0"]).reset_index()[["GID_0", "NAME_0", "geometry"]]
    OUT.mkdir(parents=True, exist_ok=True)
    admin1.to_file(OUT / "boundaries_admin1.gpkg", driver="GPKG")
    country.to_file(OUT / "boundaries_country.gpkg", driver="GPKG")
    print(len(country), "countries,", len(admin1), "admin-1 units")


if __name__ == "__main__":
    main()
