"""Shared paths, country boxes and the manifest, used by every covariate script."""

import json

from datetime import date
from pathlib import Path

import numpy as np

from climate_risk.config.registry import load_place
from climate_risk.data_functions.shapefiles_data_loader import load_place_boundary, load_shapefile
from climate_risk.geo.raster import dissolve_place_boundary

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
COUNTRIES = ["NPL", "ZMB", "CRI", "LAO", "MAR"]
RAW_ROOT = DATA / "raw" / "covariates"
PROCESSED = DATA / "processed" / "covariates"
PAD_DEGREES = 0.5
FIRST_YEAR, LAST_YEAR = 1981, 2025


def raw_dir(iso3: str) -> Path:
    """Raw download folder of one country, git-ignored with the rest of ``data/``."""
    return RAW_ROOT / iso3


def country_boundary(iso3: str):
    """
    A country's dissolved boundary (one polygon, EPSG:4326).

    Reads the place config's own boundary archive when it has one. If that download is gone (the
    Laos archive on HDX now answers 404), falls back to the World Bank country outline, which is all
    a country-level grid needs.
    """
    try:
        return dissolve_place_boundary(load_place_boundary(load_place(iso3.lower()), DATA))
    except Exception as error:  # a dead archive URL surfaces as a requests error
        print(f"{iso3}: place boundary unavailable ({type(error).__name__}); using the World Bank outline")
        world = load_shapefile("world", DATA)
        return dissolve_place_boundary(world[world["ISO_A3"] == iso3])


def country_box(iso3: str) -> tuple[float, float, float, float]:
    """(west, south, east, north): the boundary padded by 0.5 degrees, rounded outward to 0.1."""
    west, south, east, north = country_boundary(iso3).total_bounds
    return (
        float(np.floor((west - PAD_DEGREES) * 10) / 10),
        float(np.floor((south - PAD_DEGREES) * 10) / 10),
        float(np.ceil((east + PAD_DEGREES) * 10) / 10),
        float(np.ceil((north + PAD_DEGREES) * 10) / 10),
    )


def record_source(iso3: str, source: str, **fields) -> None:
    """Write or update one source's row in a country's manifest, stamping today as the access date."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / f"manifest_{iso3}.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[source] = {**manifest.get(source, {}), **fields, "access_date": date.today().isoformat()}
    path.write_text(json.dumps(manifest, indent=2))
