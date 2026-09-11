"""The gallery pages under ``examples/`` are documentation, not a package module, so they have no
sister file. They earn a test because each one renders its source declaration into a committed cell
output, and nothing re-runs a notebook when the declaration beneath it changes."""

import dataclasses
import json

from pathlib import Path

import pytest

from climate_risk.data.co2 import CO2
from climate_risk.data.fred import FRED
from climate_risk.data.gadm import GADM
from climate_risk.data.geo_disasters import GEO_DISASTERS
from climate_risk.data.geonames import country_dump
from climate_risk.data.ghsl import population_source
from climate_risk.data.gpcc import FULL_DATA
from climate_risk.data.hadcrut import HADCRUT
from climate_risk.data.ipcc import IPCC
from climate_risk.data.ocean_heat import OCEAN_HEAT
from climate_risk.data.pwt import PWT
from climate_risk.data.source import ApiSource, DataSource, ManualSource, VendoredSource
from climate_risk.data.world_bank import WORLD_BANK
from climate_risk.data_functions.emdat_processing import EMDAT
from climate_risk.data_functions.rivers_data_loader import RIVERS

GALLERY = Path(__file__).resolve().parents[1] / "examples"

# Every page and the declaration it renders, keyed by the section folder and stem the gallery uses.
# A new vignette is added here, and a license-walled source will widen the annotation to its own
# declaration type.
DECLARED: dict[str, DataSource | VendoredSource | ApiSource | ManualSource] = {
    "climate/co2": CO2,
    "climate/ocean_heat": OCEAN_HEAT,
    "climate/hadcrut": HADCRUT,
    "climate/gpcc": FULL_DATA.sources[0],
    "climate/ipcc": IPCC,
    "hazard/emdat": EMDAT,
    "hazard/geo_disasters": GEO_DISASTERS,
    "economic/world_bank": WORLD_BANK,
    "economic/fred": FRED,
    "economic/pwt": PWT,
    "geospatial/gadm": GADM,
    "geospatial/ghsl": population_source(2020),
    "geospatial/geonames": country_dump("KH"),
    "geospatial/rivers": RIVERS,
}


def provenance(name: str) -> str:
    """Return the provenance table a page last committed, as the markdown its cell emitted."""
    notebook = json.loads((GALLERY / f"{name}.ipynb").read_text())
    tables = [
        "".join(output["data"]["text/markdown"])
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if "text/markdown" in output.get("data", {})
    ]

    assert len(tables) == 1, f"{name} renders {len(tables)} markdown outputs, expected the provenance table"

    return tables[0]


def test_every_gallery_page_is_declared():
    """A page in a section nobody listed here is a page nothing checks, which is how a whole
    section joins the gallery unnoticed."""
    published = {f"{path.parent.name}/{path.stem}" for path in GALLERY.glob("*/*.ipynb")}

    assert sorted(DECLARED) == sorted(published)


@pytest.mark.parametrize("name", DECLARED, ids=DECLARED)
def test_a_page_shows_every_field_its_declaration_carries(name):
    """A license stated more narrowly than the declaration is a legal claim, not a wording choice.
    Every other field goes stale by the same route, so the whole row set is pinned."""
    table = provenance(name)

    for field, value in dataclasses.asdict(DECLARED[name]).items():
        assert f"| {field} | {value} |" in table, field
