"""The gallery pages under ``examples/`` are documentation, not a package module, so they have no
sister file. They earn a test because each one renders its source declaration into a committed cell
output, and nothing re-runs a notebook when the declaration beneath it changes."""

import dataclasses
import json

from pathlib import Path

import pytest

from climate_risk.data.co2 import CO2
from climate_risk.data.gpcc import FULL_DATA
from climate_risk.data.hadcrut import HADCRUT
from climate_risk.data.ipcc import IPCC
from climate_risk.data.ocean_heat import OCEAN_HEAT
from climate_risk.data.source import DataSource, VendoredSource

PAGES = Path(__file__).resolve().parents[1] / "examples" / "climate"

# Every page and the declaration it renders. A new vignette is added here, and a license-walled
# source will widen the annotation to its own declaration type.
DECLARED: dict[str, DataSource | VendoredSource] = {
    "co2": CO2,
    "ocean_heat": OCEAN_HEAT,
    "hadcrut": HADCRUT,
    "gpcc": FULL_DATA.sources[0],
    "ipcc": IPCC,
}


def provenance(name: str) -> str:
    """Return the provenance table a page last committed, as the markdown its cell emitted."""
    notebook = json.loads((PAGES / f"{name}.ipynb").read_text())
    tables = [
        "".join(output["data"]["text/markdown"])
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if "text/markdown" in output.get("data", {})
    ]

    assert len(tables) == 1, f"{name} renders {len(tables)} markdown outputs, expected the provenance table"

    return tables[0]


def test_every_declared_page_exists():
    """A renamed or moved page would otherwise leave its entry here checking nothing."""
    assert sorted(DECLARED) == sorted(path.stem for path in PAGES.glob("*.ipynb"))


@pytest.mark.parametrize("name", DECLARED, ids=DECLARED)
def test_a_page_shows_every_field_its_declaration_carries(name):
    """A license stated more narrowly than the declaration is a legal claim, not a wording choice.
    Every other field goes stale by the same route, so the whole row set is pinned."""
    table = provenance(name)

    for field, value in dataclasses.asdict(DECLARED[name]).items():
        assert f"| {field} | {value} |" in table, field
