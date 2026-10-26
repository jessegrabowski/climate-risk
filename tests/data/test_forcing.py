import polars as pl
import pytest
import requests

from climate_risk.data.forcing import FORCING_ESTIMATES, forcing_source, load_effective_radiative_forcing
from climate_risk.exceptions import DataValidationError

# The header Annex III publishes for the scenario tables, three years of it.
PUBLISHED = (
    "year,co2,ch4,aerosol-cloud_interactions,volcanic,total_anthropogenic,total_natural,total\n"
    "1750,0.0,0.0,0.0,0.19,0.0,0.29,0.29\n"
    "1751,0.0014,0.0005,0.0047,0.19,0.007,0.27,0.28\n"
    "1752,0.0028,0.0010,0.0094,0.18,0.014,0.26,0.27\n"
)


@pytest.fixture
def published(monkeypatch):
    """Serve one Annex III table without a socket, recording each request so a warm hit is observable."""
    calls = []

    class Response:
        headers = {"Content-Length": str(len(PUBLISHED))}

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield PUBLISHED.encode()

    def fake_get(url, **kwargs):
        calls.append(url)
        return Response()

    monkeypatch.setattr(requests, "get", fake_get)

    return calls


@pytest.mark.parametrize(
    ("scenario", "table"), [("historical", "A3.3_historical_ERF_1750-2019"), ("ssp245", "A3.4c_ssp245_ERF_1750-2500")]
)
def test_each_scenario_names_its_own_table(scenario, table):
    """The tables are lettered by scenario, so a wrong letter would load another scenario's forcing and
    nothing downstream would notice."""
    source = forcing_source(scenario, "best_estimate")

    assert source.filename == f"table_{table}_best_estimate.csv"
    assert source.url.startswith("https://zenodo.org/records/5705391/files/")
    assert source.filename in source.url


def test_a_scenario_or_estimate_annex_iii_does_not_publish_is_an_error():
    """A typo would otherwise surface as a 404 halfway through a download."""
    with pytest.raises(DataValidationError, match="ssp246"):
        forcing_source("ssp246", "best_estimate")
    with pytest.raises(DataValidationError, match="median"):
        forcing_source("ssp245", "median")


def test_every_published_estimate_has_a_table():
    """The estimates are stated as Zenodo spells them, so a misspelled one in the loader fails here
    instead of as a 404."""
    assert {forcing_source("ssp585", estimate).filename for estimate in FORCING_ESTIMATES} == {
        "table_A3.4e_ssp585_ERF_1750-2500_best_estimate.csv",
        "table_A3.4e_ssp585_ERF_1750-2500_5th_percentile.csv",
        "table_A3.4e_ssp585_ERF_1750-2500_95th_percentile.csv",
    }


def test_the_chosen_table_is_downloaded_into_the_forcing_directory(tmp_path, published):
    """The layout is stated literally, since a path derived from the loader passes wherever it looks."""
    load_effective_radiative_forcing("ssp245", "best_estimate", tmp_path)

    assert "table_A3.4c_ssp245_ERF_1750-2500_best_estimate.csv" in published[0]
    assert (tmp_path / "forcing" / "table_A3.4c_ssp245_ERF_1750-2500_best_estimate.csv").exists()


def test_a_warm_cache_does_not_reach_the_network(tmp_path, published):
    load_effective_radiative_forcing("ssp245", "best_estimate", tmp_path)
    load_effective_radiative_forcing("ssp245", "best_estimate", tmp_path)

    assert len(published) == 1


def test_each_scenario_and_estimate_is_its_own_cache_entry(tmp_path, published):
    """One cache name serves every table, so the parameters are all that keep them apart."""
    load_effective_radiative_forcing("ssp245", "best_estimate", tmp_path)
    load_effective_radiative_forcing("ssp585", "best_estimate", tmp_path)
    load_effective_radiative_forcing("ssp245", "95th_percentile", tmp_path)

    assert len(published) == 3


def test_the_table_comes_back_one_row_per_year_with_every_agent(tmp_path, published):
    forcing = load_effective_radiative_forcing("ssp245", "best_estimate", tmp_path)

    assert forcing["year"].dtype == pl.Int64
    assert forcing["year"].to_list() == [1750, 1751, 1752]
    assert forcing.columns[-3:] == ["total_anthropogenic", "total_natural", "total"]
    assert forcing["total"].to_list() == [0.29, 0.28, 0.27]


def test_force_reload_goes_back_to_the_source(tmp_path, published):
    load_effective_radiative_forcing("ssp245", "best_estimate", tmp_path)
    load_effective_radiative_forcing("ssp245", "best_estimate", tmp_path, force_reload=True)

    assert len(published) == 2
