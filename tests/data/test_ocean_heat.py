from datetime import date

import polars as pl
import pytest
import requests

from climate_risk.data.ocean_heat import OCEAN_HEAT, load_ocean_heat_data, transform_ocean_heat

# NCEI serves no header row, so the first line is an observation. The record is quarterly and the
# month is written unpadded below October, so both spellings appear here.
PUBLISHED = "1960-3,0.0\n1960-6,2.0\n1961-9,10.0\n1961-12,20.0\n"


@pytest.fixture
def published(monkeypatch):
    """Serve the NCEI file without a socket, recording each request so a warm hit is observable."""
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


@pytest.fixture
def seasonal():
    return pl.DataFrame({"Date": ["1960-3", "1960-6", "1961-9", "1961-12"], "Temp": [0.0, 2.0, 10.0, 20.0]})


def test_annual_means_are_shifted_onto_the_baseline(seasonal):
    """The offset is stated literally: the published results move if it is ever retuned."""
    annual = transform_ocean_heat(seasonal)

    assert annual["Temp"].to_list() == pytest.approx([153.0, 167.0])


def test_a_month_upstream_left_unpadded_is_still_read(seasonal):
    """NCEI writes 1960-3, not 1960-03. A %Y-%m parse rejects 214 of the 285 published rows."""
    annual = transform_ocean_heat(seasonal)

    assert annual["Date"].to_list() == [date(1960, 1, 1), date(1961, 1, 1)]
    assert annual["Temp"].to_list() == pytest.approx([153.0, 167.0])


def test_years_are_stamped_at_their_start(seasonal):
    """The panel joins on a year-start timestamp, which resampling leaves at year-end."""
    annual = transform_ocean_heat(seasonal)

    assert annual["Date"].to_list() == [date(1960, 1, 1), date(1961, 1, 1)]


def test_the_columns_are_named_by_the_loader(tmp_path, published):
    frame = load_ocean_heat_data(tmp_path)

    assert frame.columns == ["Date", "Temp"]


def test_the_first_published_season_is_not_read_as_a_header(tmp_path, published):
    """Reading a header off a headerless file drops the earliest observation, and the year it falls
    in is then averaged over the seasons that survive."""
    frame = load_ocean_heat_data(tmp_path)

    assert len(frame) == 2, "both years survive"
    assert frame["Temp"].to_list() == pytest.approx([153.0, 167.0]), "1960 averages 0.0 and 2.0"


def test_a_warm_cache_does_not_reach_the_network(tmp_path, published):
    load_ocean_heat_data(tmp_path)
    load_ocean_heat_data(tmp_path)

    assert len(published) == 1


def test_the_processed_cache_survives_the_raw_download_being_deleted(tmp_path, published):
    """Nobody keeps the raw files. This is the test that proves fetch runs inside the builder."""
    load_ocean_heat_data(tmp_path)
    OCEAN_HEAT.path(tmp_path).unlink()

    frame = load_ocean_heat_data(tmp_path)

    assert len(published) == 1
    assert frame["Temp"].to_list() == pytest.approx([153.0, 167.0])


def test_force_reload_goes_back_to_the_source(tmp_path, published):
    load_ocean_heat_data(tmp_path)
    load_ocean_heat_data(tmp_path, force_reload=True)

    assert len(published) == 2


def test_the_raw_download_is_named_for_the_file_upstream_serves(tmp_path, published):
    """The old cache used this loader's name for processed rows; reusing it would misread them."""
    load_ocean_heat_data(tmp_path)

    assert OCEAN_HEAT.path(tmp_path).name == "ohc_levitus_climdash_seasonal.csv"
    assert [path.name for path in tmp_path.glob("ocean_heat*.parquet")], "the processed rows land elsewhere"
