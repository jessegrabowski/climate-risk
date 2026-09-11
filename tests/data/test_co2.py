from datetime import date

import polars as pl
import pytest
import requests

from climate_risk.data.co2 import CO2, load_co2_data, transform_co2

# NOAA publishes 43 lines of license and methodology above the header. Stated literally, not
# derived from the loader, so a wrong skiprows fails instead of agreeing with itself.
PREAMBLE_LINES = 43
PUBLISHED = "# comment\n" * PREAMBLE_LINES + "year,mean,unc\n1959,315.98,0.12\n1960,316.91,0.12\n"


@pytest.fixture
def published(monkeypatch):
    """Serve the NOAA file without a socket, recording each request so a warm hit is observable."""
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


def test_the_annual_means_become_a_dated_co2_column():
    raw = pl.DataFrame({"year": [1959, 1960], "mean": [315.98, 316.91], "unc": [0.12, 0.12]})

    frame = transform_co2(raw)

    assert frame.columns == ["Date", "co2"]
    assert frame["Date"].to_list() == [date(1959, 1, 1), date(1960, 1, 1)]
    assert frame["co2"].to_list() == [315.98, 316.91]


def test_the_uncertainty_column_is_dropped():
    """Downstream sums the frame, so an extra numeric column would silently join the total."""
    raw = pl.DataFrame({"year": [1959], "mean": [315.98], "unc": [0.12]})

    assert "unc" not in transform_co2(raw).columns


def test_the_published_preamble_is_skipped(tmp_path, published):
    frame = load_co2_data(tmp_path)

    assert frame["co2"].to_list() == [315.98, 316.91]


def test_a_warm_cache_does_not_reach_the_network(tmp_path, published):
    """Four loaders have had this bug: the download ran even when the processed cache was warm."""
    load_co2_data(tmp_path)
    load_co2_data(tmp_path)

    assert len(published) == 1


def test_the_processed_cache_survives_the_raw_download_being_deleted(tmp_path, published):
    """Nobody keeps the raw files; deleting them must not force a re-download."""
    load_co2_data(tmp_path)
    CO2.path(tmp_path).unlink()

    frame = load_co2_data(tmp_path)

    assert len(published) == 1
    assert frame["co2"].to_list() == [315.98, 316.91]


def test_force_reload_goes_back_to_the_source(tmp_path, published):
    load_co2_data(tmp_path)
    load_co2_data(tmp_path, force_reload=True)

    assert len(published) == 2


def test_the_raw_download_is_named_for_the_file_upstream_serves(tmp_path, published):
    """A processed cache under the same name would be read back as a fresh download."""
    load_co2_data(tmp_path)

    assert CO2.path(tmp_path).name == "co2_annmean_mlo.csv"
    assert (tmp_path / "co2.parquet").exists()
