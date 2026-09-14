from datetime import date

import polars as pl
import pytest
import requests

from climate_risk.data.co2 import CO2, load_co2_data, transform_co2

# NOAA publishes 40 lines of license and methodology above the header. Stated literally, not
# derived from the loader, so a wrong skiprows fails instead of agreeing with itself.
PREAMBLE_LINES = 40

# Two whole years, each held flat, so the annual mean reads back as the monthly value.
PUBLISHED = (
    "# comment\n" * PREAMBLE_LINES
    + "year,month,decimal date,average,deseasonalized,ndays,sdev,unc\n"
    + "".join(
        f"{year},{month},{year + (month - 0.5) / 12:.4f},{level},{level},25,0.5,0.1\n"
        for year, level in ((1959, 315.98), (1960, 316.91))
        for month in range(1, 13)
    )
)


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


def test_the_monthly_means_become_a_dated_co2_column():
    raw = pl.DataFrame({"year": [1959, 1959], "month": [3, 11], "average": [315.71, 315.02], "unc": [0.1, 0.1]})

    frame = transform_co2(raw)

    assert frame.columns == ["Date", "co2"]
    assert frame["Date"].to_list() == [date(1959, 3, 1), date(1959, 11, 1)]
    assert frame["co2"].to_list() == [315.71, 315.02]


def test_the_publishers_other_columns_are_dropped():
    """Downstream averages the frame, so an extra numeric column would silently join the mean."""
    raw = pl.DataFrame({"year": [1959], "month": [3], "average": [315.71], "deseasonalized": [314.4], "unc": [0.1]})

    assert transform_co2(raw).columns == ["Date", "co2"]


def test_the_published_preamble_is_skipped(tmp_path, published):
    frame = load_co2_data(tmp_path)

    assert frame["co2"].to_list() == [pytest.approx(315.98), pytest.approx(316.91)]


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
    assert frame["co2"].to_list() == [pytest.approx(315.98), pytest.approx(316.91)]


def test_force_reload_goes_back_to_the_source(tmp_path, published):
    load_co2_data(tmp_path)
    load_co2_data(tmp_path, force_reload=True)

    assert len(published) == 2


def test_the_raw_download_is_named_for_the_file_upstream_serves(tmp_path, published):
    """A processed cache under the same name would be read back as a fresh download."""
    load_co2_data(tmp_path)

    assert CO2.path(tmp_path).name == "co2_mm_mlo.csv"
    assert (tmp_path / "co2__resolution=monthly.parquet").exists()


def test_the_default_is_the_annual_mean_of_the_months(tmp_path, published):
    frame = load_co2_data(tmp_path)

    assert frame["Date"].to_list() == [date(1959, 1, 1), date(1960, 1, 1)]
    assert frame["co2"].to_list() == [pytest.approx(315.98), pytest.approx(316.91)]


def test_a_monthly_request_returns_every_published_month(tmp_path, published):
    frame = load_co2_data(tmp_path, frequency="monthly")

    assert len(frame) == 24
    assert frame["Date"][0] == date(1959, 1, 1)
    assert frame["Date"][-1] == date(1960, 12, 1)
