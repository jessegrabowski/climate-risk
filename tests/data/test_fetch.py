import pytest
import requests

from climate_risk.data.fetch import ATTEMPTS, USER_AGENT, fetch
from climate_risk.data.source import DataSource

PAYLOAD = b"Year,co2\n1990,354.4\n"


def make_source(**overrides) -> DataSource:
    fields = {
        "url": "https://example.org/co2.csv",
        "filename": "noaa_co2.csv",
        "license": "public domain",
        "citation": "NOAA GML",
        "retrieved": "2026-08-03",
    }
    return DataSource(**(fields | overrides))


class FakeResponse:
    def __init__(self, payload: bytes, status_error: Exception | None = None, status_code: int = 200, stop_after=None):
        self.payload = payload
        self.status_error = status_error
        self.status_code = status_code
        self.stop_after = stop_after
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    # Fixed small pieces regardless of what the caller asks for, so a drop part-way through a
    # payload is expressible without a megabyte of fixture.
    PIECE = 64

    def iter_content(self, chunk_size):
        sent = 0
        for start in range(0, len(self.payload), self.PIECE):
            if self.stop_after is not None and sent >= self.stop_after:
                raise requests.ConnectionError("the host dropped the transfer")
            chunk = self.payload[start : start + self.PIECE]
            sent += len(chunk)
            yield chunk


@pytest.fixture
def downloads(monkeypatch):
    """Record every request the code makes, and serve PAYLOAD without touching a socket."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return FakeResponse(PAYLOAD)

    monkeypatch.setattr(requests, "get", fake_get)

    return calls


def test_the_file_lands_in_the_cache(tmp_path, downloads):
    path = fetch(make_source(), tmp_path)

    assert path == tmp_path / "noaa_co2.csv"
    assert path.read_bytes() == PAYLOAD


def test_a_present_file_is_not_downloaded_again(tmp_path, downloads):
    """Assert on what the caller gets, so this survives any change to how downloading works."""
    source = make_source()
    source.path(tmp_path).write_bytes(b"already here")

    assert fetch(source, tmp_path).read_bytes() == b"already here"


def test_force_downloads_over_a_present_file(tmp_path, downloads):
    fetch(make_source(), tmp_path)
    fetch(make_source(), tmp_path, force=True)

    assert len(downloads) == 2


def test_a_browser_agent_is_sent(tmp_path, downloads):
    """Hosts reject urllib's default agent; the World Bank archive answers 403 without one."""
    fetch(make_source(), tmp_path)

    assert downloads[0]["headers"]["User-Agent"] == USER_AGENT


def test_an_interrupted_download_is_not_mistaken_for_a_cache_hit(tmp_path, downloads):
    """A partial multi-GB archive left in the cache would otherwise be read as the real file."""
    leftover = tmp_path / "noaa_co2.csv.part"
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(b"half a giga")

    path = fetch(make_source(), tmp_path)

    assert path.read_bytes() == PAYLOAD
    assert not leftover.exists()


def test_an_http_error_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        requests, "get", lambda url, **kwargs: FakeResponse(b"", status_error=requests.HTTPError("404"))
    )

    with pytest.raises(requests.HTTPError):
        fetch(make_source(), tmp_path)

    assert not (tmp_path / "noaa_co2.csv").exists()


def test_a_cache_directory_that_does_not_exist_yet_is_created(tmp_path, downloads):
    """Loaders point at subdirectories of the cache, which nothing has made on a fresh clone."""
    nested = tmp_path / "gpcc" / "raw"

    assert fetch(make_source(), nested).read_bytes() == PAYLOAD


LONG_PAYLOAD = b"".join(f"{year},{year % 97}\n".encode() for year in range(1900, 2001))


def serve_with_ranges(monkeypatch, payload, *, drop_first_after=None, ignore_range=False):
    """A host that honours Range, and optionally drops the first transfer part-way through."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        start = 0
        offered = kwargs.get("headers", {}).get("Range")
        if offered and not ignore_range:
            start = int(offered.removeprefix("bytes=").rstrip("-"))
            return FakeResponse(payload[start:], status_code=206)
        return FakeResponse(payload, status_code=200)

    def dropping_get(url, **kwargs):
        response = fake_get(url, **kwargs)
        if len(calls) == 1:
            response.stop_after = drop_first_after
        return response

    monkeypatch.setattr(requests, "get", dropping_get if drop_first_after else fake_get)

    return calls


def test_a_dropped_transfer_continues_from_the_bytes_already_written(tmp_path, monkeypatch):
    """A host serving half-gigabyte rasters drops long transfers, and restarting each time never
    converges. The second attempt asks for the remainder rather than the whole file."""
    calls = serve_with_ranges(monkeypatch, LONG_PAYLOAD, drop_first_after=len(LONG_PAYLOAD) // 4)

    path = fetch(make_source(), tmp_path)

    assert path.read_bytes() == LONG_PAYLOAD
    assert "Range" not in calls[0]["headers"]
    assert calls[1]["headers"]["Range"].startswith("bytes=")


def test_a_host_ignoring_the_range_restarts_rather_than_doubling(tmp_path, monkeypatch):
    """Answering a ranged request with the whole file is allowed. Appending it to what is already
    on disk would write a file of the right name and twice the length."""
    serve_with_ranges(monkeypatch, LONG_PAYLOAD, drop_first_after=len(LONG_PAYLOAD) // 4, ignore_range=True)

    assert fetch(make_source(), tmp_path).read_bytes() == LONG_PAYLOAD


def test_a_short_transfer_is_not_moved_into_place(tmp_path, monkeypatch):
    """The declared length is the only check that the bytes are all there; without it a truncated
    archive gets the real filename and every later read fails somewhere further away."""

    class ShortResponse(FakeResponse):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.headers = {"Content-Length": str(len(PAYLOAD) * 2)}

    monkeypatch.setattr(requests, "get", lambda url, **kwargs: ShortResponse(PAYLOAD))

    with pytest.raises(requests.ConnectionError, match="declared"):
        fetch(make_source(), tmp_path)

    assert not (tmp_path / "noaa_co2.csv").exists()


def test_a_host_that_keeps_dropping_gives_up(tmp_path, monkeypatch):
    """Retrying forever on a host that is down hangs a loader with no way to tell why."""
    attempts = []

    def always_drops(url, **kwargs):
        attempts.append(url)
        return FakeResponse(LONG_PAYLOAD, status_code=200, stop_after=0)

    monkeypatch.setattr(requests, "get", always_drops)

    with pytest.raises(requests.ConnectionError):
        fetch(make_source(), tmp_path)

    assert len(attempts) == ATTEMPTS


def test_a_complete_partial_is_accepted_rather_than_refetched(tmp_path, monkeypatch):
    """A transfer that dropped after the last byte leaves a whole file waiting to be renamed. Asking
    for bytes past the end answers 416, which means done rather than failed."""
    requested = []

    class DropsAfterTheLastByte(FakeResponse):
        def iter_content(self, chunk_size):
            yield self.payload
            raise requests.ConnectionError("the host dropped after the last byte")

    def range_not_satisfiable(url, **kwargs):
        requested.append(kwargs.get("headers", {}).get("Range"))
        if len(requested) == 1:
            return DropsAfterTheLastByte(PAYLOAD, status_code=200)
        return FakeResponse(b"", status_code=416)

    monkeypatch.setattr(requests, "get", range_not_satisfiable)

    assert fetch(make_source(), tmp_path).read_bytes() == PAYLOAD
    assert requested[1] == f"bytes={len(PAYLOAD)}-"
