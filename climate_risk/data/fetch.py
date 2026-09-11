import logging
import os

from pathlib import Path

import requests

from tqdm.auto import tqdm

from climate_risk.data.source import DataSource

_log = logging.getLogger(__name__)

# Data hosts reject urllib's default agent: the World Bank boundaries archive answers 403 to
# `Python-urllib/3.x` and 200 to a browser, from the same URL.
USER_AGENT = "climate-risk (+https://github.com/jessegrabowski/laos-climate-change)"

CHUNK_BYTES = 1 << 20
TIMEOUT_SECONDS = 60

# Hosts serving multi-hundred-megabyte rasters drop long transfers, so one read timeout is not a
# failure. Each attempt after the first asks for the bytes the last one stopped at.
ATTEMPTS = 5

PARTIAL_CONTENT = 206
RANGE_NOT_SATISFIABLE = 416


def fetch(source: DataSource, cache_dir: Path, *, force: bool = False) -> Path:
    """
    Download ``source`` into ``cache_dir`` unless it is already there, and return its path.

    The download goes to a sibling ``.part`` file and is moved into place only once complete, so an
    interrupted transfer can never be mistaken for a cache hit. A dropped connection is retried, and
    each retry continues from the bytes already written rather than starting the file again. A
    ``.part`` left by an earlier call is discarded instead: nothing proves it is a prefix of this
    file, and appending to bytes from elsewhere would write a corrupt archive that looks complete.

    Parameters
    ----------
    source : DataSource
        What to fetch.
    cache_dir : Path
        Directory to store the file in. Created if absent.
    force : bool, optional
        Re-download even when the file is already present. Default False.

    Returns
    -------
    path : Path
        The downloaded file.
    """
    destination = source.path(cache_dir)

    if destination.exists() and not force:
        return destination

    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)

    _log.info(f"Downloading {source.filename} from {source.url}")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            _download(source, partial)
            break
        except requests.RequestException as err:
            if attempt == ATTEMPTS:
                raise
            _log.warning(f"{source.filename}: attempt {attempt} of {ATTEMPTS} stopped ({err}), continuing from disk")

    os.replace(partial, destination)

    return destination


def _download(source: DataSource, partial: Path) -> None:
    """Write ``source`` into ``partial``, continuing from whatever it already holds."""
    have = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if have:
        headers["Range"] = f"bytes={have}-"

    with requests.get(source.url, stream=True, timeout=TIMEOUT_SECONDS, headers=headers) as response:
        if have and response.status_code == RANGE_NOT_SATISFIABLE:
            return

        response.raise_for_status()

        # A host may answer a ranged request with the whole file. Appending then would double it,
        # so the partial is thrown away and rewritten.
        continuing = have > 0 and response.status_code == PARTIAL_CONTENT
        already = have if continuing else 0
        expected = int(response.headers.get("Content-Length", 0))

        with (
            partial.open("ab" if continuing else "wb") as handle,
            tqdm(
                total=(expected + already) or None,
                initial=already,
                unit="B",
                unit_scale=True,
                desc=source.filename,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                handle.write(chunk)
                progress.update(len(chunk))

    written = partial.stat().st_size
    if expected and written != expected + already:
        # Retryable: the loop above continues from what is on disk rather than starting again.
        raise requests.ConnectionError(
            f"{source.filename}: {written} bytes written against {expected + already} declared"
        )
