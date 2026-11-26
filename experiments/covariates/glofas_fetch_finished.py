"""
Download GloFAS requests that finished on the server while no worker was waiting (for example after a reboot).

Pass the job ids as ``year=jobid``. A year whose raw file already exists is skipped. This avoids queueing again.
Usage: ``pixi run python experiments/covariates/glofas_fetch_finished.py 1983=<jobid> 2025=<jobid>``
"""

import logging
import sys

from pathlib import Path

from climate_risk.data.copernicus import EWDS_URL, make_client

RAW = Path("data/raw/covariates/global/glofas")
_log = logging.getLogger("glofas_fetch")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = make_client(EWDS_URL)
    for pair in sys.argv[1:]:
        year, job = pair.split("=")
        target = RAW / f"glofas_{year}.nc"
        if target.exists():
            continue
        partial = target.with_suffix(".nc.part")
        client.client.get_remote(job).download(str(partial))
        partial.replace(target)
        _log.info("%s downloaded", year)
