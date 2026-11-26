"""
Download annual rx5day from CMIP6 (CDS ``sis-extreme-indices-cmip6``): 16 requests, one at a time (this dataset rejects a second queued request).

Each request is skipped if its file exists. A licence refusal stops everything and prints the CDS text.
Usage: ``pixi run python experiments/cmip6/download.py [--first]``   (--first sends only the smallest request, to test the licence)
"""

import logging
import sys
import time

from pathlib import Path

from climate_risk.data import cmip6
from climate_risk.data.copernicus import CDS_URL, SourceFailed, make_client, retrieve

CACHE = Path("data")
_log = logging.getLogger("cmip6_download")


def main(first_only: bool) -> None:
    plan = cmip6.plan_requests(cmip6.fetch_constraints())
    if first_only:
        plan = sorted(plan, key=lambda request: len(request["model"]))[:1]
    directory = cmip6.raw_directory(CACHE)

    client = make_client(CDS_URL)
    for request in plan:
        target = directory / cmip6.target_name(request)
        for attempt in range(12):  # a rejected request is retried after a pause, up to an hour
            try:
                retrieve(client, cmip6.DATASET, request, target)
                break
            except SourceFailed as error:
                _log.warning("retry %d in 5 min: %s", attempt + 1, str(error)[-200:])
                time.sleep(300)
        else:
            raise SourceFailed(f"{target.name} did not go through")
        _log.info("done %s (%d models)", target.name, len(request["model"]))
        print("ok", target.name, flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main("--first" in sys.argv)
