"""
Download climate covariates for several countries into ``data/raw/covariates/<ISO3>/<source>/``.

    pixi run python experiments/covariates/download_covariates.py roni
    pixi run python experiments/covariates/download_covariates.py era5land era5 --countries NPL ZMB
    pixi run python experiments/covariates/download_covariates.py chirps glofas

Every request is cut to the country's box (boundary + 0.5 degrees); nothing global is stored.
Safe to stop and rerun: every file already on disk is skipped.
"""

import argparse
import logging

from concurrent.futures import ThreadPoolExecutor

from common import COUNTRIES, FIRST_YEAR, LAST_YEAR, country_box, raw_dir, record_source

from climate_risk.data import chirps, copernicus, roni

_log = logging.getLogger("download_covariates")
GLOFAS_FIRST_YEAR = 1980


def do_roni(countries: list[str]) -> None:
    # RONI is one world series; it is stored once, under the first country's folder, and linked in the others' manifests.
    path = roni.download_roni(raw_dir("NPL") / "roni")
    for iso in countries:
        record_source(
            iso,
            "roni",
            dataset="Relative Oceanic Nino Index (RONI), seasonal",
            version="CPC RONI.ascii.txt",
            product="3-month seasonal anomalies (world series, joined by year)",
            url=roni.RONI_URL,
            licence=roni.RONI_LICENCE,
            file=str(path.relative_to(raw_dir("NPL").parent)),
            note="Recent values are revised for up to two months.",
        )


def do_chirps(countries: list[str], start: int, end: int, workers: int) -> None:
    """Daily CHIRPS v2.0 from the IRI Data Library, one request per country and year."""
    boxes = {iso: country_box(iso) for iso in countries}
    jobs = [(iso, year) for year in range(start, end + 1) for iso in countries]

    def fetch(job: tuple[str, int]) -> None:
        iso, year = job
        chirps.download_iri_year(year, boxes[iso], raw_dir(iso) / "chirps")
        _log.info("CHIRPS v2 %s %d done", iso, year)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, jobs))
    for iso in countries:
        record_source(
            iso,
            "chirps",
            dataset="CHIRPS",
            version="v2.0 (v3.0 host blocked the download; see summary)",
            product="daily improved, 0.05 degrees, via IRI Data Library, clipped server-side to box",
            url=chirps.IRI_ROOT,
            licence=chirps.IRI_LICENCE,
            box_west_south_east_north=boxes[iso],
        )


def do_copernicus(source: str, iso: str, start: int, end: int, chunk: int) -> None:
    box = country_box(iso)
    first = GLOFAS_FIRST_YEAR if source == "glofas" else start
    chunks = [list(range(y, min(y + chunk, end + 1))) for y in range(first, end + 1, chunk)]
    if source == "glofas":
        client = copernicus.make_client(copernicus.EWDS_URL)
        for years in chunks:
            target = raw_dir(iso) / "glofas" / f"glofas_v5_{years[0]}_{years[-1]}.nc"
            copernicus.retrieve(client, copernicus.GLOFAS, copernicus.glofas_request(years, box), target)
            _log.info("%s GloFAS %d-%d done", iso, years[0], years[-1])
        record_source(
            iso,
            "glofas",
            dataset=copernicus.GLOFAS,
            version="version_5_0",
            product="consolidated, average_river_discharge_in_the_last_24_hours (time_mean)",
            url="https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical",
            licence="Copernicus licence (CEMS-FLOODS / GloFAS licence accepted 2026-11-21)",
            box_west_south_east_north=box,
        )
        return

    client = copernicus.make_client(copernicus.CDS_URL)
    dataset, build = {
        "era5land": (copernicus.ERA5_LAND, copernicus.era5_land_request),
        "era5": (copernicus.ERA5_SINGLE, copernicus.era5_request),
    }[source]
    for years in chunks:
        target = raw_dir(iso) / source / f"{source}_monthly_{years[0]}_{years[-1]}.nc"
        copernicus.retrieve(client, dataset, build(years, box), target)
        _log.info("%s %s %d-%d done", iso, source, years[0], years[-1])
    record_source(
        iso,
        source,
        dataset=dataset,
        version="CDS catalogue (ERA5 family)",
        product="monthly averaged reanalysis, 00:00",
        url=f"https://cds.climate.copernicus.eu/datasets/{dataset}",
        licence="CC-BY-4.0 (Copernicus licence accepted 2026-11-21)",
        box_west_south_east_north=box,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", choices=["roni", "chirps", "era5land", "era5", "glofas"])
    parser.add_argument("--countries", nargs="+", default=COUNTRIES)
    parser.add_argument("--start", type=int, default=FIRST_YEAR)
    parser.add_argument("--end", type=int, default=LAST_YEAR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=10, help="years per ERA5/ERA5-Land request")
    parser.add_argument("--glofas-chunk", type=int, default=5, help="years per GloFAS request")
    args = parser.parse_args()
    for name in args.sources:
        if name == "roni":
            do_roni(args.countries)
        elif name == "chirps":
            do_chirps(args.countries, args.start, args.end, args.workers)
        else:
            for iso in args.countries:
                do_copernicus(name, iso, args.start, args.end, args.glofas_chunk if name == "glofas" else args.chunk)
