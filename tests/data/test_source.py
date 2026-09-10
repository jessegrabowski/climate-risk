import re

import pytest
import requests

from climate_risk.config.registry import CONFIG_ROOT, read_place
from climate_risk.config.schema import CountryConfig
from climate_risk.data.co2 import CO2
from climate_risk.data.fetch import USER_AGENT
from climate_risk.data.gadm import GADM
from climate_risk.data.geonames import COUNTRY_INFO, country_dump
from climate_risk.data.gpcc import FULL_DATA, MONITORING
from climate_risk.data.hadcrut import HADCRUT
from climate_risk.data.ocean_heat import OCEAN_HEAT
from climate_risk.data.source import DataSource, ManualSource, ShapefileArchive, VendoredSource
from climate_risk.data_functions.emdat_processing import EMDAT
from climate_risk.data_functions.rivers_data_loader import RIVERS
from climate_risk.data_functions.shapefiles_data_loader import COASTLINE, WORLD

FIELDS = {
    "url": "https://example.org/co2.csv",
    "filename": "noaa_co2.csv",
    "license": "public domain",
    "citation": "NOAA GML",
    "retrieved": "2026-08-03",
}


def source(**overrides) -> DataSource:
    return DataSource(**(FIELDS | overrides))


def test_a_source_is_stored_under_its_filename(tmp_path):
    assert source().path(tmp_path) == tmp_path / "noaa_co2.csv"


@pytest.mark.parametrize("filename", ["../escape.csv", "nested/file.csv", "/absolute.csv", ""], ids=repr)
def test_a_filename_carrying_a_path_is_rejected(filename):
    """`path()` joins onto the cache directory, so a directory component writes outside it."""
    with pytest.raises(ValueError, match="bare name"):
        source(filename=filename)


@pytest.mark.parametrize("url", ["ftp://example.org/f.csv", "example.org/f.csv", "file:///etc/passwd"], ids=repr)
def test_a_non_http_url_is_rejected(url):
    with pytest.raises(ValueError, match="http"):
        source(url=url)


def test_an_archive_layer_unpacks_under_the_directory_it_is_given(tmp_path):
    assert ShapefileArchive(source(), "GSHHS_shp/f").extracted_path(tmp_path) == tmp_path / "GSHHS_shp" / "f"


@pytest.mark.parametrize("member", ["../escape.shp", "/absolute.shp", "nested/../../out.shp", ""], ids=repr)
def test_an_archive_layer_outside_the_archive_is_rejected(member):
    """`extracted_path` joins onto the cache directory, so an escaping member reads outside it."""
    with pytest.raises(ValueError, match="member must"):
        ShapefileArchive(source(), member)


MANUAL_FIELDS = {
    "filename": "gadm_410.gpkg",
    "homepage": "https://gadm.org/download_world.html",
    "license": "non-commercial use only",
    "citation": "GADM 4.1",
    "retrieved": "2026-08-09",
}


def manual(**overrides) -> ManualSource:
    return ManualSource(**(MANUAL_FIELDS | overrides))


@pytest.mark.parametrize("filename", ["../escape.gpkg", "nested/f.gpkg", "/absolute.gpkg", ""], ids=repr)
def test_a_manual_filename_carrying_a_path_is_rejected(filename):
    """`path()` joins onto the cache directory, so a directory component reads outside it."""
    with pytest.raises(ValueError, match="bare name"):
        manual(filename=filename)


@pytest.mark.parametrize("homepage", ["ftp://gadm.org", "gadm.org", ""], ids=repr)
def test_a_manual_homepage_that_is_not_a_web_page_is_rejected(homepage):
    """It goes into an error message a user is expected to follow."""
    with pytest.raises(ValueError, match="homepage must be http"):
        manual(homepage=homepage)


def test_a_manual_source_reports_its_license_when_the_file_is_missing(tmp_path):
    """A user hitting this has to know the terms before going to fetch the file."""
    with pytest.raises(NotImplementedError, match="non-commercial use only"):
        manual().require(tmp_path)


# Every file a person has to place in the cache themselves. A loader that raises for a missing
# hand-placed file belongs here, so the terms it is held under are recorded in one place.
MANUAL_SOURCES = {"EMDAT": EMDAT, "GADM": GADM}


@pytest.mark.parametrize("declared", MANUAL_SOURCES.values(), ids=MANUAL_SOURCES.keys())
def test_every_manual_source_says_where_to_get_it_and_on_what_terms(declared):
    """These declarations are the only record of terms binding data the repo cannot ship."""
    assert declared.homepage.startswith("https://")
    assert declared.license.strip()
    assert declared.citation.strip()


@pytest.mark.parametrize("declared", MANUAL_SOURCES.values(), ids=MANUAL_SOURCES.keys())
def test_a_manual_source_is_never_fetchable(declared):
    """Both forbid redistribution or automated download; a url would let `fetch` take one anyway."""
    assert not hasattr(declared, "url")


VENDORED_FIELDS = {
    "filename": "ipcc_ar6_syr_csb2_fig1a.xlsx",
    "homepage": "https://doi.org/10.7927/baxv-nj53",
    "license": "Attribution 4.0 International (CC BY 4.0)",
    "citation": "IPCC, 2024",
    "retrieved": "2026-09-09",
}


def vendored(**overrides) -> VendoredSource:
    return VendoredSource(**(VENDORED_FIELDS | overrides))


@pytest.mark.parametrize("filename", ["../escape.xlsx", "nested/f.xlsx", "/absolute.xlsx", ""], ids=repr)
def test_a_vendored_filename_carrying_a_path_is_rejected(filename):
    """`path()` joins onto the packaged directory, so a directory component reads outside it."""
    with pytest.raises(ValueError, match="bare name"):
        vendored(filename=filename)


@pytest.mark.parametrize("homepage", ["ftp://doi.org", "doi.org", ""], ids=repr)
def test_a_vendored_homepage_that_is_not_a_web_page_is_rejected(homepage):
    """It is the only record of where a redistributed file came from."""
    with pytest.raises(ValueError, match="homepage must be http"):
        vendored(homepage=homepage)


def test_a_vendored_source_is_never_fetchable():
    """It ships in the wheel, so a url would put a dead publisher into the reachability check."""
    assert not hasattr(vendored(), "url")


def test_a_vendored_source_resolves_inside_the_package():
    """It takes no cache directory, so the packaged tree is the only place it can name."""
    assert vendored().path().parent.name == "vendored"


def test_an_unparseable_retrieved_date_is_rejected():
    """Sources are import-time literals, so the error must name the one that failed."""
    with pytest.raises(ValueError, match=re.escape("noaa_co2.csv: retrieved must be an ISO date")):
        source(retrieved="03-08-2026")


def place_boundaries() -> dict[str, DataSource]:
    """Every boundary a shipped place declares, so moving a source into TOML does not lose it here."""
    places = (read_place(path) for path in sorted(CONFIG_ROOT.glob("*/*.toml")))

    return {
        f"{place.iso3}_BOUNDARY": place.boundary.source
        for place in places
        if isinstance(place, CountryConfig) and place.boundary is not None
    }


# Every source the library downloads. A new one declared in code is added here; one declared in a
# place file is picked up automatically. The monitoring archives all share one URL pattern, so its
# ends are checked rather than every month.
SOURCES = (
    {
        "CO2": CO2,
        "OCEAN_HEAT": OCEAN_HEAT,
        "HADCRUT": HADCRUT,
        "WORLD": WORLD,
        "COASTLINE": COASTLINE,
        "RIVERS": RIVERS,
        "GEONAMES_COUNTRY_INFO": COUNTRY_INFO,
        # Every country dump shares one URL pattern, so one stands for all of them.
        "GEONAMES_DUMP": country_dump("PH"),
    }
    | {archive.filename: archive for archive in FULL_DATA.sources}
    | {archive.filename: archive for archive in (MONITORING.sources[0], MONITORING.sources[-1])}
    | place_boundaries()
)


def test_a_boundary_declared_in_a_place_file_reaches_the_check():
    """The network check walks `SOURCES`; a place file is invisible to it unless collected here.

    Runs unmarked so it fails when the collection breaks, rather than only under `--run-network`.
    """
    assert "LAO_BOUNDARY" in SOURCES


@pytest.mark.network
@pytest.mark.parametrize("declared", SOURCES.values(), ids=SOURCES.keys())
def test_every_declared_source_is_still_published(declared):
    """Upstream reorganizes without notice, and a dead URL surfaces as a failed run months later."""
    # Hosts vary their answer by agent, so the check has to send the one fetch will send.
    headers = {"User-Agent": USER_AGENT}
    response = requests.head(declared.url, timeout=30, allow_redirects=True, headers=headers)
    if response.status_code >= 400:
        # Some hosts refuse HEAD; ask for one byte instead of downloading the archive.
        response = requests.get(declared.url, timeout=30, headers=headers | {"Range": "bytes=0-0"}, stream=True)
        response.close()

    assert response.status_code < 400
