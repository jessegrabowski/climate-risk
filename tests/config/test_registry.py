import pytest

from climate_risk.config.registry import load_place, read_place, resolve_isos
from climate_risk.config.schema import CountryConfig, EventFilters, RegionConfig


@pytest.fixture
def config_root(tmp_path):
    """Return a callable writing a place file, and the root the registry should read it from."""

    def write(subdirectory, stem, body):
        directory = tmp_path / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{stem}.toml").write_text(body)
        return tmp_path

    return write


def test_a_country_takes_the_defaults_it_does_not_state(config_root):
    """A place file states what differs; everything else has to come from the schema."""
    root = config_root("places", "zmb", 'iso3 = "ZMB"\nname = "Zambia"\n')

    place = load_place("zmb", root=root)

    assert place == CountryConfig(iso3="ZMB", name="Zambia")
    assert place.geometry.projected_crs == "EPSG:3395"


def test_a_sub_table_becomes_the_object_the_schema_declares(config_root):
    """Left as a dict it would reach the loaders as raw TOML and fail at the first attribute."""
    root = config_root(
        "places",
        "lao",
        'iso3 = "LAO"\nname = "Lao PDR"\n\n[geometry]\nprojected_crs = "EPSG:32648"\n\n[events]\nstart_year = 1970\nend_year = 2024\n',
    )

    place = load_place("lao", root=root)

    assert place.geometry.projected_crs == "EPSG:32648"
    assert place.events == EventFilters(start_year=1970, end_year=2024)


def test_event_overrides_arrive_as_coordinate_pairs(config_root):
    """TOML gives arrays; the loaders index them as longitude and latitude."""
    root = config_root(
        "places",
        "lao",
        'iso3 = "LAO"\nname = "Lao PDR"\n\n[event_location_overrides]\n"1971-0048-LAO" = [102.6331, 17.9757]\n',
    )

    place = load_place("lao", root=root)

    assert isinstance(place, CountryConfig)
    assert place.event_location_overrides == {"1971-0048-LAO": (102.6331, 17.9757)}


BOUNDARY_BLOCK = (
    '[boundary]\nmember = "lao_admin2.shp"\n'
    'url = "https://example.org/lao.zip"\nfilename = "lao.zip"\n'
    'license = "CC BY 3.0 IGO"\ncitation = "NGD"\nretrieved = "2026-08-08"\n'
)


def test_a_boundary_block_becomes_a_fetchable_archive(config_root):
    """It reuses `DataSource`, so a place's own boundary goes through the same fetch as everything else."""
    root = config_root("places", "lao", 'iso3 = "LAO"\nname = "Lao PDR"\n\n' + BOUNDARY_BLOCK)

    place = load_place("lao", root=root)

    assert isinstance(place, CountryConfig)
    assert place.boundary is not None
    assert place.boundary.source.filename == "lao.zip"
    assert place.boundary.member == "lao_admin2.shp"


def test_a_boundary_that_does_not_name_its_layer_is_an_error(config_root):
    """The archive holds several layers, so guessing one would read a different admin level."""
    without_member = BOUNDARY_BLOCK.replace('member = "lao_admin2.shp"\n', "")
    root = config_root("places", "lao", 'iso3 = "LAO"\nname = "Lao PDR"\n\n' + without_member)

    with pytest.raises(ValueError, match="must set member"):
        load_place("lao", root=root)


def test_the_directory_decides_whether_a_file_is_a_country_or_a_region(config_root):
    """Both schemas would accept a `name`, so nothing else distinguishes them."""
    config_root("places", "lao", 'iso3 = "LAO"\nname = "Lao PDR"\n')
    root = config_root("regions", "sea", 'key = "sea"\nname = "Southeast Asia"\nmembers = ["LAO", "THA"]\n')

    assert isinstance(load_place("lao", root=root), CountryConfig)
    assert isinstance(load_place("sea", root=root), RegionConfig)


def test_a_key_the_schema_does_not_define_is_an_error(config_root):
    """A typo in a place file would otherwise be ignored and the default used instead."""
    root = config_root("places", "lao", 'iso3 = "LAO"\nname = "Lao PDR"\nisland_nation = true\n')

    with pytest.raises(ValueError, match="does not match the CountryConfig schema"):
        load_place("lao", root=root)


def test_a_value_the_schema_rejects_names_the_file_that_holds_it(config_root):
    """The schema only sees the value, and a bare complaint about it does not say where to go fix it."""
    root = config_root("places", "lao", 'iso3 = "Lao"\nname = "Lao PDR"\n')

    with pytest.raises(ValueError, match=r"lao\.toml does not match the CountryConfig schema"):
        load_place("lao", root=root)


def test_a_file_that_will_not_parse_names_itself(config_root):
    root = config_root("places", "lao", 'iso3 = "LAO"\nname = Lao PDR\n')

    with pytest.raises(ValueError, match="not valid TOML"):
        load_place("lao", root=root)


def test_an_unknown_place_lists_the_ones_that_exist(config_root):
    """The message is how a caller discovers what they can ask for."""
    root = config_root("places", "lao", 'iso3 = "LAO"\nname = "Lao PDR"\n')

    with pytest.raises(ValueError, match="Known places: lao"):
        load_place("atlantis", root=root)


def test_a_country_resolves_to_itself_and_a_region_to_its_members():
    """The one replacement for the region dispatch the loaders used to carry."""
    assert resolve_isos(CountryConfig(iso3="LAO", name="Lao PDR")) == ("LAO",)
    assert resolve_isos(RegionConfig(key="sea", name="SEA", members=("LAO", "THA"))) == ("LAO", "THA")


def test_reading_a_region_file_directly_gives_a_region(tmp_path):
    """`read_place` is what the directory-wide conformance test walks, so it must not need a key."""
    path = tmp_path / "regions" / "sea.toml"
    path.parent.mkdir(parents=True)
    path.write_text('key = "sea"\nname = "Southeast Asia"\nmembers = ["LAO", "THA"]\n')

    assert read_place(path) == RegionConfig(key="sea", name="Southeast Asia", members=("LAO", "THA"))
