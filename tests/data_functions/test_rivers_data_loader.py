import pytest

from climate_risk import load_rivers_data
from climate_risk.data_functions import rivers_data_loader
from climate_risk.data_functions.rivers_data_loader import transform_rivers
from tests.conftest import NetworkAccessError, toy_rivers


@pytest.fixture
def warm_cache(write_rivers_cache):
    """Both processed files, holding different rivers."""
    cache_dir = write_rivers_cache(toy_rivers().iloc[:1])
    write_rivers_cache(toy_rivers().iloc[:2], include_medium=True)

    return cache_dir


@pytest.mark.parametrize(("cutoff", "kept"), [(5, [4]), (6, [4, 5])])
def test_the_discharge_class_cutoff_excludes_its_own_class(cutoff, kept):
    """The warm-cache tests read back what a fixture wrote, so this is where the filter is tested."""
    kept_rivers = transform_rivers(toy_rivers(), cutoff)

    assert kept_rivers["ORD_FLOW"].tolist() == kept


def test_big_rivers_are_read_by_default(warm_cache):
    rivers = load_rivers_data(warm_cache)

    assert rivers["ORD_FLOW"].tolist() == [4]


def test_include_medium_reads_a_separate_cache_entry(warm_cache):
    rivers = load_rivers_data(warm_cache, include_medium=True)

    assert rivers["ORD_FLOW"].tolist() == [4, 5]


def test_processed_cache_alone_does_not_download(write_rivers_cache):
    cache_dir = write_rivers_cache(toy_rivers().iloc[:1])

    assert load_rivers_data(cache_dir)["ORD_FLOW"].tolist() == [4]


def test_cold_cache_reaches_for_the_archive(tmp_path):
    with pytest.raises(NetworkAccessError):
        load_rivers_data(tmp_path)


def test_editing_the_filter_turns_the_cache_over(tmp_path, monkeypatch):
    """The entry keys on how the network was read as well as on the cutoff, so a changed
    `transform_rivers` builds a second entry rather than being served the first one's rivers.
    """
    shapefile = tmp_path / "network.shp"
    toy_rivers().to_file(shapefile)
    monkeypatch.setattr(rivers_data_loader, "_extract_rivers", lambda cache_dir: shapefile)

    first = load_rivers_data(tmp_path)

    monkeypatch.setattr(rivers_data_loader, "transform_rivers", lambda rivers, cutoff: rivers.iloc[:0])
    second = load_rivers_data(tmp_path)

    assert first["ORD_FLOW"].tolist() == [4]
    assert second.empty, "the changed filter was served the first entry's rivers"
    assert len(list((tmp_path / "rivers").glob("rivers__*.parquet"))) == 2
