import csv
import re

from collections import defaultdict

import geopandas as gpd
import pytest

from climate_risk.data import place_names
from climate_risk.data.place_names import (
    CONTAINED_BY,
    CORRECTED,
    DERIVED,
    INFERRED,
    LOCATED,
    NAME_CORRECTIONS,
    NAMED,
    NAMES_NO_UNIT,
    Placement,
    Unit,
    match_key,
    name_shapes,
    names_no_unit,
    nearest_name,
    read_gazetteer,
    read_name_corrections,
    repair_mojibake,
    resolve_event_places,
    resolve_place,
    successor_state,
)


@pytest.mark.parametrize(
    ("written", "published"),
    [
        ("Kalin-Aapayo province", "Kalin Aapayo"),
        ("Ron Phibun districts", "Ron Phibun"),
        ("Luzon Isl.", "Luzon"),
        ("Đà Nẵng city", "Da Nang"),
    ],
    ids=["province", "plural district", "abbreviated island", "transliterated"],
)
def test_a_written_mention_and_a_published_name_reach_the_same_key(written, published):
    """EM-DAT says what kind of unit it means and GADM does not, so the noun has to come off before
    the two can meet."""
    assert match_key(written) == match_key(published)


def test_a_name_that_is_only_a_unit_noun_reaches_nothing():
    """`districts` on its own is what a trailing separator leaves behind. Keyed on the empty string
    it would collide with every other such fragment and match whatever came first."""
    assert match_key("districts") == ""


def test_units_are_indexed_under_every_name_they_are_published_under(write_gadm_cache):
    """GADM carries a variant name for many units, and EM-DAT writes whichever it has."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert Unit("LAO.1_1", 1, None) in gazetteer.names[match_key("Attapu")]
    assert gazetteer.names[match_key("Attopeu")] == {Unit("LAO.1_1", 1, None)}, "the variant name reaches the same unit"
    assert gazetteer.names[match_key("Houei Sai")] == {Unit("LAO.2.1_1", 2, "LAO.2_1")}, "one of two piped variants"


def test_every_level_the_archive_publishes_is_indexed(write_gadm_cache):
    """A location string names provinces, districts and villages in one breath, and stopping at
    adm2 loses the finest of them: adm3 and adm4 carry a fifth of everything that resolves."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert Unit("LAO.1.1.1_1", 3, "LAO.1.1_1") in gazetteer.names[match_key("Ban Mai")]


def test_a_district_knows_the_province_holding_it(write_gadm_cache):
    """Parentage decides which of two same-named units a container selects, so it has to come from
    the row rather than from the shape of the identifier: 37 GADM rows do not nest."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert {unit.parent for unit in gazetteer.names[match_key("Houayxay")]} == {"LAO.2_1"}


def test_a_repeated_name_reaches_every_unit_that_carries_it(write_gadm_cache):
    """A name is not an identifier: Attapu is a province and, elsewhere, a district of Bokeo.
    Keeping whichever unit was read first would silently pick one and look decisive about it."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Attapu", None, gazetteer) == {"LAO.1_1", "LAO.2.2_1"}


def test_a_container_chooses_between_units_sharing_a_name(write_gadm_cache):
    """The container is what the location prose buys over a bare name match. Without it the choice
    between same-named units is a similarity threshold, which is the approach this project rejected."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Attapu", "Bokeo", gazetteer) == {"LAO.2.2_1"}, "the district inside Bokeo"


def test_a_container_naming_nothing_leaves_the_candidates_alone(write_gadm_cache):
    """The prose routinely names a region GADM does not model — `Calabarzon`, `Bicol` — and an
    unrecognized container is missing information, not evidence against the candidates."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Sanamxay", "Some Region GADM Never Heard Of", gazetteer) == {"LAO.1.1_1"}


def test_a_container_holding_none_of_the_candidates_leaves_them_alone(write_gadm_cache):
    """EM-DAT's containers are sometimes wrong or coarser than GADM's hierarchy. Dropping every
    candidate would turn a mistaken container into a lost place."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Sanamxay", "Bokeo", gazetteer) == {"LAO.1.1_1"}


def test_a_name_no_unit_carries_reaches_nothing(write_gadm_cache):
    """Two fifths of written places name no GADM unit at all, and the caller has to tell that from
    a match."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Calabarzon", None, gazetteer) == set()


def test_a_seat_named_after_its_district_resolves_to_the_district(write_gadm_cache):
    """Two thirds of ambiguous mentions are one nesting chain, not two places. The district is the
    whole of what the mention can mean, and returning the seat beside it invents a choice."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Samakhixay", None, gazetteer) == {"LAO.1.2_1"}


def test_a_container_reaches_past_the_level_directly_below_it(write_gadm_cache):
    """Prose names the province and then a village, skipping the district between them, so a
    container matched against the immediate parent alone would discard the village."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Ban Mai", "Attapu", gazetteer) == {"LAO.1.1.1_1"}


def test_an_unambiguous_place_narrows_the_ambiguous_ones_beside_it(write_gadm_cache):
    """An event's places share a footprint. Resolving each alone throws that away and leaves
    Attapu split between a province and a district in the province the event is plainly in."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_event_places([("Houayxay", None), ("Attapu", None)], gazetteer) == [
        Placement({"LAO.2.1_1"}, NAMED),
        Placement({"LAO.2.2_1"}, NAMED),
    ]


def test_an_event_pinning_nothing_leaves_its_places_alone(write_gadm_cache):
    """Half the ambiguous mentions sit in events where nothing resolves uniquely, and narrowing
    against an empty envelope would drop every candidate."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_event_places([("Attapu", None), ("Calabarzon", None)], gazetteer) == [
        Placement({"LAO.1_1", "LAO.2.2_1"}, NAMED),
        Placement(set(), NAMED),
    ]


def test_a_country_the_archive_does_not_cover_reads_back_empty(write_gadm_cache):
    """EM-DAT codes states GADM never modeled — Serbia and Montenegro, the USSR — and a caller
    walking every ISO in the workbook reaches them."""
    gazetteer = read_gazetteer("SCG", write_gadm_cache())

    assert gazetteer.names == {}
    assert resolve_place("Beograd", None, gazetteer) == set()


def test_a_unit_the_archive_nests_inside_itself_is_read_as_containing_nothing(write_gadm_cache):
    """GADM writes an unnamed Ukrainian unit as `?` at two levels, which reads as its own container
    and makes every walk over parentage non-terminating. The reader is where that has to stop."""
    gazetteer = read_gazetteer("UKR", write_gadm_cache())

    assert gazetteer.parent_of["?"] is None
    assert gazetteer.ancestry("?") == {"?"}


@pytest.mark.parametrize(
    ("written", "published"),
    [
        ("Bengkulu area", "Bengkulu"),
        ("Cagayan Prov.", "Cagayan"),
        ("City of Bandung", "Bandung"),
        ("Near Villavicencio", "Villavicencio"),
        ("Coast of Zamboanga", "Zamboanga"),
    ],
    ids=["area", "abbreviated province", "city of", "near", "coast of"],
)
def test_a_decorated_mention_reaches_the_published_name(written, published):
    """A fifth of the names nothing matches are a published name with a word like this attached."""
    assert match_key(written) == match_key(published)


def test_a_conjunction_reaches_both_places_it_joins(write_gadm_cache):
    """EM-DAT writes a run of provinces joined by `and`, and refusing to split leaves every one of
    them unplaced."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Sanamxay and Samakhixay", None, gazetteer) == {"LAO.1.1_1", "LAO.1.2_1"}


def test_a_unit_whose_own_name_carries_a_conjunction_survives(write_gadm_cache):
    """Splitting `Newfoundland and Labrador` destroys the province it names, so the whole string
    has to be tried before its parts."""
    gazetteer = read_gazetteer("CAN", write_gadm_cache())

    assert resolve_place("Newfoundland and Labrador", None, gazetteer) == {"CAN.5_1"}


def test_a_conjunction_keeps_whichever_parts_name_something(write_gadm_cache):
    """EM-DAT joins a place it spelled well to one it spelled badly. Discarding both because one
    failed throws away geography the prose plainly gives."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Nowhere At All and Sanamxay", None, gazetteer) == {"LAO.1.1_1"}


def test_a_place_named_between_two_others_is_in_neither(write_gadm_cache):
    """`Between Java and Bali` is a stretch of sea named by its shores. Splitting it puts the event
    on whichever shore resolves, which is a coastline away from where it happened."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Between Bokeo and Sanamxay", None, gazetteer) == set()


def test_a_place_naming_nothing_falls_back_to_the_container_it_was_written_in(write_gadm_cache):
    """A third of the events still unplaced name a container that resolves — `Pesisir Selaten (West
    Sumatra province)` gives the province even where the district is unrecognizable."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_event_places([("Nowhere At All", "Bokeo")], gazetteer) == [Placement({"LAO.2_1"}, CONTAINED_BY)]


def test_a_container_standing_in_for_a_place_is_marked_as_coarser(write_gadm_cache):
    """The province is not what the prose named, and a caller aggregating footprints has to be able
    to tell a district it asked for from the province it settled for."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    named, fallen_back = resolve_event_places([("Sanamxay", None), ("Nowhere At All", "Bokeo")], gazetteer)

    assert (named.how, fallen_back.how) == (NAMED, CONTAINED_BY)


def test_a_container_standing_in_for_a_place_is_narrowed_like_any_other(write_gadm_cache):
    """`Attapu` names a province and a district elsewhere. Falling back to it without the narrowing
    a named place gets would hand back both, when the event has already pinned which one."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    _, fallen_back = resolve_event_places([("Houayxay", None), ("Nowhere At All", "Attapu")], gazetteer)

    assert fallen_back == Placement({"LAO.2.2_1"}, CONTAINED_BY), "the district inside the pinned province"


def test_a_unit_below_a_province_names_the_province_as_its_outermost(write_gadm_cache):
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert gazetteer.top_container("LAO.1.1.1_1") == "LAO.1_1"


def test_the_outermost_container_of_the_placeholder_unit_is_itself(write_gadm_cache):
    """Walking outwards from a unit the archive nests inside itself must still reach a top."""
    gazetteer = read_gazetteer("UKR", write_gadm_cache())

    assert gazetteer.top_container("?") == "?"


def test_a_misspelling_reaches_the_name_it_misspells(write_gadm_cache):
    """Misspellings are the largest thing left that no gazetteer places: `Barrranquilla`,
    `Marizales`, `Santa Rose de Cabal`."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert nearest_name("Sanamxai", gazetteer, name_shapes(gazetteer)) == "sanamxay"


def test_a_name_equally_close_to_two_published_names_reaches_neither(write_gadm_cache):
    """Two names one edit away is not a near miss, it is a choice. `Bocolod` sits one edit from
    both Bacolod the city and Boclod the barangay, and picking either looks equally confident."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert nearest_name("Attopu", gazetteer, name_shapes(gazetteer)) is None, "one edit from both Attapu and Attopeu"


def test_a_name_too_short_to_misspell_is_refused(write_gadm_cache):
    """One edit in a five-letter name changes too much of it to call the rest a match."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert nearest_name("Bokea", gazetteer, name_shapes(gazetteer)) is None


def test_a_physical_feature_is_never_taken_for_a_unit_it_resembles(write_gadm_cache):
    """A bay is not an administrative unit however close its name sits to one. Units named after a
    feature still match exactly; only the approximate lookup refuses them."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert nearest_name("Nambok", gazetteer, name_shapes(gazetteer)) == "nambak", (
        "the same unit is reachable by a plain misspelling"
    )
    assert nearest_name("Nam Bay", gazetteer, name_shapes(gazetteer)) is None


def test_a_name_that_already_matches_is_not_approximated(write_gadm_cache):
    """An exact match is the answer, and offering a near one beside it would let a misspelling of
    some other place override a name GADM publishes."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert nearest_name("Sanamxay", gazetteer, name_shapes(gazetteer)) is None


def test_a_checked_misspelling_reaches_the_unit_it_stands_for(write_gadm_cache):
    """Approximate matching proposes corrections; only the ones written into the table are applied."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())._replace(corrections={"sanamxai": ("sanamxay",)})

    assert resolve_event_places([("Sanamxai", None)], gazetteer) == [Placement({"LAO.1.1_1"}, CORRECTED)]


def test_a_misspelling_nobody_checked_is_left_unplaced(write_gadm_cache):
    """One edit from a published name is a proposal, not a correction: `Lynmouth` sits one edit from
    Lynemouth, four hundred miles away."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_event_places([("Sanamxai", None)], gazetteer) == [Placement(set(), NAMED)]


def test_a_name_two_edits_away_is_not_a_misspelling_of_it(write_gadm_cache):
    """One slip is a misspelling. At two, `Sanamxay` and a different district are equally far, and
    the match is a guess dressed as a correction."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert nearest_name("Sanamxii", gazetteer, name_shapes(gazetteer)) is None


def test_a_point_is_preferred_to_the_container_a_place_was_written_in(write_gadm_cache):
    """The container claims every unit inside it. A point names one, so it wins wherever it exists:
    over half the places that fall back to a container have one."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Nowhere At All", "Bokeo")], gazetteer, located={"Nowhere At All": "LAO.2.1_1"})

    assert placed == Placement({"LAO.2.1_1"}, LOCATED), "the district, not the whole of Bokeo"


def test_a_place_its_own_name_reaches_ignores_the_point_offered_for_it(write_gadm_cache):
    """A name GADM publishes is better evidence than a coordinate somebody geocoded from it."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Sanamxay", None)], gazetteer, located={"Sanamxay": "LAO.2.1_1"})

    assert placed == Placement({"LAO.1.1_1"}, NAMED)


def test_a_point_chooses_between_the_units_one_name_reaches(write_gadm_cache):
    """`Attapu` is both a province and a district of another province. Keeping both puts the event
    in a place it never happened, and the point says which one the writer meant."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Attapu", None)], gazetteer, located={"Attapu": "LAO.2.2_1"})

    assert placed == Placement({"LAO.2.2_1"}, NAMED), "the district, not the province as well"


def test_a_point_inside_a_candidate_chooses_it_from_any_depth(write_gadm_cache):
    """A geocoder answers with the finest unit holding the point, which is below the level the
    written name reaches. Matching the two literally would settle nothing."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Attapu", None)], gazetteer, located={"Attapu": "LAO.1.1.1_1"})

    assert placed == Placement({"LAO.1_1"}, NAMED), "the province holding the village the point fell in"


def test_a_point_outside_every_candidate_settles_nothing(write_gadm_cache):
    """Geocoders are wrong often enough that a point landing somewhere else is evidence of nothing.
    Discarding candidates on it would trade an ambiguous placement for a confident wrong one."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Attapu", None)], gazetteer, located={"Attapu": "LAO.2.1_1"})

    assert placed == Placement({"LAO.1_1", "LAO.2.2_1"}, NAMED)


def test_a_name_a_point_settled_narrows_the_places_beside_it(write_gadm_cache):
    """A settled name is as good as one that was never ambiguous, so it has to reach `pinned` before
    the event's other places are read against it."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    _, fallen_back = resolve_event_places(
        [("Attapu", None), ("Nowhere At All", "Attapu")], gazetteer, located={"Attapu": "LAO.1_1"}
    )

    assert fallen_back == Placement({"LAO.1_1"}, CONTAINED_BY), "the province the point settled on, not both"


def test_a_correction_is_read_only_for_the_country_that_declares_it(write_gadm_cache, tmp_path):
    """The same written name means different places in different countries, and a table keyed only
    on the name would carry one country's correction into every other."""
    table = tmp_path / "corrections.csv"
    table.write_text("iso,written,corrected\nLAO,Sanamxai,sanamxay\nZMB,Sanamxai,kabwe\n", encoding="utf-8")

    assert read_name_corrections("LAO", path=table) == {"sanamxai": ("sanamxay",)}


def test_a_correction_the_table_leaves_blank_is_not_applied(write_gadm_cache, tmp_path):
    """A row can record that a candidate was checked and rejected; that is not a correction."""
    table = tmp_path / "corrections.csv"
    table.write_text("iso,written,corrected\nLAO,Lynmouth,\n", encoding="utf-8")

    assert read_name_corrections("LAO", path=table) == {}


def test_no_country_corrects_one_name_to_two_different_units(write_gadm_cache):
    """`Badakhstan` and `Badakhstan province` reduce to one key, which is fine while they agree.
    Two rows disagreeing would resolve on file order, and the table is curated by hand."""
    targets: dict[tuple[str, str], set[str]] = defaultdict(set)
    with NAME_CORRECTIONS.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            targets[(row["iso"], match_key(row["written"]))].add(row["corrected"])

    conflicting = {key: sorted(published) for key, published in targets.items() if len(published) > 1}

    assert not conflicting, f"one name corrected two ways: {sorted(conflicting.items())[:5]}"


def test_a_state_with_one_successor_needs_no_evidence(write_gadm_cache):
    """East and West Germany both became Germany, so their events need no disambiguating at all."""
    assert successor_state([], "DFR", write_gadm_cache()) == "DEU"


def test_the_successor_holding_the_places_takes_the_event(write_gadm_cache):
    """A Czechoslovak flood naming Prague is Czech. EM-DAT files 417 events under states GADM no
    longer models, and the location text is the only thing that says where they happened."""
    cache_dir = write_gadm_cache()

    assert successor_state([("Praha", None)], "CSK", cache_dir) == "CZE"
    assert successor_state([("Bratislavsky", None)], "CSK", cache_dir) == "SVK"


def test_a_name_both_successors_publish_settles_nothing(write_gadm_cache):
    """Czechia and Slovakia both have a Nové Město. Taking whichever was read first would put the
    event in a country on alphabetical order alone."""
    assert successor_state([("Nove Mesto", None)], "CSK", write_gadm_cache()) is None


def test_the_only_successor_in_the_archive_still_has_to_place_something(write_gadm_cache):
    """Being the sole candidate is not evidence. Assigning the event to the largest successor would
    put every Soviet flood in Russia by default."""
    assert successor_state([("Nowhere At All", None)], "YUG", write_gadm_cache()) is None


def test_a_state_whose_successors_place_nothing_stays_unplaced(write_gadm_cache):
    assert successor_state([("Nowhere At All", None)], "CSK", write_gadm_cache()) is None


def test_a_country_that_still_exists_has_no_successor(write_gadm_cache):
    """The mapping covers dissolved states only; anything else is read from its own gazetteer."""
    assert successor_state([("Sanamxay", None)], "LAO", write_gadm_cache()) is None


def test_a_gazetteer_covers_the_territory_its_country_administers(write_gadm_cache):
    """Fifty-five Indian and twenty-nine Pakistani events name places in Kashmir, which GADM holds
    at the right coordinates under a code neither country's own gazetteer reads."""
    gazetteer = read_gazetteer("IND", write_gadm_cache())

    assert resolve_place("Srinagar", None, gazetteer) == {"Z01.1.1_1"}
    assert resolve_place("Kochi", None, gazetteer) == {"IND.1.1_1"}


@pytest.mark.parametrize(
    "written",
    [
        "North",
        "Countrywide",
        "N.A. on the source",
        "Between Java and Bali",
        "Off the coast of Luzon",
        "Java Sea",
        "Congo river",
        "Mer d'Adaman",
        "Golfe du Lion",
        "Northeastern",
        "Northwestern provinces",
        "East Coast",
        "Alps",
        "Atlantique",
        "Bight of Bangkok",
        ".",
    ],
    ids=[
        "direction",
        "countrywide",
        "not available",
        "between",
        "offshore",
        "sea",
        "river",
        "french sea",
        "french gulf",
        "adjectival direction",
        "adjectival direction with a unit word",
        "direction and a coast",
        "mountain range",
        "french ocean",
        "bight",
        "punctuation",
    ],
)
def test_a_place_that_names_no_unit_is_told_apart(written):
    """A fifth of what stays unplaced is like this. Counting it as a coverage failure understates
    what the resolver reaches, because no gazetteer could ever hold it."""
    assert names_no_unit(written)


@pytest.mark.parametrize(
    "written",
    [
        "Sanamxay",
        "Attapu",
        "Bokeo",
        "Central Java",
        "Rio de Janeiro",
        "Valle del Cauca",
        "Mar del Plata",
        "Sierra Leone",
        "Alpes-Maritimes",
        "Cerro Largo",
        "Northern Territory",
    ],
    ids=[
        "district",
        "province",
        "short province",
        "a direction inside a real name",
        "a river word heading a city",
        "a valley word heading a department",
        "a sea word heading a city",
        "a range word heading a country",
        "the french range word heading a department",
        "a hill word heading a department",
        "a direction heading a real territory",
    ],
)
def test_a_real_place_is_not_mistaken_for_noise(written):
    """`Central` is a Zambian province and `Coast` a Kenyan one, so the direction test is on the
    whole string. The feature words are searched anywhere, so the ones heading real units — `Rio`,
    `Valle`, `Mar`, `Costa`, `Sierra`, `Lago` — are left out of that list entirely."""
    assert not names_no_unit(written)


def test_an_unplaceable_place_is_recorded_as_naming_no_unit(write_gadm_cache):
    """A caller measuring coverage has to be able to leave these out of the denominator without
    guessing which of the unplaced rows were ever placeable."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    unplaceable, missing = resolve_event_places([("Java Sea", None), ("Nowhere At All", None)], gazetteer)

    assert (unplaceable.how, missing.how) == (NAMES_NO_UNIT, NAMED)


def test_a_stretch_of_water_written_inside_a_province_still_reaches_the_province(write_gadm_cache):
    """`Java Sea (West Java province)` names no unit and a container that does. The container is
    what the prose gives, so the event is placed there rather than discarded as noise."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Java Sea", "Bokeo")], gazetteer)

    assert placed == Placement({"LAO.2_1"}, CONTAINED_BY)


def test_a_gazetteer_read_twice_reads_the_same_units(write_gadm_cache):
    """The index is cached to disk between calls, so a rebuild and a cache hit have to agree — the
    cached form is a flat table and the units are rebuilt from it."""
    cache_dir = write_gadm_cache()

    built = read_gazetteer("LAO", cache_dir, force_reload=True)
    from_cache = read_gazetteer("LAO", cache_dir)

    assert (from_cache.names, from_cache.parent_of) == (built.names, built.parent_of)


def test_changing_how_a_name_is_keyed_turns_the_cached_index_over(write_gadm_cache, monkeypatch):
    """The index is built with `match_key`, so a cache keyed on the country alone serves an index
    built under rules that no longer apply — every unit word added would be invisible."""
    cache_dir = write_gadm_cache()
    read_gazetteer("LAO", cache_dir)

    monkeypatch.setattr(place_names, "UNIT_NOUNS", re.compile(r"\b(sanamxay)\b", re.IGNORECASE))
    rekeyed = read_gazetteer("LAO", cache_dir)

    assert "sanamxay" not in rekeyed.names, "the stripped word cannot survive as a key"


def test_a_container_naming_several_places_gives_the_finest_that_resolves(write_gadm_cache):
    """EM-DAT writes `Wayanad district, Kerala state` finest first. Read whole it matches nothing,
    and the village inside it goes unplaced — 1,173 places carry a container like this."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Nowhere At All", "Sanamxay district, Attapu province")], gazetteer)

    assert placed == Placement({"LAO.1.1_1"}, CONTAINED_BY), "the district, not the province beside it"


def test_a_compound_container_narrows_an_ambiguous_name(write_gadm_cache):
    """The container is used to choose between same-named units as well as to stand in for them, and
    both readings have to see every place it names."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Attapu", "Houayxay district, Bokeo province", gazetteer) == {"LAO.2.2_1"}


def test_a_container_whose_first_place_is_unknown_falls_through_to_the_next(write_gadm_cache):
    """The finest place named is often a village GADM does not carry, and the state beside it is
    the whole reason the container is worth reading."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Nowhere At All", "Nowhere district, Bokeo province")], gazetteer)

    assert placed == Placement({"LAO.2_1"}, CONTAINED_BY)


@pytest.mark.parametrize(
    ("written", "published"),
    [("Damavand/Rou Dehan", "Damavand"), ("Kwilu et Tshuapa", "Kwilu"), ("Seoul + Chungchong", "Seoul")],
    ids=["slash", "french and", "plus"],
)
def test_a_place_joined_by_any_of_the_written_separators_is_split(written, published, write_gadm_cache):
    """EM-DAT joins places with `/`, `+`, `&` and the French `et` as readily as with `and`."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())
    joined = written.replace("Damavand", "Sanamxay").replace("Kwilu", "Sanamxay").replace("Seoul", "Sanamxay")

    assert resolve_place(joined, None, gazetteer) == {"LAO.1.1_1"}


@pytest.mark.parametrize(
    ("written", "published"),
    [
        ("Arr. Mechelen", "Mechelen"),
        ("Barangay Rizal", "Rizal"),
        ("towns of Kauswagan", "Kauswagan"),
        ("woredas of Dale", "Dale"),
        ("Oshai Darray village", "Oshai Darray"),
        ("Lamu archipel", "Lamu"),
        ("Kalehe territory", "Kalehe"),
        ("Ayeyarwady divisions", "Ayeyarwady"),
        ("Lancashire CC", "Lancashire"),
        ("Tafawa alewa Local Government Area", "Tafawa alewa"),
        ("Keumbu market", "Keumbu"),
        ("Goalanda Upazilla", "Goalanda"),
    ],
    ids=[
        "abbreviated prefix",
        "barangay",
        "town of",
        "woreda of",
        "village",
        "archipel",
        "territory",
        "division",
        "county council",
        "local government area",
        "market",
        "upazila",
    ],
)
def test_a_unit_word_before_the_name_is_stripped_like_one_after(written, published):
    """EM-DAT writes the unit word on whichever side the local convention puts it, and a whole
    Belgian event can carry `Arr.` on every name."""
    assert match_key(written) == match_key(published)


def test_a_unit_word_that_names_a_real_place_is_left_alone():
    """GADM carries units called Canton, so stripping the word would reduce them to nothing and
    drop them out of the index entirely."""
    assert match_key("Canton") == "canton"


def test_a_transposition_is_one_slip_from_the_name_it_garbles(write_gadm_cache):
    """`Heart` for Herat and `Tapei` for Taipei are single transpositions, which a substitution-only
    rule cannot see. Approximate matching proposes them; the table still decides."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert nearest_name("Snaamxay", gazetteer, name_shapes(gazetteer)) == "sanamxay"


@pytest.mark.parametrize(
    "written",
    ["anamxay", "Zanamxay", "Sanamxai", "Snaamxay"],
    ids=["first letter lost", "first letter wrong", "last letter wrong", "transposed"],
)
def test_a_slip_in_the_first_letter_still_finds_the_name(written, write_gadm_cache):
    """Names are filed for lookup under their opening letters, so a name that lost or changed its
    first one is otherwise unreachable however close it is: `llinois` is one letter from Illinois."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert nearest_name(written, gazetteer, name_shapes(gazetteer)) == "sanamxay"


def test_one_entry_can_stand_for_several_units(write_gadm_cache):
    """`Nghe Tinh` was a province until 1991 and is now two; `Western Visayas` is a statistical
    region GADM never carried. Neither is a misspelling of any single unit."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())._replace(
        corrections={"oldattapu": ("sanamxay", "samakhixay")}
    )

    (placed,) = resolve_event_places([("Old Attapu", None)], gazetteer)

    assert placed == Placement({"LAO.1.1_1", "LAO.1.2_1"}, CORRECTED)


def test_an_entry_naming_a_unit_gadm_dropped_reaches_the_rest(write_gadm_cache):
    """A curated entry is written by hand against a GADM vintage that moves under it, so a name it
    lists may no longer exist. The units that do still resolve."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())._replace(
        corrections={"oldattapu": ("sanamxay", "gone from the archive")}
    )

    (placed,) = resolve_event_places([("Old Attapu", None)], gazetteer)

    assert placed == Placement({"LAO.1.1_1"}, CORRECTED)


def test_an_entry_naming_nothing_the_archive_holds_is_not_a_correction(write_gadm_cache):
    """A curated entry outlives the GADM vintage it was written against. One that reaches no unit
    at all is a miss, and recording it as a correction would claim geography we do not have."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())._replace(
        corrections={"oldattapu": ("gone from the archive",)}
    )

    assert resolve_event_places([("Old Attapu", None)], gazetteer) == [Placement(set(), NAMED)]


def test_an_entry_naming_several_units_is_read_as_several(write_gadm_cache, tmp_path):
    """`Nghe Tinh` was one province until 1991 and is two now, so the table has to be able to say
    so — the same shape a statistical region or an island group needs."""
    table = tmp_path / "corrections.csv"
    table.write_text("iso,written,corrected\nVNM,Nghe Tinh,nghean|hatinh\n", encoding="utf-8")

    assert read_name_corrections("VNM", path=table) == {"nghetinh": ("nghean", "hatinh")}


def test_a_country_reads_the_territory_it_no_longer_contains(write_gadm_cache):
    """An event recorded before a secession names a place its country then held. `Malakal` under
    SDN is South Sudanese now, and reaches nothing unless the successor is read alongside."""
    gazetteer = read_gazetteer("ETH", write_gadm_cache())

    assert resolve_place("Asmara", None, gazetteer) == {"ERI.1.1_1"}
    assert resolve_place("Mekele", None, gazetteer) == {"ETH.1.1_1"}


def test_a_country_that_lost_nothing_reads_only_itself(write_gadm_cache):
    """The table covers secessions only; every other country is read from its own code alone."""
    gazetteer = read_gazetteer("ERI", write_gadm_cache())

    assert resolve_place("Mekele", None, gazetteer) == set()


def test_a_checked_misspelling_is_preferred_to_the_container(write_gadm_cache):
    """A correction someone checked is as certain as the container and names one unit where the
    container names everything inside it."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())._replace(corrections={"sanamxai": ("sanamxay",)})

    (placed,) = resolve_event_places([("Sanamxai", "Bokeo")], gazetteer)

    assert placed == Placement({"LAO.1.1_1"}, CORRECTED)


def test_an_unchecked_slip_the_container_vouches_for_is_taken(write_gadm_cache):
    """`Zheijang` beside Guangdong, Hunan and Fujian is not a guess. Where the container holds
    exactly one candidate, one edit from a published name is enough."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Sanamxai", "Attapu")], gazetteer)

    assert placed == Placement({"LAO.1.1_1"}, INFERRED)


def test_an_unchecked_slip_a_sibling_vouches_for_is_taken(write_gadm_cache):
    """The event's other places do the vouching where the prose gave no container."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    _, inferred = resolve_event_places([("Samakhixay", None), ("Sanamxai", None)], gazetteer)

    assert inferred == Placement({"LAO.1.1_1"}, INFERRED)


def test_an_unchecked_slip_nothing_vouches_for_is_refused(write_gadm_cache):
    """`Lynmouth` sits one edit from Lynemouth, four hundred miles away. With no container and no
    sibling, one edit is a guess and the place stays unplaced."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_event_places([("Sanamxai", None)], gazetteer) == [Placement(set(), NAMED)]


def test_a_slip_the_container_does_not_hold_is_refused(write_gadm_cache):
    """A container that vouches for none of the candidates is not corroboration."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Sanamxai", "Bokeo")], gazetteer)

    assert placed == Placement({"LAO.2_1"}, CONTAINED_BY), "falls back to the container, not the slip"


def test_a_slip_vouched_for_at_two_places_at_once_is_refused(write_gadm_cache):
    """`Attapu` is a province and, elsewhere, a district of Bokeo. An event naming places in both
    vouches for both candidates, which is not corroboration — it is a coin toss."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    *_, ambiguous = resolve_event_places([("Sanamxay", None), ("Houayxay", None), ("Attapy", None)], gazetteer)

    assert ambiguous == Placement(set(), NAMED)


def test_a_place_whose_name_ends_in_a_unit_word_survives():
    """`Rhode Island` and `Coast` are real units. The words are stripped from both the mention and
    the published name, so the two still meet — but a name reduced to nothing would be lost."""
    assert match_key("Rhode Island") == match_key("Rhode Island")
    assert match_key("Kelantan") == "kelantan"


def test_a_container_with_a_direction_on_it_reaches_the_unit(write_gadm_cache):
    """`northern Queensland` names a container the prose plainly gives, and Queensland is a unit.
    Everything built for reading a place name has to reach the container too."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    (placed,) = resolve_event_places([("Nowhere At All", "Northern Bokeo")], gazetteer)

    assert placed == Placement({"LAO.2_1"}, CONTAINED_BY)


def test_a_misspelled_container_the_event_vouches_for_reaches_the_unit(write_gadm_cache):
    """GADM spells it Hirat and EM-DAT writes Herat. The container carries the same slips a name
    does, and the event's other places vouch for it the same way."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    _, fallen_back = resolve_event_places([("Sanamxay", None), ("Nowhere At All", "Samakhixai")], gazetteer)

    assert fallen_back == Placement({"LAO.1.2_1"}, CONTAINED_BY)


def test_an_exact_container_beats_a_misspelling_of_a_finer_one(write_gadm_cache):
    """The prose writes a container finest first, but an exact match on the coarser part is better
    evidence than a slip on the finer one — even when the event vouches for the slip."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    _, fallen_back = resolve_event_places([("Sanamxay", None), ("Nowhere At All", "Samakhixai, Bokeo")], gazetteer)

    assert fallen_back == Placement({"LAO.2_1"}, CONTAINED_BY), "Bokeo exactly, not a slip for Samakhixay"


@pytest.mark.parametrize(
    ("mangled", "intended"),
    [("SÃ©dhiou", "Sédhiou"), ("SÃƒÂ©dhiou", "Sédhiou"), ("HanoÃ¯", "Hanoï")],
    ids=["decoded once through the wrong codec", "decoded twice", "a different accent"],
)
def test_a_name_decoded_through_the_wrong_codec_reaches_the_name_it_mangles(mangled, intended):
    """EM-DAT carries names written as UTF-8 and read back as cp1252, some of them twice over. Until
    the mangling comes off they key to letters GADM never published."""
    assert match_key(mangled) == match_key(intended)


@pytest.mark.parametrize(
    "written",
    ["Sédhiou", "Đà Nẵng", "Attapu", "Kraków", "São Paulo", "Côte-d'Or"],
    ids=["accented", "vietnamese", "plain", "polish", "portuguese", "french"],
)
def test_a_name_that_was_never_mangled_survives_the_repair(written):
    """The repair is only safe because it is self-guarding: a correctly written name encodes to
    bytes that are not valid UTF-8, so the round trip fails and leaves it alone. A name that
    silently changed here would be corrupted for every lookup."""
    assert repair_mojibake(written) == written


def test_a_dash_joining_two_places_reaches_both(write_gadm_cache):
    """EM-DAT joins a pair of districts with a dash as readily as with `and`, and refusing to split
    leaves both unplaced."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Sanamxay-Samakhixay", None, gazetteer) == {"LAO.1.1_1", "LAO.1.2_1"}


def test_a_dash_only_half_of_which_names_a_place_is_left_whole(write_gadm_cache):
    """A dash sits inside a single name — `Nord-Ubangi`, `Alpes-Maritimes` — more often than it
    joins two. Only unanimity tells a pair from one name the gazetteer spells differently, so a
    half-match is read as a misspelling rather than split."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Sanamxay-Nowhere", None, gazetteer) == set()


def test_a_dash_split_refuses_a_part_too_short_to_be_a_place(write_gadm_cache):
    """`Ali-Shan` is one Taiwanese mountain whose halves are both Chinese counties. A syllable is
    not a place, whatever it happens to spell."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Xay-Sanamxay", None, gazetteer) == set(), "Xay names a district and is still too short"


def test_a_route_running_between_two_places_is_not_split_across_them(write_gadm_cache):
    """`Qom-Teheran highway` names the road, not the provinces at its ends. Splitting it claims a
    footprint the event never had."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Sanamxay-Samakhixay road", None, gazetteer) == set()


def test_a_place_beside_a_road_reaches_the_place(write_gadm_cache):
    """A route noun attaches to one place as often as it spans two, and `Bankass road` is a cercle
    of Mali with a word after it."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert resolve_place("Sanamxay road", None, gazetteer) == {"LAO.1.1_1"}


def test_an_adjective_built_from_a_place_reaches_it(write_gadm_cache):
    """EM-DAT writes a Polish powiat as `Tarnobrzeski` and GADM publishes Tarnobrzeg. Suffixing
    reshapes the stem, so the match is on what the name opens with rather than what it equals."""
    gazetteer = read_gazetteer("POL", write_gadm_cache())

    (placed,) = resolve_event_places([("Tarnobrzeski", None)], gazetteer)

    assert placed == Placement({"POL.1.1_1"}, DERIVED)


def test_an_adjective_whose_stem_opens_two_names_reaches_neither(write_gadm_cache):
    """`Rybnicki` leaves a stem that opens both Rybnik and Rybno. Two names equally reachable is a
    choice, and this makes none."""
    gazetteer = read_gazetteer("POL", write_gadm_cache())

    (placed,) = resolve_event_places([("Rybnicki", None)], gazetteer)

    assert placed.gids == set()


def test_a_country_that_writes_no_adjectives_leaves_a_name_ending_that_way_alone(write_gadm_cache):
    """The suffix is Polish morphology, not a general rule: `Nagasaki` and `Helsinki` end the same
    way and are the names themselves."""
    gazetteer = read_gazetteer("LAO", write_gadm_cache())

    assert gazetteer.adjective is None
    assert place_names._derived_unit("Sanamxayski", gazetteer) == set()


def test_a_keying_rule_with_no_pattern_of_its_own_still_turns_the_cache_over(write_gadm_cache, monkeypatch):
    """Fingerprinting the unit-noun pattern covers only the rules that are patterns. Repairing a
    mis-decoded name is a codec round trip, and a cache that cannot see it changing serves an index
    built under rules that no longer apply."""
    cache_dir = write_gadm_cache()
    read_gazetteer("LAO", cache_dir)

    monkeypatch.setattr(place_names, "repair_mojibake", lambda text: text.replace("a", "z"))
    rekeyed = read_gazetteer("LAO", cache_dir)

    assert "sznzmxzy" in rekeyed.names, "the index was rebuilt under the changed rule"


def test_changing_which_rows_the_index_reads_turns_the_cache_over(write_gadm_cache, monkeypatch):
    """Fingerprinting the keying rules covers how a name becomes a key, not which rows get one.
    Ethiopia's index carries Eritrea because a table here says it used to hold it, and a cache that
    cannot see that table change serves an index built when it said something else."""
    cache_dir = write_gadm_cache()
    read_gazetteer("ETH", cache_dir)

    monkeypatch.setattr(place_names, "FORMERLY_INCLUDED", {})
    reread = read_gazetteer("ETH", cache_dir)

    assert "maekel" not in reread.names, "Eritrea's units cannot survive the table that pulled them in"


def test_one_layer_does_not_answer_for_another(write_gadm_cache):
    """`layer` chooses which table the index is built from, so two layers are two different indexes.
    Keyed alike the second caller is handed the first one's units."""
    cache_dir = write_gadm_cache()
    archive = cache_dir / "gadm" / "gadm_410.gpkg"
    everything = gpd.read_file(archive, layer="gadm_410")
    everything[everything["GID_0"] == "ZMB"].to_file(archive, layer="zambia_only")

    read_gazetteer("LAO", cache_dir)
    from_zambia_only = read_gazetteer("LAO", cache_dir, layer="zambia_only")

    assert not from_zambia_only.names, "a layer holding no Laos must not answer with another layer's units"
