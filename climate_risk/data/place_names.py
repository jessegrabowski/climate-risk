import csv
import hashlib
import re
import sqlite3

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from itertools import batched
from pathlib import Path
from typing import NamedTuple

import polars as pl

from anyascii import anyascii

from climate_risk.data.cache import builder_fingerprint, cached, polars_parquet
from climate_risk.data.gadm import GADM_LAYER, administered_territories, gadm_dir, gadm_path

# Words naming what a unit is rather than which one it is. EM-DAT writes them inconsistently and
# GADM does not carry them, so `Kalin-Aapayo province` has to reach `Kalin-Aapayo`.
# Nouns that can join to a name with `of`, as `towns of Kauswagan` does.
_JOINED_BY_OF = (
    "towns?",
    "villages?",
    "districts?",
    "provinces?",
    "regions?",
    "states?",
    "departments?",
    "cities",
    "city",
    "municipalit(?:y|ies)",
    "governorates?",
    "prefectures?",
    "woredas?",
)

# Nouns written on their own, before or after the name, and the abbreviations they appear as.
_STANDS_ALONE = (
    "sub[- ]?districts?",
    "local government areas?",
    "divisions?",
    "territor(?:y|ies)",
    "emirates?",
    "upazillas?",
    "upazilas?",
    "communes?",
    "markets?",
    "cc",
    "prov",
    "regencies",
    "regency",
    "islands?",
    "isl",
    "archipel(?:ago)?",
    "counties",
    "county",
    "areas?",
    "near",
    "around",
    "arrond",
    "arr",
    "barangay",
    "brgy",
    "kel",
    "kec",
    "kab",
    "desa",
    "roads?",
    "highways?",
    "airports?",
    "pass(?:es)?",
)

# Prepositional phrases, which have no bare form.
_LOCATES = ("outskirts of", "vicinity of", "coast of")

# Alternation is leftmost-first, so the `of` forms have to precede the bare nouns they contain:
# a bare `town` would otherwise match inside `towns of` and strand the preposition.
UNIT_NOUNS = re.compile(
    r"\b("
    + "|".join((*(f"{noun}\\s+of" for noun in _JOINED_BY_OF), *_LOCATES, *_JOINED_BY_OF, *_STANDS_ALONE))
    + r")\b\.?",
    re.IGNORECASE,
)


# EM-DAT joins places with `and`, its French `et`, and the bare `&`, `+` and `/`. GADM writes
# `Newfoundland and Labrador` and `Komenda/Edina/Eguafo/Abirem`, so the whole string has to be tried
# before its parts.
CONJUNCTION = re.compile(r"\s+(?:and|et)\s+|\s*[&+/]\s*", re.IGNORECASE)


# A road runs between places rather than sitting in one. The noun comes off like any other, so
# `Bankass road` reaches Bankass, but a route naming two places is a line and splitting it puts
# the event on both ends of a journey it only crossed.
ROUTE = re.compile(r"\b(?:roads?|highways?)\b", re.IGNORECASE)

# A part of a dash-joined name shorter than this is a syllable rather than a place: splitting
# `Ali-Shan` yields two Chinese counties and the event was in neither.
MIN_SPLIT_LENGTH = 4

# A dash joins two places as often as it sits inside one name, so the whole is always read first.
DASH = re.compile(r"\s*[-\u2013\u2014]\s*")

# `Wayanad district, Kerala state` names the same place twice over, finest first. Read whole it
# matches nothing; read in parts each one matches a unit.
CONTAINER_PARTS = re.compile(r"\s*[,;]\s*")

# `Between Java and Bali` names a stretch of sea by its shores. Splitting it puts the event on
# whichever shore happens to resolve.
RELATIONAL = re.compile(r"^\s*(?:between|off|offshore|au large)\b", re.IGNORECASE)


# A sea, a river or a mountain range is not an administrative unit, whatever it is one edit from.
# Units named after a feature — `River Nile State`, `Bay of Plenty` — still match exactly; only the
# approximate lookup refuses them.
FEATURE = re.compile(
    r"\b(seas?|oceans?|rivers?|lakes?|gulfs?|straits?|detroit|bays?|mountains?|mts?|mount|peaks?|"
    r"valleys?|deltas?|peninsulas?|channels?|canals?|reservoirs?|dams?|volcanoe?s?|glaciers?|"
    r"basins?|capes?|sounds?|fjords?|lagoons?|swamps?|forests?|deserts?|"
    # EM-DAT carries a good deal of French, and some Spanish and Portuguese.
    r"mers?|oc[ée]ans?|fleuves?|rivi[èe]res?|golfes?|golfos?|estrechos?|baies?|bah[ía]as?|"
    r"montagnes?|monta[ñn]as?|vall[ée]es?|massifs?|presqu.[îi]les?|"
    r"bights?|collines?|quebradas?|himalayas?|andes|atlantique|pacifique|m[ée]diterran[ée]e|alps)\b",
    re.IGNORECASE,
)

# Written through the keying rules, so the cache fingerprint moves when any of them does, whether
# or not the rule can be expressed as a pattern.
_KEYING_PROBE = "SÃƒÂ©dhiou province & Kalin-Aapayo district"

# Below this a single edit is a large share of the name and the match stops meaning anything:
# `Arora` sits one edit from both `Aurora` and `Arora` in half the countries that have either.
MIN_APPROXIMATE_LENGTH = 6

# Corrections a person or a model has checked one at a time. Approximate matching proposes them;
# only what is written here is ever applied, so an unreviewed near miss leaves a place unplaced
# rather than renaming it.
NAME_CORRECTIONS = Path(__file__).parent / "name_corrections.csv"

# EM-DAT files an event under the country that existed at the time, and GADM only models the ones
# that exist now. Where a state left one successor the mapping is settled; where it left several
# the location text has to choose between them.
SUCCEEDED_BY = {
    "AZO": ("PRT",),
    "CSK": ("CZE", "SVK"),
    "DDR": ("DEU",),
    "DFR": ("DEU",),
    "SUN": (
        "ARM",
        "AZE",
        "BLR",
        "EST",
        "GEO",
        "KAZ",
        "KGZ",
        "LTU",
        "LVA",
        "MDA",
        "RUS",
        "TJK",
        "TKM",
        "UKR",
        "UZB",
    ),
    "YMD": ("YEM",),
    "YMN": ("YEM",),
    "YUG": ("BIH", "HRV", "MKD", "MNE", "SRB", "SVN", "XKO"),
}

# Written places that name no administrative unit and never will: a compass point covering the
# whole country, a position between two others, a stretch of water, or nothing at all. They are
# recorded rather than counted as failures, because no source can place them.
_COMPASS = r"(?:north|south|east|west|central|centre|center)(?:ern)?"

QUALIFIER = re.compile(
    rf"^(?:{_COMPASS}(?:[-\s]*{_COMPASS})?"
    r"(?:\s+(?:coast|coastal|regions?|areas?|parts?|provinces?|districts?|states?|counties))?"
    r"|country[- ]?wide|nation[- ]?wide|entire (?:country|nation).*|whole country|all country"
    r"|much of (?:the )?(?:country|nation)|widespread|unknown|not available|no information"
    r"|n\.?a\.?(?: on the source)?)$",
    re.IGNORECASE,
)
NOTHING_LEGIBLE = re.compile(r"^[\W\d\s]*$")

# A country that still exists but no longer contains what it did when the event was recorded. The
# text names a real place; GADM files it under the state that holds it now.
FORMERLY_INCLUDED = {
    "ETH": ("ERI",),
    "IDN": ("TLS",),
    "PAK": ("BGD",),
    "SDN": ("SSD",),
    "SRB": ("MNE", "XKO"),
}

# `South Bihar` and `Ontario Central` qualify a unit rather than naming one. The words are part of
# plenty of real names — Lower Shabelle, West Bengal, Northern Territory — so the qualifier only
# comes off once the name as written has failed.
DIRECTIONAL_QUALIFIER = re.compile(
    r"^(?:north|south|east|west|central|centre|upper|lower|greater)(?:ern)?(?:\s+of)?\s+"
    r"|\s+(?:north|south|east|west|central|centre|upper|lower)(?:ern)?$",
    re.IGNORECASE,
)

# EM-DAT writes a Polish powiat as the adjective built from its seat — `Sieradzki` for the powiat
# of Sieradz — while GADM publishes the seat. Suffixing reshapes the stem as well as ending it,
# so what the strip leaves opens the published name rather than equalling it.
DERIVED_ADJECTIVE = {"POL": re.compile(r"(?:owski|inski|icki|ecki|eski|ski|cki|zki|ki)$")}

# How a written place reached the units it did.
NAMED = "named"
LOCATED = "located"
CONTAINED_BY = "container"
CORRECTED = "corrected"
INFERRED = "inferred"
QUALIFIED = "qualified"
NAMES_NO_UNIT = "names no unit"
DERIVED = "derived"


# GADM publishes a level 5, for Belgium and Rwanda only.
INDEXABLE_LEVELS = (1, 2, 3, 4)

# The flattened index, as it is cached: one row per name a unit is published under.
INDEX_COLUMNS = {"key": pl.String, "gid": pl.String, "level": pl.Int8, "parent": pl.String}


class Placement(NamedTuple):
    """Where one written place put an event.

    Parameters
    ----------
    gids : set of str
        The GADM identifiers reached, empty where the place reached none.
    how : str
        ``NAMED`` where the written place itself resolved, ``CONTAINED_BY`` where only the container
        it was written in did, which is coarser than the prose supports. ``NAMED`` where ``gids`` is
        empty and nothing was reached at all.
    """

    gids: set[str]
    how: str


class Unit(NamedTuple):
    """A GADM unit a name can refer to.

    Parameters
    ----------
    gid : str
        The GADM identifier.
    level : int
        Administrative level, 1 through 4.
    parent : str or None
        The identifier of the unit one level up, or None at level 1.
    """

    gid: str
    level: int
    parent: str | None


class Gazetteer(NamedTuple):
    """One country's units, searchable by name and walkable upwards.

    Parameters
    ----------
    names : dict mapping str to set of Unit
        Units keyed on :func:`match_key` of every name they are published under.
    parent_of : dict mapping str to str or None
        The unit one level up from each identifier.
    shapes : dict mapping tuple to set of str
        The keys sharing a first character and a length, which is what an approximate lookup scans
        instead of every name in the country.
    corrections : dict mapping str to tuple of str
        The published names each checked entry stands for, keyed on :func:`match_key`. A misspelling
        gives one. A name GADM never carried as a unit, such as a merged province since split or a
        statistical region, gives the several it covers.
    adjective : re.Pattern or None
        The suffix this country's language builds a place adjective with, where written mentions use
        one. None where they do not.
    """

    names: dict[str, set[Unit]]
    parent_of: dict[str, str | None]
    shapes: dict[tuple[str, int], set[str]]
    corrections: dict[str, tuple[str, ...]]
    adjective: re.Pattern[str] | None = None

    def _upwards(self, gid: str) -> Iterator[str]:
        """Yield the unit and each container above it, outwards."""
        seen: set[str] = set()
        current: str | None = gid
        # A Gazetteer can be built by hand as well as read, so the walk does not assume acyclic
        # parentage even though :func:`read_gazetteer` refuses to record it.
        while current and current not in seen:
            seen.add(current)
            yield current
            current = self.parent_of.get(current)

    def ancestry(self, gid: str) -> set[str]:
        """Name the unit and every unit containing it."""
        return set(self._upwards(gid))

    def top_container(self, gid: str) -> str:
        """Name the outermost unit holding this one, which is the unit itself where it has no parent."""
        *_, outermost = self._upwards(gid)

        return outermost


_NAME_FIELDS = ("GID", "NAME", "VARNAME")


def _name_columns(level: int, available: set[str]) -> str:
    """Select one level's identifier and names, substituting an empty literal for an absent column."""
    return ", ".join(f"{field}_{level}" if f"{field}_{level}" in available else "''" for field in _NAME_FIELDS)


def repair_mojibake(text: str) -> str:
    """
    Undo text that was decoded as cp1252 after being written as UTF-8, however many times.

    ``SÃ©dhiou`` is Sédhiou read through the wrong codec, and EM-DAT carries names that went
    through it twice. The repair is its own guard: text that was never mangled does not survive
    the round trip, because the bytes a correct name encodes to are not valid UTF-8.

    Parameters
    ----------
    text : str
        The name as written.

    Returns
    -------
    name : str
        The name with the mangling undone, or unchanged where there was none.
    """
    while True:
        try:
            once = text.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
        if once == text:
            return text
        text = once


def match_key(name: str) -> str:
    """
    Reduce a place name to what a written mention and a GADM name can be expected to share.

    Repaired of any mis-decoding, transliterated, stripped of the noun saying what kind of unit it
    is, and reduced to letters and digits, so ``Kalin-Aapayo province`` and ``Kalin Aapayo`` meet.

    Parameters
    ----------
    name : str
        A place as written, from either source.

    Returns
    -------
    key : str
        The comparison key, empty where the name was only a unit noun.
    """
    repaired = UNIT_NOUNS.sub(" ", repair_mojibake(name))

    return "".join(character for character in anyascii(repaired) if character.isalnum()).casefold()


def keying_fingerprint() -> str:
    """Fingerprint the rules that turn a name into a key, so a change to them invalidates the cache."""
    rules = "|".join((UNIT_NOUNS.pattern, str(INDEXABLE_LEVELS), match_key(_KEYING_PROBE)))

    return hashlib.sha256(rules.encode()).hexdigest()[:12]


def read_gazetteer(iso: str, cache_dir: Path, *, layer: str = GADM_LAYER, force_reload: bool = False) -> Gazetteer:
    """
    Read one country's GADM units into a gazetteer.

    Every level the archive publishes is indexed, because written mentions reach all of them: a
    location string names provinces, districts and villages in the same breath. A name reaches a
    set rather than a unit because names repeat: a province and its capital district often share
    one, and a village name recurs freely.

    Parentage comes from the row rather than from the shape of the identifier. Identifiers usually
    nest, but 37 rows across five countries do not, and a prefix test would place those units
    outside their own province.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code of the country to read.
    cache_dir : Path
        Directory the caches live under.
    layer : str, optional
        Layer to read inside the GeoPackage. Default ``GADM_LAYER``.
    force_reload : bool, optional
        Rebuild even when the index is already cached. Default False.

    Returns
    -------
    gazetteer : Gazetteer
        The country's units.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.place_names import read_gazetteer

        gazetteer = read_gazetteer("LAO", Path("data"))
    """

    def build() -> pl.DataFrame:
        indexed: list[tuple[str, str, int, str | None]] = []
        with sqlite3.connect(gadm_path(cache_dir)) as connection:
            available = {row[1] for row in connection.execute(f'PRAGMA table_info("{layer}")')}
            # A variant-name column is optional; an identifier and a name are what make a level readable.
            levels = [level for level in INDEXABLE_LEVELS if {f"GID_{level}", f"NAME_{level}"} <= available]
            columns = ", ".join(_name_columns(level, available) for level in levels)
            # Kashmir is filed under codes of its own rather than under the country administering
            # it, and a pre-secession event names places the country no longer contains.
            held = (
                iso,
                *administered_territories(iso, cache_dir, layer=layer),
                *FORMERLY_INCLUDED.get(iso, ()),
            )
            placeholders = ", ".join("?" * len(held))
            query = f'SELECT DISTINCT {columns} FROM "{layer}" WHERE GID_0 IN ({placeholders})'

            for row in connection.execute(query, held):
                parent = None
                for level, (gid, name, variants) in zip(levels, batched(row, len(_NAME_FIELDS)), strict=True):
                    # GADM writes an unnamed unit as `?` at every level it spans, and a unit cannot
                    # contain itself: the chain ends where the identifier stops changing.
                    if not gid or gid == parent:
                        break
                    # A row without a key still records the unit's parentage: GADM leaves some
                    # units unnamed, and a walk upwards passes through them.
                    indexed.append(("", gid, level, parent))
                    for field in (name, variants):
                        for alternative in str(field or "").split("|"):
                            if key := match_key(alternative):
                                indexed.append((key, gid, level, parent))
                    parent = gid

        return pl.DataFrame(indexed, schema=INDEX_COLUMNS, orient="row")

    indexed = cached(
        gadm_dir(cache_dir),
        "gazetteer",
        build,
        polars_parquet(),
        # `layer` chooses what is read; `keying` covers the rules that turn a name into a key,
        # and `reading` the rules that decide which rows carry one.
        params={
            "iso": iso,
            "keying": keying_fingerprint(),
            "layer": layer,
            "reading": builder_fingerprint(build, _NAME_FIELDS, FORMERLY_INCLUDED.get(iso, ())),
        },
        force=force_reload,
    )

    names: dict[str, set[Unit]] = defaultdict(set)
    parent_of: dict[str, str | None] = {}
    for key, gid, level, parent in indexed.iter_rows():
        if key:
            names[key].add(Unit(gid, level, parent))
        parent_of[gid] = parent

    return Gazetteer(
        dict(names), parent_of, _shape_index(names), read_name_corrections(iso), DERIVED_ADJECTIVE.get(iso)
    )


def read_name_corrections(iso: str, *, path: Path = NAME_CORRECTIONS) -> dict[str, tuple[str, ...]]:
    """
    Read the checked corrections for one country.

    Parameters
    ----------
    iso : str
        ISO 3166-1 alpha-3 code of the country to read.
    path : Path, optional
        The corrections table. Default the one shipped with the package.

    Returns
    -------
    corrections : dict mapping str to tuple of str
        The published names each entry stands for, keyed by :func:`match_key`. An entry naming
        several units, as a merged province since split does, separates them with a pipe.
    """
    if not path.exists():
        return {}

    with path.open(encoding="utf-8", newline="") as handle:
        return {
            match_key(row["written"]): tuple(row["corrected"].split("|"))
            for row in csv.DictReader(handle)
            if row["iso"] == iso and row["corrected"]
        }


def _shape_index(keys: Iterable[str]) -> dict[tuple[str, int], set[str]]:
    """File each key under its length and each of its first two letters."""
    shapes: dict[tuple[str, int], set[str]] = defaultdict(set)
    for key in keys:
        for letter in {key[0], key[1] if len(key) > 1 else key[0]}:
            shapes[(letter, len(key))].add(key)

    return dict(shapes)


def name_shapes(gazetteer: Gazetteer) -> dict[tuple[str, int], set[str]]:
    """
    Group a country's names for approximate lookup, by length and by each of their first two letters.

    Filing a name under its second letter as well as its first is what makes a slip in the first one
    reachable: ``llinois`` has lost the letter its bucket would otherwise be chosen by.
    """
    return _shape_index(gazetteer.names)


def _one_slip_apart(written: str, published: str) -> bool:
    """Whether one insertion, deletion, substitution or transposition turns one key into the other."""
    if abs(len(written) - len(published)) > 1:
        return False
    if len(written) == len(published):
        differing = [
            position for position, pair in enumerate(zip(written, published, strict=True)) if pair[0] != pair[1]
        ]
        if len(differing) <= 1:
            return True
        first, second = differing[0], differing[-1]

        return (
            len(differing) == 2
            and second == first + 1
            and written[first] == published[second]
            and written[second] == published[first]
        )

    longer, shorter = (written, published) if len(written) > len(published) else (published, written)
    skipped = next(
        (position for position, pair in enumerate(zip(longer, shorter, strict=False)) if pair[0] != pair[1]),
        len(shorter),
    )

    return longer[skipped + 1 :] == shorter[skipped:]


def nearest_name(name: str, gazetteer: Gazetteer, shapes: dict[tuple[str, int], set[str]]) -> str | None:
    """
    Find the one published name a written place is a misspelling of.

    A match is only offered where the written name is long enough for a single edit to be a small
    part of it, does not name a physical feature, and is one slip from exactly one published name.
    A slip is an insertion, deletion, substitution or transposition. Two names equally close is not
    a near miss but a choice, and this makes none.

    Parameters
    ----------
    name : str
        The place as written.
    gazetteer : Gazetteer
        The country's units, from :func:`read_gazetteer`.
    shapes : dict mapping tuple to set of str
        The country's names grouped for searching, from :func:`name_shapes`.

    Returns
    -------
    match : str or None
        The matching key in ``gazetteer.names``, or None where the gates reject the name or no
        single published name is close enough.
    """
    if FEATURE.search(name):
        return None

    key = match_key(name)
    if len(key) < MIN_APPROXIMATE_LENGTH or key in gazetteer.names:
        return None

    # Names are filed under each of their first two letters, so a slip in either one still finds
    # them while the search stays local.
    opening = {key[0], key[1]} if len(key) > 1 else {key[0]}
    nearby = {
        candidate
        for letter in opening
        for length in range(len(key) - 1, len(key) + 2)
        for candidate in shapes.get((letter, length), ())
        if _one_slip_apart(key, candidate)
    }

    return next(iter(nearby)) if len(nearby) == 1 else None


def container_parts(parent: str) -> list[str]:
    """
    Split a container into the places it names, finest first.

    Parameters
    ----------
    parent : str
        The container as the prose wrote it.

    Returns
    -------
    parts : list of str
        Each place it names, in the order written.
    """
    return [part for part in CONTAINER_PARTS.split(parent) if part.strip()]


def _derived_unit(name: str, gazetteer: Gazetteer) -> set[str]:
    """
    Name the unit an adjective was built from, where the stem it leaves belongs to exactly one.

    ``Tarnobrzeski`` is built from Tarnobrzeg, so the suffix takes part of the stem with it and what
    remains opens the published name rather than equalling it. A stem opening several names is not a
    reading, so it is refused.
    """
    if gazetteer.adjective is None:
        return set()

    key = match_key(name)
    stem = gazetteer.adjective.sub("", key)
    if not stem or stem == key:
        return set()

    built_on = {published for published in gazetteer.names if published != key and published.startswith(stem)}
    if len(built_on) != 1:
        return set()

    return _outermost({unit.gid for unit in gazetteer.names[next(iter(built_on))]}, gazetteer)


def _dashed_units(name: str, parent: str | None, gazetteer: Gazetteer) -> set[str]:
    """
    Name the units a dash-joined list reaches, where every one of its parts names a unit.

    A dash joins two places as often as it belongs inside a single name, and nothing in the text
    tells them apart. Unanimity does: ``Grevena-Kozani`` is a pair because both halves are
    prefectures, while a name that only half resolves is one name the gazetteer spells differently.
    """
    parts = [part.strip() for part in DASH.split(name) if part.strip()]
    if len(parts) < 2 or any(len(match_key(part)) < MIN_SPLIT_LENGTH for part in parts):
        return set()

    reached = [_units_named(part, parent, gazetteer) for part in parts]

    return _outermost(set().union(*reached), gazetteer) if all(reached) else set()


def _qualified_unit(name: str, gazetteer: Gazetteer) -> set[str]:
    """
    Name the unit a directional qualifier was applied to, where it names exactly one.

    ``South Bihar`` is part of Bihar and reaches it once the qualifier comes off, which is coarser
    than the prose but the only reading GADM can hold. A base naming several units is not a reading,
    so it is refused.
    """
    base = DIRECTIONAL_QUALIFIER.sub(" ", name).strip()
    if not base or base == name.strip():
        return set()

    qualified = _outermost(_units_named(base, None, gazetteer), gazetteer)

    return qualified if len(qualified) == 1 else set()


def _corroborated_slip(name: str, container: set[str], pinned: set[str], gazetteer: Gazetteer) -> set[str]:
    """
    Take a one-slip match only where the rest of the event agrees with it.

    One edit from a published name is a guess on its own. ``Lynmouth`` sits one edit from Lynemouth,
    four hundred miles away. Where the container the prose gave, or a place the event named
    unambiguously, holds exactly one of the candidates, the guess stops being one.
    """
    near = nearest_name(name, gazetteer, gazetteer.shapes)
    if near is None:
        return set()

    vouching = pinned | {above for gid in container for above in gazetteer.ancestry(gid)}
    if not vouching:
        return set()

    candidates = _outermost({unit.gid for unit in gazetteer.names[near]}, gazetteer)
    inside = {gid for gid in candidates if gazetteer.ancestry(gid) & vouching}

    return inside if len(inside) == 1 else set()


def _corrected_units(name: str, gazetteer: Gazetteer) -> set[str]:
    """Name the units a checked entry says this name stands for."""
    published = gazetteer.corrections.get(match_key(name), ())

    return _outermost(
        {unit.gid for stood_for in published for unit in gazetteer.names.get(stood_for, set())}, gazetteer
    )


def _innermost_container(parent: str | None, pinned: set[str], gazetteer: Gazetteer) -> set[str]:
    """
    Name the units a container reaches, reading it the same way a written place is read.

    Stronger evidence wins over finer granularity: an exact match on the state is taken before a
    misspelling of the district beside it. Within a reading the finest part wins, since the prose
    writes the container finest first.
    """
    parts = container_parts(parent or "")
    readings = (
        lambda part: resolve_place(part, None, gazetteer),
        lambda part: _corrected_units(part, gazetteer),
        lambda part: _corroborated_slip(part, set(), pinned, gazetteer),
        lambda part: _derived_unit(part, gazetteer),
        lambda part: _qualified_unit(part, gazetteer),
    )
    for reading in readings:
        for part in parts:
            if found := reading(part):
                return found

    return set()


def _chosen_by_point(gids: set[str], inside: str | None, gazetteer: Gazetteer) -> set[str]:
    """
    Let a point choose between the candidates a name cannot separate itself.

    The candidates a name reaches are disjoint, so a point sits in at most one of them. Where it
    sits in none, or where no source placed the name, every candidate is kept.
    """
    if inside is None or len(gids) < 2:
        return gids

    return (gids & gazetteer.ancestry(inside)) or gids


def _narrowed(gids: set[str], pinned: set[str], gazetteer: Gazetteer) -> set[str]:
    """Keep the candidates inside something the event already pinned, or all of them if none are."""
    if len(gids) < 2:
        return gids

    return {gid for gid in gids if gazetteer.ancestry(gid) & pinned} or gids


def _outermost(gids: set[str], gazetteer: Gazetteer) -> set[str]:
    """Drop every unit another candidate already contains, leaving what the name can mean at its coarsest."""
    return {gid for gid in gids if not (gazetteer.ancestry(gid) - {gid}) & gids}


def _units_named(name: str, parent: str | None, gazetteer: Gazetteer) -> set[str]:
    """Name the units one written place reaches, narrowed by its stated container where it has one."""
    gids = {unit.gid for unit in gazetteer.names.get(match_key(name), set())}
    if not gids or parent is None:
        return gids

    containers = {unit.gid for part in container_parts(parent) for unit in gazetteer.names.get(match_key(part), set())}
    inside = {gid for gid in gids if (gazetteer.ancestry(gid) - {gid}) & containers}

    return inside or gids


def resolve_place(name: str, parent: str | None, gazetteer: Gazetteer) -> set[str]:
    """
    Name the GADM units a written place refers to, using its stated container to choose between them.

    Where the prose gives a container, only units it holds at any depth survive, which is what
    separates one ``Pitogo`` from the other without a similarity threshold. A container that names
    nothing, or that holds none of the candidates, is ignored rather than treated as a
    contradiction: the prose routinely names a region GADM does not model.

    A name joined by ``and`` is split only where the whole fails, so ``Newfoundland and Labrador``
    stays one unit while ``Aceh and West Sumatra`` becomes two, and every part that names something
    is kept. A dash is weaker evidence of a list and is split only where every part names a unit.
    Neither is split where the name places the event between its parts rather than in them, or along
    a route running through both.

    Parameters
    ----------
    name : str
        The place as written.
    parent : str or None
        The container the prose gave, or None.
    gazetteer : Gazetteer
        The country's units, from :func:`read_gazetteer`.

    Returns
    -------
    gids : set of str
        The GADM identifiers the name reaches, empty where it reaches none.

    Examples
    --------
    The container picks between units sharing a name:

    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.place_names import read_gazetteer, resolve_place

        gazetteer = read_gazetteer("LAO", Path("data"))
        gids = resolve_place("Xaythany", "Vientiane", gazetteer)
    """
    gids = _units_named(name, parent, gazetteer)
    if gids:
        return _outermost(gids, gazetteer)

    if RELATIONAL.match(name) or ROUTE.search(name):
        return set()

    parts = [part.strip() for part in CONJUNCTION.split(name) if part.strip()]
    if len(parts) > 1:
        reached = [found for part in parts if (found := _units_named(part, parent, gazetteer))]
        if reached:
            return _outermost(set().union(*reached), gazetteer)

    return _dashed_units(name, parent, gazetteer)


def _place_one(
    name: str,
    parent: str | None,
    named: set[str],
    located: Mapping[str, str],
    pinned: set[str],
    gazetteer: Gazetteer,
) -> Placement:
    """
    Place one written place, taking the first reading that holds.

    The order is what the readings are worth:

    1. the name itself,
    2. a point,
    3. a correction someone checked,
    4. a slip the event vouches for,
    5. the unit an adjective was built from or a direction qualified,
    6. the container, which claims everything inside it.
    """
    if named:
        return Placement(_narrowed(named, pinned, gazetteer), NAMED)

    if by_point := located.get(name):
        return Placement({by_point}, LOCATED)

    if stands_for := _corrected_units(name, gazetteer):
        return Placement(_narrowed(stands_for, pinned, gazetteer), CORRECTED)

    container = _innermost_container(parent, pinned, gazetteer)
    if inferred := _corroborated_slip(name, container, pinned, gazetteer):
        return Placement(inferred, INFERRED)

    if built_on := _derived_unit(name, gazetteer):
        return Placement(built_on, DERIVED)

    if qualified := _qualified_unit(name, gazetteer):
        return Placement(qualified, QUALIFIED)

    if container:
        return Placement(_narrowed(container, pinned, gazetteer), CONTAINED_BY)

    return Placement(set(), NAMES_NO_UNIT if names_no_unit(name) else NAMED)


def resolve_event_places(
    places: Iterable[tuple[str, str | None]],
    gazetteer: Gazetteer,
    *,
    located: Mapping[str, str] | None = None,
) -> list[Placement]:
    """
    Resolve the places one event names together, letting the unambiguous ones narrow the rest.

    A name reaching several units is settled first by a point, which sits in at most one of them
    because the candidates are disjoint. What is left is settled by the event. Its places share a footprint, so a
    mention that lands on exactly one unit tells the ambiguous mentions beside it where to look. A mention a point has
    just settled speaks with the same authority. Candidates
    outside everything the event has pinned are dropped, and a mention narrowed to nothing keeps its
    candidates.

    A place naming nothing takes the unit a point put it in where one is offered. Failing that it
    falls back to the container the prose wrote it in, so ``Pesisir Selaten (West Sumatra province)``
    becomes the province, which spans fourteen districts at the median. The result records which of
    the two it was reached by. A container naming several places, as ``Wayanad district,
    Kerala state`` does, gives the finest of them that resolves. A name one edit from a published one is taken only
    where the container or an unambiguous sibling holds exactly one candidate. A name that only qualifies a unit by
    direction reaches the whole of it.

    Parameters
    ----------
    places : iterable of tuple
        Each a name as written and the container the prose gave, or None.
    gazetteer : Gazetteer
        The country's units, from :func:`read_gazetteer`.
    located : mapping of str to str, optional
        The unit a point put a written place in, keyed on the name. Default None. A point chooses
        between the units a name reaches, and beats the container where the name reaches none,
        because it names one unit where the container names everything inside it.

    Returns
    -------
    placements : list of Placement
        Where each place put the event, in the order the places were given.

    Examples
    --------
    .. code-block:: python

        from pathlib import Path

        from climate_risk.data.place_names import read_gazetteer, resolve_event_places

        gazetteer = read_gazetteer("LAO", Path("data"))
        placements = resolve_event_places([("Xaythany", "Vientiane"), ("Pakse", None)], gazetteer)
    """
    written = list(places)
    points = located or {}
    resolved = [
        _chosen_by_point(resolve_place(name, parent, gazetteer), points.get(name), gazetteer)
        for name, parent in written
    ]

    pinned: set[str] = set()
    for gids in resolved:
        if len(gids) == 1:
            pinned |= gazetteer.ancestry(next(iter(gids)))

    return [
        _place_one(name, parent, gids, points, pinned, gazetteer)
        for (name, parent), gids in zip(written, resolved, strict=True)
    ]


def successor_state(
    places: Iterable[tuple[str, str | None]],
    iso: str,
    cache_dir: Path,
    *,
    layer: str = GADM_LAYER,
) -> str | None:
    """
    Name the modern country whose gazetteer an event's places belong to.

    A state that left one successor needs no evidence. Where it left several, the successor placing strictly more of the
    written places than any other takes the event. A Soviet flood naming Tashkent is Uzbek. One naming nothing
    recognizable stays unplaced rather than being assigned to the largest successor.

    Parameters
    ----------
    places : iterable of tuple
        Each a name as written and the container the prose gave, or None.
    iso : str
        ISO 3166-1 alpha-3 code EM-DAT filed the event under.
    cache_dir : Path
        Directory the caches live under.
    layer : str, optional
        Layer to read inside the GeoPackage. Default ``GADM_LAYER``.

    Returns
    -------
    successor : str or None
        The successor's ISO code, or None where the state has no successor recorded or its
        successors cannot be told apart.
    """
    successors = SUCCEEDED_BY.get(iso)
    if not successors:
        return None
    if len(successors) == 1:
        return successors[0]

    written = list(places)
    placed: dict[str, int] = {}
    for successor in successors:
        gazetteer = read_gazetteer(successor, cache_dir, layer=layer)
        if gazetteer.names:
            placed[successor] = sum(bool(resolve_place(name, parent, gazetteer)) for name, parent in written)

    best = max(placed.values(), default=0)
    if best == 0:
        return None

    winners = [successor for successor, count in placed.items() if count == best]

    return winners[0] if len(winners) == 1 else None


def names_no_unit(name: str) -> bool:
    """
    Whether a written place can never name an administrative unit.

    A compass point standing in for the whole country, a position between two other places, a sea
    or a river, and a string with no letters in it are all beyond any gazetteer. Counting them as
    coverage failures understates what the resolver reaches, so they are told apart.

    Parameters
    ----------
    name : str
        The place as written.

    Returns
    -------
    no_unit : bool
        True where no administrative unit could answer to this name.
    """
    written = name.strip()

    return bool(
        NOTHING_LEGIBLE.match(written)
        or QUALIFIER.match(written)
        or RELATIONAL.match(written)
        or FEATURE.search(written)
    )
