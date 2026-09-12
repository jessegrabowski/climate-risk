import hashlib
import inspect
import logging
import os
import re

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl

from climate_risk.utils import _log_maybe

_log = logging.getLogger(__name__)

PARAMETER_SEPARATOR = "__"
VALUE_SEPARATOR = "="

# A key becomes a filename, so a value carrying a path separator would write outside the cache, and
# one carrying a key separator would make two different parameter sets collide on a single entry. A
# dot collides the same way: the suffix is taken from the last one, so `4.1` and `4.2` name one file.
FORBIDDEN_IN_KEY = ("/", "\\", ".", PARAMETER_SEPARATOR, VALUE_SEPARATOR)

# The tail of the repr Python gives an object that defines none: the address moves every process.
DEFAULT_REPR_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+>")


@dataclass(frozen=True, slots=True)
class CacheFormat[T]:
    """
    A matched reader and writer. Declaring them together is what stops the two drifting apart.

    Parameters
    ----------
    suffix : str
        File extension the artifact is stored under.
    read : callable
        Reads the artifact back from a path.
    write : callable
        Writes the artifact to a path.
    atomic : bool
        Whether ``write`` can be sent to a temporary path and moved into place. False for formats
        carrying sidecar files, which a single move would leave behind.
    """

    suffix: str
    read: Callable[[Path], T]
    write: Callable[[T, Path], None]
    atomic: bool


def pandas_parquet() -> CacheFormat[pd.DataFrame]:
    """A parquet round-trip. The file carries dtypes and the index, so nothing is restored by hand."""
    return CacheFormat(
        suffix=".parquet",
        read=pd.read_parquet,
        write=lambda frame, path: frame.to_parquet(path),
        atomic=True,
    )


def polars_parquet() -> CacheFormat[pl.DataFrame]:
    """A parquet round-trip for a tidy frame. The file carries dtypes, so nothing is restored by hand."""
    return CacheFormat(
        suffix=".parquet",
        read=pl.read_parquet,
        write=lambda frame, path: frame.write_parquet(path),
        atomic=True,
    )


def geo_parquet() -> CacheFormat[gpd.GeoDataFrame]:
    """A GeoParquet round-trip. Column names, dtypes and CRS all survive, and it is a single file."""
    return CacheFormat(
        suffix=".parquet",
        read=gpd.read_parquet,
        write=lambda frame, path: frame.to_parquet(path),
        atomic=True,
    )


def cache_key(name: str, params: Mapping[str, object] | None = None) -> str:
    """
    Build the cache filename stem for ``name`` under ``params``.

    Parameters are sorted, so the same set always yields the same key however the caller ordered it.

    Parameters
    ----------
    name : str
        Logical name of the cached artifact.
    params : mapping of str to object, optional
        Values distinguishing this entry from others under the same name. Each must be a string,
        integer or boolean. Default None, meaning the name alone identifies the entry.

    Returns
    -------
    key : str
        A filename stem, such as ``points__grid_size=400__region=sea``.
    """
    parts = [_key_part(name, "name")]
    for key, value in sorted((params or {}).items()):
        parts.append(f"{_key_part(key, 'parameter')}{VALUE_SEPARATOR}{_key_part(value, f'value of {key}')}")

    return PARAMETER_SEPARATOR.join(parts)


def _key_part(value: object, described_as: str) -> str:
    if not isinstance(value, str | int):
        raise ValueError(f"{described_as} must be a string, integer or boolean, got {value!r}")

    text = str(value)
    if not text:
        raise ValueError(f"{described_as} must not be empty")

    forbidden = [character for character in FORBIDDEN_IN_KEY if character in text]
    if forbidden:
        raise ValueError(f"{described_as} must not contain {forbidden}, got {text!r}")

    return text


def builder_fingerprint(builder: Callable[..., object], *rules: object) -> str:
    """
    Fingerprint how an artifact is built, so a change to the builder turns over what it cached.

    Parameters record what was asked for and nothing records how the answer was produced, so an
    entry built under rules that have since changed reads back as a hit. Reformatting the builder or
    editing a comment inside it turns the cache over as surely as changing a rule does.

    Parameters
    ----------
    builder : callable
        The function that produces the artifact. Only its source is read, so its arity does not
        matter and a transform taking arguments serves as well as a closure taking none.
    *rules : object
        Values the builder reads that its own source does not show, such as a table it consults, the
        columns it selects, or a function it calls. A function or class is read for its source and
        everything else for its repr, which must be the same in every process.

    Returns
    -------
    fingerprint : str
        A short digest of the builder's source and the rules it was given.
    """
    declared = inspect.getsource(builder) + "".join(_rule_text(rule) for rule in rules)

    return hashlib.sha256(declared.encode()).hexdigest()[:12]


def _rule_text(rule: object) -> str:
    """Return what a rule contributes to the digest, taken from its source where it has one."""
    if inspect.isroutine(rule) or inspect.isclass(rule):
        return inspect.getsource(rule)

    text = repr(rule)
    if DEFAULT_REPR_ADDRESS.search(text):
        raise ValueError(f"a rule whose repr carries an object address cannot key a cache entry: {text}")

    return text


def cached[T](
    cache_dir: Path,
    name: str,
    builder: Callable[[], T],
    fmt: CacheFormat[T],
    *,
    params: Mapping[str, object] | None = None,
    force: bool = False,
    verbose: bool = False,
) -> T:
    """
    Return the cached artifact for ``name``, building and storing it if it is not already there.

    One implementation of check-then-build-then-write, so a call site cannot spell the key one way
    when it writes and another when it reads.

    Parameters
    ----------
    cache_dir : Path
        Directory to store the artifact in. Created if absent.
    name : str
        Logical name of the artifact.
    builder : callable
        Produces the artifact. Called only on a miss.
    fmt : CacheFormat
        How to read and write it.
    params : mapping of str to object, optional
        Values distinguishing this entry from others under the same name. Default None.
    verbose : bool, optional
        Log whether the artifact was read or built. Default False: one call can touch a dozen
        artifacts, and a dozen lines buries whatever the caller meant to show.
    force : bool, optional
        Rebuild even when the artifact is already cached. Default False.

    Returns
    -------
    artifact : T
        The artifact, read from the cache or freshly built.
    """
    path = (cache_dir / cache_key(name, params)).with_suffix(fmt.suffix)

    if path.exists() and not force:
        _log_maybe(f"Loading cached {name} from {path}", logger=_log, verbose=verbose)
        return fmt.read(path)

    cache_dir.mkdir(parents=True, exist_ok=True)

    _log_maybe(f"Building {name}", logger=_log, verbose=verbose)
    artifact = builder()

    if fmt.atomic:
        partial = path.with_suffix(path.suffix + ".part")
        fmt.write(artifact, partial)
        os.replace(partial, path)
    else:
        fmt.write(artifact, path)

    return artifact
