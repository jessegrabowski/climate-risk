# Design notes

Four things in this package are easy to read as accidents. Each is a decision, and each is written
down here with the reason behind it, because a reader who assumes otherwise changes the wrong thing.

## Disaster classes are this project's vocabulary, not EM-DAT's

EM-DAT publishes a `Disaster Type` per event: `Storm`, `Flood`, `Drought`, `Wildfire`,
`Extreme temperature`, `Mass movement (wet)`. It also publishes its own subgroups, which this
project does not use.

`DISASTER_CLASSES` maps those types onto two classes of its own, `Hydrometeorological` and
`Climatological`, and `load_emdat_events` writes the result as a `disaster_class` column. The
grouping is a modelling decision: the two classes are the split the damage regressions are specified
over, and it does not correspond to any partition EM-DAT ships.

Two consequences. A type EM-DAT adds is not silently absorbed --- `replace_strict` maps an unlisted
type to null rather than guessing. And code should refer to a class through the constant,
`HYDROMETEOROLOGICAL` or `CLIMATOLOGICAL`, rather than repeating the string, so the vocabulary has
one definition.

## The cache directory is an argument, always

There is no default `cache_dir`, no environment variable, and no search for a project root. The
package reads no environment variables at all.

The absence is deliberate. A path discovered from ambient state would make every function's
behavior depend on something the caller cannot see in the call.
[The cache directory](../get_started/cache-directory.rst) has the contract and the reasoning.

## Constants live next to their consumer

There is no `constants.py`. A URL sits in the module of the loader that fetches it, a column mapping
sits in the module that renames the columns, and a threshold sits beside the comparison that uses it.

A central constants module collects values whose only shared property is being constant. Reading one
then means reading a file that has nothing to do with the code you were reading, and changing one
means grepping to find out who was affected. Keeping the value next to its single consumer makes
both questions local.

The corollary: a constant genuinely shared by two modules goes in the one that owns the concept, and
the other imports it. `GEOGRAPHIC_CRS` lives in `climate_risk.geo.crs` because projection is that
module's subject, not because it is a constant.

## Configuration is data, not code

A country is a TOML file. Nothing in `climate_risk/` names a country outside
`climate_risk/config/places/`, and there is no registry to update when a file is added -- the
directory is globbed.

The test of this is the one in [adding a country](adding-a-country.md): if you find yourself editing
a loader to make a new country work, the loader is wrong. A loader that branches on a country code
has moved configuration into code, where the next country cannot reach it.
