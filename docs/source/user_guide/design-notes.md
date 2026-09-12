# Design notes

The decisions below are easy to read as accidents. Each is written down with the reason behind it,
because a reader who assumes otherwise changes the wrong thing.

## Disaster classes are this project's vocabulary, not EM-DAT's

EM-DAT publishes a `Disaster Type` per event: `Storm`, `Flood`, `Drought`, `Wildfire`,
`Extreme temperature`, `Mass movement (wet)`. It also publishes its own subgroups, which this
project does not use.

`DISASTER_CLASSES` maps those types onto two classes of its own, `Hydrometeorological` and
`Climatological`, and `load_emdat_events` writes the result as a `disaster_class` column. The
grouping is a modeling decision: the two classes are the split the damage regressions are specified
over, and it does not correspond to any partition EM-DAT ships.

`DISASTER_CLASSES` is the only place the vocabulary is written down. `DISASTER_TYPES` is its keys, so
the types the panel counts are exactly the types that have a class, and `types_in_class` reads it the
other way round for the count columns. A type listed in one of those and not the others is not
possible, which is what keeps the count columns and the damage columns meaning the same thing.

A type EM-DAT adds is not silently absorbed, because `replace_strict` maps an unlisted type to null
rather than guessing. Code should refer to a class through the constant, `HYDROMETEOROLOGICAL` or
`CLIMATOLOGICAL`, rather than repeating the string.

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

## Two labour shares, under two names

Penn World Table publishes `labour_share` and the ILO publishes `labour_income_share`. Both are
labour compensation as a share of GDP, and the panel carries them as separate columns rather than
one filled column.

They are built from different data. Penn World Table works from national accounts with an imputation
for the self-employed, and covers 1950 onward. The ILO estimates labour income from harmonized
survey microdata, and covers 2004 onward. Where both exist they disagree by up to eight points of
GDP, and the sign of the gap is not consistent across countries, so neither is a correction of the
other.

Eight points is not a rounding difference. A Cobb-Douglas decomposition reads the labour share as
:math:`\theta` and the capital share as :math:`1 - \theta`, so for Laos the two sources put capital
at 0.60 and 0.52. Filling one column from whichever source has a value would put two countries in
one regression on two definitions, with nothing downstream able to tell which.

Choosing between them is a modeling decision and belongs to the model. Penn World Table has no
`labsh` at all for Nepal, Bangladesh, Pakistan, Bhutan or Myanmar, so a model covering those
countries has to read the ILO series and say so.

## A damage total is null when nobody priced the events

A country-year can hold three different things, and the panel keeps them apart. No qualifying event
gives null. Events that happened but that nobody attached a figure to also gives null. Only a figure
somebody actually recorded gives a number.

Collapsing the middle case to zero is the tempting mistake, and it is what a plain sum does, because
polars totals an all-null group to zero. Damage is the worst column for it: EM-DAT prices well under
half of the country-years that have events, so the zeros would outnumber the real figures and every
one of them would read as a disaster that cost nothing.

The panel publishes `Total_Damage_Adjusted_hydro` and `Total_Damage_Adjusted_clim` separately, in
the thousands of US dollars EM-DAT reports, and combines neither. Adding the two classes means
deciding what a class with no events contributes, and rescaling the money means picking the units one
model's priors were calibrated against. Both are choices for the model, so the panel leaves them to it.

## Money is stated in constant 2015 US dollars

A money column is comparable across years only once the price level is divided out of it.
`climate_risk.data.deflate` does that against the US consumer price index, and `BASE_YEAR` there is
2015.

The year is 2015 because the covariates already are. `NY.GDP.PCAP.KD` and the other World Bank
constant-price indicators are 2015 US dollars, and the partner-activity index is based there too.
Restating damages in another year puts the two sides of a regression in different units, and nothing
downstream can detect that.

Two functions apply the factors, and choosing between them is where this goes wrong. `deflate` moves
each row by the factor for the year beside it, which is right for an amount measured in the dollars
of that year. `rebase` moves every row by one factor, which is right for an amount already held
constant in a single year's dollars. EM-DAT's adjusted damage columns are the second kind, because
the year in the row is the event's date rather than the date of the money.

## Configuration is data, not code

A country is a TOML file. Nothing in `climate_risk/` names a country outside
`climate_risk/config/places/`, and there is no registry to update when a file is added, because the
directory is globbed.

The test of this is the one in [adding a country](adding-a-country.md): if you find yourself editing
a loader to make a new country work, the loader is wrong. A loader that branches on a country code
has moved configuration into code, where the next country cannot reach it.
