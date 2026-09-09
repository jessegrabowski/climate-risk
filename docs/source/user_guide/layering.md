# Layering

The package is arranged so that heavy dependencies stay out of the parts that only move data
around. The rules are short, they hold today, and none of them is visible from inside any single
module, which is why they are written down here.

## The subpackages

- `climate_risk.exceptions`: the error types. Imports nothing from the package.
- `climate_risk.geo`: coordinate systems, grids, distances and raster reduction. Imports only
  `exceptions`.
- `climate_risk.data`: source declarations, fetching, caching and one loader per upstream source.
- `climate_risk.config`: place files and the schema they parse into.
- `climate_risk.data_functions`: the frames built by combining loaders, covering EM-DAT
  processing, shapefiles, rivers, and the country-year panel.
- `climate_risk.models`: the PyMC models and the aggregations they are built from.
- `climate_risk.stats`, `climate_risk.dsge`, `climate_risk.sample`, `climate_risk.plotting`: leaves
  that import nothing else in the package.

## What may not be imported

**`climate_risk.data` and `climate_risk.geo` must not import `pymc`, `arviz`, `sklearn` or
`matplotlib`.** Loading a CSV should not cost a PyTensor compile. Keeping the inference stack out of
the data path is what makes a data-only script start in under a second, and what lets the data tests
run without any of it installed.

**The polars layer must not import geopandas, and the geopandas layer must not import polars.** The
two frame libraries have incompatible ideas about indexes, geometry columns and null handling, and a
module that reaches for both ends up converting implicitly in the middle of a transformation.
Conversions happen at named boundary functions instead, where the cost and the loss are visible.

`climate_risk.data.cache` is the one module that imports both, and that is what it is for: it
declares a `CacheFormat` per frame library, so a loader names the format it wants and never touches
the other library's reader.

## Checking

The rules are conventions rather than machinery. No import-graph test enforces them, and a grep is
enough to check the first:

```
grep -rn "pymc\|arviz\|sklearn\|matplotlib" climate_risk/data/ climate_risk/geo/
```

Anything that matches is a violation.

## Where a dependency belongs instead

A transformation that needs scikit-learn is not a loader. Put the loading in `climate_risk.data`,
returning a plain frame, and the fitting in `climate_risk.models`, taking one. The seam is almost
always in the right place already: loaders return frames, and everything that needs the heavy stack
takes a frame as an argument.

Plotting is the same story from the other end. `climate_risk.plotting` imports nothing from the
package, so a figure is a function of frames rather than a method on a loader.
