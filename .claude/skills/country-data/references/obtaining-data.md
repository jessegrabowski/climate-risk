# Obtaining the data

`climate_risk` ships no data. Everything is fetched into the cache directory the caller
names, and most of it is fetched by the library itself on first use. Four sources are not, because their
publishers forbid automated download.

## The four that must be placed by hand

Each is declared as a `ManualSource`. Nothing downloads them. A loader that needs one and does not
find it raises `NotImplementedError` naming the license, the homepage, and the exact path:

```
NotImplementedError: No emdat.xlsx was found at `/data/climate-risk/emdat.xlsx`.
License: Free for non-commercial use with attribution. Redistribution of the database is not
permitted; users must download it themselves after registering.
Obtain it from https://public.emdat.be/ and place it at `/data/climate-risk/emdat.xlsx`.
```

That message is the designed behavior. It is not a gap to route around.

**EM-DAT**: the disaster event record, one row per event with damages and dates. The country-year
panel is built from it.
- Place at `<cache_dir>/emdat.xlsx`
- From https://public.emdat.be/ after registering, exported to `xlsx`

**GADM 4.1**: administrative boundaries worldwide to level 4, used to place events and aggregate
onto units.
- Place at `<cache_dir>/gadm/gadm_410.gpkg`
- From https://gadm.org/download_world.html, the version 4.1 GeoPackage

**Geo-Disasters**: geocoded EM-DAT footprints, 1990-2023, on GAUL 2015 geometries.
- Place at `<cache_dir>/geo_disasters/disaster_subnational_90_23.gpkg`
- From https://doi.org/10.5281/zenodo.15487667

**Penn World Table 10.0**: cross-country national accounts: output, capital stock, employment,
prices.
- Place at `<cache_dir>/pwt100.xlsx`
- From https://www.rug.nl/ggdc/productivity/pwt/

Rename the download to the filename given above. The loaders look for that exact name, and PWT in
particular publishes under a version-stamped name that does not match.

**The licenses restrict what may be redistributed.** EM-DAT and GADM forbid it outright, and
Geo-Disasters' geometries are GAUL, non-commercial with attribution. Do not commit any of these
files, or a derived extract of them, into a repository.

## What downloads itself

Declared as `DataSource` and fetched on first use. No account, no key, nothing to place.

- `load_co2_data`: NOAA CO2, annual mean atmospheric CO2 at Mauna Loa
- `load_ocean_heat_data`: NOAA ocean heat, seasonal ocean heat content 0-700 m
- `load_hadcrut_data`: HadCRUT5, gridded surface temperature anomalies
- `load_gpcc_data`: GPCC, gridded monthly precipitation from gauges
- `load_place_points`: GeoNames, place-name gazetteer per country
- `load_rivers_data`: HydroRIVERS v1.0, river networks with stream order
- `load_shapefile("world", ...)`: World Bank boundaries, country polygons at 1:10m
- `load_shapefile("coastline", ...)`: GSHHG 2.3.7, global coastlines
- `population_on_cells`: GHS-POP R2023A, population rasters at 30 arcsec, one per five-year epoch

GPCC and GHS-POP are the large ones: GPCC arrives an archive per decade plus one per month after
2020, and GHS-POP is roughly a gigabyte per epoch. A country panel needs GPCC and does not need
GHS-POP.

## Services queried live

World Bank Indicators (`load_wb_data`), FRED (`load_fred_data`), IMF IMTS
(`load_partner_activity`) and Nominatim (`osm_geocoder`) answer a query rather than serving a file.
Responses are cached like anything else, so a warm run does not call them again. Nominatim's usage
policy caps requests at one per second and the loader honors it, so geocoding a country's events for
the first time takes as long as it takes.

## Shipped in the wheel

One file is redistributed with the package: the IPCC AR6 SYR CSB.2 Figure 1(a) emissions workbook
read by `process_ipcc_scenarios`. It is CC BY 4.0, and it is vendored because SEDAC, which published
it, was decommissioned in June 2025 and no successor serves it to an unattended client.
