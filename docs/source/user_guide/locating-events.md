# Locating events

EM-DAT gives most events a country and nothing finer. A minority carry administrative units EM-DAT
coded itself. The rest carry a free-text `Location` column, or nothing at all. This is how that text
becomes GADM units.

## The path a location takes

```
"Muang Nakhon Si Thammarat, Hua Sai districts (Nakhon Si Thammarat province)"
        |
        |  named_places        climate_risk/data_functions/emdat_processing.py
        v
[("Muang Nakhon Si Thammarat", "Nakhon Si Thammarat province"), ("Hua Sai districts", ...)]
        |
        |  resolve_place       climate_risk/data/place_names.py
        v
{"THA.40.5_1"}, {"THA.40.3_1"}
```

`named_places` splits on commas and semicolons outside parentheses, and treats a parenthesised
group as the container of every place since the last one. It never splits on `and`: 75 GADM units
are named like `Newfoundland and Labrador`.

`resolve_place` matches a name against every name GADM publishes a unit under, having stripped the
noun saying what kind of unit it is. It resolves ambiguity three ways, in order: a unit contained by
another candidate is dropped, the container from the prose narrows what is left, and
`resolve_event_places` lets the places an event names unambiguously narrow the ones it does not.
Where the whole string reaches nothing and every part of an `and` does, the parts are taken instead.

What is left over goes to a gazetteer of points. `geonames_geocoder` answers with a longitude and
latitude, and the point is placed in whichever GADM unit contains it.

Reproducing this from a fresh clone, and scoring the geocoder against GADM, is in
[the contributor guide](../dev/contributing.rst).
