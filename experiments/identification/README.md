# CO2 identification tests: names

The Cowork task (`tasks/2026-11-21-climate-identification-tests.md`) numbers the tests 1-4. Each one has a
short name too, and the names (not the numbers) label files, fit folders and the `test` column of
`results/2026-11-21-identification/fits.csv`.

| # | Name | Question | Status |
|---|---|---|---|
| 1 | `test1_time_driver` | Does swapping `co2` for another common time driver (none, or a global temperature shock) give a usable climate link, with the year effect in? | not run |
| 2 | `test2_geophysical_control` | Does `co2` act on hydro events but not on geophysical ones, when both share one year effect? | not run |
| 3 | `test3_high_impact` | Does `co2` survive when only deadly hydro events (more than 100 deaths) are counted? | done |
| 4 | `test4_post_2000` | Does `co2` survive when the sample is cut to 2000-2023? | done |

Fits are in `data/idata/identification/<name>_<plain|year_effect>.nc` (gitignored).
