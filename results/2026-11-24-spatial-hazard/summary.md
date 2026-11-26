# Space-time areal model with rx5day, Southeast Asia (task 24b, part C)

Scripts: `experiments/spatial_hazard/` (`laos_stationary.py`, `space_time.py`). Notebooks and library models unchanged. Intervals are 89% unless marked.

## C0: the notebook's Laos model reproduces
River slope **-1.01** (90% -1.90 to -0.27); notebook 12: -1.06 (-1.86 to -0.20). 1 divergence, max r_hat 1.03, 36.7 recorded events (notebook 37). So C1 went ahead.

## C1: what changed from the notebook
Atoms are GADM ADM2 units (not 10 km cells). Log intensity per km² = intercept + country offset + river slope × log distance to a major river + rx5day slope × rx5day (ERA5, population-weighted, z-scored within each unit) + shared random-walk year effect. Population-dependent detection as in the notebook. Each event uses its own year. **Left out, because the task's rate did not list them:** the smooth spatial field and the ADM1/ADM2 offsets. River distance is measured from one point inside each unit.

| | Laos | Southeast Asia (9 countries) |
|---|---|---|
| Atoms x years | 142 x 34 = 4,828 | 4,400 x 34 = 149,600 |
| Events (distinct records) | 36 (34) | 1,294 (1,260) |
| Run time per model | 14 to 26 s | 9 to 26 min |
| **rx5day slope** (M2) | **0.47** (0.13 to 0.77) | **0.22** (0.17 to 0.27) |
| River slope without rx5day (M1) | -0.31 (-0.86 to 0.24) | 0.14 (0.06 to 0.20) |
| River slope with rx5day (M2) | -0.32 (-0.90 to 0.14) | 0.14 (0.06 to 0.21) |
| Divergences M0 / M1 / M2 / M3 | 0 / 1 / 4 / 4 | 0 / 2 / 6 / 7 |
| Max r_hat | 1.007 | 1.010 |

M0 = river only, no year effect; M1 = river + year effect; M2 = M1 + rx5day; M3 = rx5day + year effect, no river.

## PSIS-LOO on the same record terms (higher is better; difference from the best, ± standard error)
| Model | Laos | Southeast Asia |
|---|---|---|
| M2 river + rx5day + year | -0.2 ± 0.6 | best |
| M3 rx5day + year | best | -2.7 ± 3.2 |
| M1 river + year | -1.7 ± 2.0 | -18.3 ± 6.2 |
| M0 stationary | -1.5 ± 2.3 | -58.9 ± 11.0 |

The score uses the record terms (where and in which year each event is recorded), not the single total-count term. No Southeast Asia M2 record has Pareto k above 0.7.

## Plain reading
- **rx5day adds information at district level in Southeast Asia:** its slope is above 0, and M2 beats M1 by 18 (about 3 standard errors). A district's wetter-than-usual years have more recorded events.
- **Laos alone cannot decide:** the slope's interval is above 0, but 36 events do not separate the models (all within 1 standard error).
- **River distance:** in Southeast Asia the slope is slightly positive (more events farther from major rivers), the opposite sign to the notebook's Laos result; adding rx5day does not change it, and dropping it costs little fit (2.7 ± 3.2). In C1 Laos it is negative but its interval includes 0. C1 differs from C0 in atom size, in how distance is measured and in having no spatial field; I did not test which difference moves it. Do not quote a river effect from C1.
- Events are 409 Philippines, 300 Indonesia, 220 Vietnam, 142 Thailand, 223 the other five; floods 689, storms 441. I did not fit hazard types or countries separately.
- Not checked: posterior predictive checks, a placebo, and sensitivity to the detection prior. As in the notebook, event totals depend on that prior.
