"""Placebo: geophysical counts (no CO2 pathway, shares EM-DAT reporting drift) through Stage 2 and year-effect models."""

import sys

from pathlib import Path

import arviz as az
import numpy as np

sys.path.insert(0, "/home/jan/work/scratch/yeareffect_run")
from placebo_data import GEOPHYSICAL, placebo_panel

from climate_risk.data_functions.emdat_processing import load_emdat_events
from climate_risk.models.frequency import build_hierarchical_frequency_model, build_year_effect_frequency_model
from climate_risk.sample import sample_or_load

present = set(load_emdat_events(Path("data"))["Disaster Type"].unique())
assert set(GEOPHYSICAL) <= present, f"missing types: {set(GEOPHYSICAL) - present}"
panel = placebo_panel()
kw = {"chains": 4, "draws": 2000, "tune": 1000, "random_seed": 1, "target_accept": 0.95}
for label, builder in [
    ("stage2", build_hierarchical_frequency_model),
    ("yeareffect", build_year_effect_frequency_model),
]:
    idata = sample_or_load(f"data/idata/placebo_geophysical_{label}.nc", model=builder(panel), sample_kwargs=kw)
    div = int(np.asarray(idata.sample_stats.diverging.values).sum())
    names = ["beta"] + (["year_effect_sd"] if label == "yeareffect" else [])
    s = az.summary(idata, var_names=names, ci_prob=0.94, ci_kind="hdi")
    print(f"\n=== PLACEBO geophysical, {label}: divergences {div} ===", flush=True)
    print(s[["mean", "sd", "hdi94_lb", "hdi94_ub", "r_hat", "ess_bulk"]].to_string(), flush=True)
