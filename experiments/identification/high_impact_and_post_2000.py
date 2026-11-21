"""
Test 3 (high_impact) and Test 4 (post_2000) of the CO2 identification task: does ``co2`` survive a high-impact target and a post-2000 sample?

Each test fits Stage 2 (hierarchical country intercept) without and then with the shared year effect,
and appends one row per fit to ``results/2026-11-21-identification/fits.csv`` straight after sampling.

Usage (from the repo root): ``pixi run python experiments/identification/high_impact_and_post_2000.py``
"""

import json

import polars as pl

from common import CACHE, FITS, SAMPLE_KWARGS, already_done, record, summarise

from climate_risk.data_functions.emdat_processing import load_emdat_events
from climate_risk.models.frequency import (
    TARGET,
    _prepare_panel,
    build_hierarchical_frequency_model,
    build_year_effect_frequency_model,
)
from climate_risk.models.identification import with_high_impact_hydro
from climate_risk.replication_data import create_replication_data
from climate_risk.sample import sample_or_load

BUILDERS = {"plain": build_hierarchical_frequency_model, "year_effect": build_year_effect_frequency_model}


def main() -> None:
    panel = create_replication_data(CACHE)
    events = load_emdat_events(CACHE)

    samples = {
        ("test3_high_impact", "hydro, deaths > 100"): with_high_impact_hydro(panel, events, min_deaths=100),
        ("test4_post_2000", "2000-2023"): panel.filter(pl.col("date").dt.year() >= 2000),
    }
    done = already_done()

    for (test, sample), data in samples.items():
        for model_label, builder in BUILDERS.items():
            if (test, model_label) in done:
                continue
            idata = sample_or_load(
                str(FITS / f"{test}_{model_label}.nc"), model=builder(data), sample_kwargs=SAMPLE_KWARGS
            )
            prepared = _prepare_panel(data)
            row = summarise(
                test, sample, model_label, "co2", idata, rows=len(prepared), events=float(prepared[TARGET].sum())
            )
            record(row)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
