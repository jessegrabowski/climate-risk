from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from climate_risk.models.climate_economy import OBSERVED, SHOCKS
from climate_risk.models.climate_economy_fit import ObservedSeries, build_priors, growth_accounting, solow_residual

YEARS = pd.to_datetime([str(year) for year in range(2000, 2006)])


def _pwt_row(*, code, year, gdp, capital, employment, population, labour_share=np.nan, depreciation=0.04, ctfp=np.nan):
    return {
        "country_code": code,
        "year": year,
        "pwt_real_gdp": gdp,
        "capital": capital,
        "employment": employment,
        "pwt_population": population,
        "labour_share": labour_share,
        "depreciation": depreciation,
        "tfp_relative_to_usa": ctfp,
    }


def _pwt_panel(*, labour_share, capital_growth, ctfp):
    rows = []
    for code, gdp0, growth in (("USA", 100.0, 1.02), ("ZZZ", 10.0, 1.03)):
        for i, year in enumerate(YEARS):
            rows.append(
                _pwt_row(
                    code=code,
                    year=year,
                    gdp=gdp0 * growth**i,
                    capital=2 * gdp0 * capital_growth**i,
                    employment=5.0 * 1.01**i,
                    population=8.0 * 1.01**i,
                    labour_share=labour_share if code == "ZZZ" else 0.6,
                    ctfp=ctfp if code == "ZZZ" else 1.0,
                )
            )
    return pd.DataFrame(rows)


def _observed_series(*, pwt):
    frontier_tfp = solow_residual(code="USA", panel=pwt)
    observed = pd.DataFrame(
        {name: np.linspace(0.0, 1.0, len(YEARS)) for name in OBSERVED},
        index=YEARS,
    )
    observed["log_tfp_frontier"] = frontier_tfp.reindex(YEARS)
    observed["log_pop"] = np.log(np.linspace(8.0, 8.5, len(YEARS)))

    doublings = pd.Series(np.linspace(0.5, 0.6, len(YEARS)), index=YEARS)

    return ObservedSeries(
        observed=observed,
        doublings=doublings,
        first=YEARS[0],
        last=YEARS[-1],
        pwt_last=YEARS[-1],
        sample_years=(YEARS[-1] - YEARS[0]).days / 365.25,
    )


def test_solow_residual_is_centred_on_its_own_mean():
    pwt = _pwt_panel(labour_share=0.5, capital_growth=1.02, ctfp=0.5)

    log_tfp = solow_residual(code="ZZZ", panel=pwt)

    assert log_tfp.mean() == pytest.approx(0.0, abs=1e-10)


def test_growth_accounting_uses_the_countrys_own_measured_labour_share():
    pwt = _pwt_panel(labour_share=0.55, capital_growth=1.02, ctfp=0.5)
    series = _observed_series(pwt=pwt)

    accounting = growth_accounting(iso="ZZZ", frontier="USA", series=series, pwt=pwt, cache_dir=Path("unused"))

    assert accounting.labour_share == pytest.approx(0.55)
    assert accounting.labour_share_is_measured


def test_growth_accounting_requires_an_override_when_labour_share_is_missing():
    pwt = _pwt_panel(labour_share=np.nan, capital_growth=1.02, ctfp=0.5)
    series = _observed_series(pwt=pwt)

    with pytest.raises(ValueError, match="labour_share_override"):
        growth_accounting(iso="ZZZ", frontier="USA", series=series, pwt=pwt, cache_dir=Path("unused"))


def test_growth_accounting_uses_the_override_when_labour_share_is_missing():
    pwt = _pwt_panel(labour_share=np.nan, capital_growth=1.02, ctfp=0.5)
    series = _observed_series(pwt=pwt)

    accounting = growth_accounting(
        iso="ZZZ", frontier="USA", series=series, pwt=pwt, cache_dir=Path("unused"), labour_share_override=0.487
    )

    assert accounting.labour_share == 0.487
    assert not accounting.labour_share_is_measured


def test_growth_accounting_falls_back_to_zero_gap_drift_without_capital():
    pwt = _pwt_panel(labour_share=0.5, capital_growth=1.02, ctfp=0.5)
    pwt.loc[pwt["country_code"] == "ZZZ", "capital"] = np.nan
    series = _observed_series(pwt=pwt)

    accounting = growth_accounting(iso="ZZZ", frontier="USA", series=series, pwt=pwt, cache_dir=Path("unused"))

    assert accounting.gap_drift == 0.0


def test_growth_accounting_measures_the_gap_from_ctfp_when_present():
    pwt = _pwt_panel(labour_share=0.5, capital_growth=1.02, ctfp=0.5)
    series = _observed_series(pwt=pwt)

    accounting = growth_accounting(iso="ZZZ", frontier="USA", series=series, pwt=pwt, cache_dir=Path("unused"))

    assert accounting.gap_anchor == pytest.approx(-np.log(0.5))
    assert accounting.gap_anchor_is_measured


def test_build_priors_covers_every_shock_observed_series_and_elicited_parameter():
    priors = build_priors()

    expected_keys = set(SHOCKS) | set(OBSERVED) | {"climate_response", "theta", "phi_pop", "phi_emp"}
    assert set(priors) == expected_keys
