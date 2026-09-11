import numpy as np
import polars as pl
import pymc as pm
import pytest

from climate_risk.models.damage import FEATURES, TARGET, build_damage_model


def synthetic_panel(
    rng: np.random.Generator, n: int = 40, isos: tuple[str, ...] = ("AAA", "BBB", "CCC")
) -> pl.DataFrame:
    data = {feature: rng.normal(size=n) for feature in FEATURES}
    data[TARGET] = np.where(rng.random(n) < 0.5, 0.0, rng.lognormal(mean=10.0, sigma=2.0, size=n))
    data["ISO"] = rng.choice(isos, size=n)

    return pl.DataFrame(data)


def test_a_null_covariate_drops_the_row():
    """There is nothing to standardise a missing covariate against."""
    panel = synthetic_panel(np.random.default_rng(0), n=10)
    panel = panel.with_columns(
        pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col(FEATURES[0])).alias(FEATURES[0])
    )

    model = build_damage_model(panel)

    assert model.coords["obs_id"] == tuple(range(7))


def test_a_null_damage_figure_is_kept_as_a_zero():
    """Unlike the frequency target, a null Total_Damage_Adjusted_hydro only ever means no
    qualifying event occurred, so filling it with zero is not the same assumption.
    """
    panel = synthetic_panel(np.random.default_rng(1), n=10)
    panel = panel.with_columns(pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col(TARGET)).alias(TARGET))

    model = build_damage_model(panel)

    assert model.coords["obs_id"] == tuple(range(10))
    np.testing.assert_allclose(model["Y"].get_value()[:3], 0.0)


def test_the_free_variables_are_exactly_the_stage_2_parameters():
    """A hierarchical intercept and covariates, and the Hurdle LogNormal's own psi and sigma."""
    model = build_damage_model(synthetic_panel(np.random.default_rng(2)))

    assert {rv.name for rv in model.free_RVs} == {
        "country_effect_loc",
        "country_effect_scale",
        "country_effect_offset",
        "beta",
        "psi",
        "sigma",
    }


def test_beta_has_one_coefficient_per_feature_and_excludes_the_squared_term():
    model = build_damage_model(synthetic_panel(np.random.default_rng(3)))

    assert model.named_vars_to_dims["beta"] == ("feature",)
    assert model.coords["feature"] == tuple(FEATURES)
    assert "square_ln_gdp_pc" not in FEATURES


def test_the_country_effect_varies_over_iso():
    """Wrong dims would broadcast a country's baseline onto every other country's observations."""
    isos = ("AAA", "BBB", "CCC")
    model = build_damage_model(synthetic_panel(np.random.default_rng(4), n=30, isos=isos))

    assert model.coords["ISO"] == isos
    assert model.named_vars_to_dims["country_effect_offset"] == ("ISO",)
    assert model.named_vars_to_dims["country_effect"] == ("ISO",)


def test_the_zero_sum_offset_sums_to_zero():
    """A plain Normal offset would leave country_loc unidentified against it."""
    model = build_damage_model(synthetic_panel(np.random.default_rng(5), n=30))

    with model:
        prior = pm.sample_prior_predictive(draws=200, random_seed=0).prior

    totals = prior["country_effect_offset"].sum(dim="ISO").values
    np.testing.assert_allclose(totals, 0.0, atol=1e-10)


def test_the_observed_target_keeps_its_raw_damage_values():
    """add_data standardizes the covariates; the target is a dollar figure and must not be rescaled."""
    panel = synthetic_panel(np.random.default_rng(6), n=15)
    model = build_damage_model(panel)

    np.testing.assert_allclose(model["Y"].get_value(), panel[TARGET].to_numpy())


def test_the_coefficient_priors_are_standard_normal():
    model = build_damage_model(synthetic_panel(np.random.default_rng(7), n=200))

    with model:
        prior = pm.sample_prior_predictive(draws=2000, random_seed=0).prior

    assert prior["beta"].mean().item() == pytest.approx(0.0, abs=0.1)
    assert prior["beta"].std().item() == pytest.approx(1.0, rel=0.1)
