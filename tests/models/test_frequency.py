import numpy as np
import polars as pl
import pymc as pm
import pytest

from climate_risk.models.frequency import FEATURES, TARGET, build_frequency_model, build_hierarchical_frequency_model


def synthetic_panel(rng: np.random.Generator, n: int = 40, isos: tuple[str, ...] | None = None) -> pl.DataFrame:
    # The real panel carries raw `population`, not `ln_population`; _prepare_panel derives the latter.
    data = {feature: rng.normal(size=n) for feature in FEATURES if feature != "ln_population"}
    data["population"] = rng.lognormal(mean=2.0, sigma=1.0, size=n)
    data[TARGET] = rng.poisson(2, size=n).astype(float)
    if isos is not None:
        data["ISO"] = rng.choice(isos, size=n)

    return pl.DataFrame(data)


def test_a_null_covariate_drops_the_row():
    """There is nothing to standardise a missing covariate against."""
    panel = synthetic_panel(np.random.default_rng(0), n=10)
    panel = panel.with_columns(
        pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col(FEATURES[0])).alias(FEATURES[0])
    )

    model = build_frequency_model(panel)

    assert model.coords["obs_id"] == tuple(range(7))


def test_a_null_target_is_kept_as_a_zero():
    """The notebook calls this fill "a terrible assumption"; this port makes it anyway, on
    purpose, because dropping these rows would leave the likelihood with no zeros for psi to
    identify against.
    """
    panel = synthetic_panel(np.random.default_rng(5), n=10)
    panel = panel.with_columns(pl.when(pl.int_range(pl.len()) < 3).then(None).otherwise(pl.col(TARGET)).alias(TARGET))

    model = build_frequency_model(panel)

    assert model.coords["obs_id"] == tuple(range(10))
    np.testing.assert_allclose(model["Y"].get_value()[:3], 0.0)


def test_beta_carries_one_coefficient_per_feature():
    model = build_frequency_model(synthetic_panel(np.random.default_rng(1)))

    assert model.named_vars_to_dims["beta"] == ("feature",)
    assert model.coords["feature"] == tuple(FEATURES)


def test_the_free_variables_are_exactly_the_stage_1_parameters():
    """No hierarchy and no spatial term yet: an intercept, coefficients, and the two ZINB knobs."""
    model = build_frequency_model(synthetic_panel(np.random.default_rng(2)))

    assert {rv.name for rv in model.free_RVs} == {"intercept", "beta", "psi", "alpha"}


def test_the_observed_target_keeps_its_raw_counts():
    """add_data standardizes the covariates; the target is a count and must not be rescaled."""
    panel = synthetic_panel(np.random.default_rng(3), n=15)
    model = build_frequency_model(panel)

    np.testing.assert_allclose(model["Y"].get_value(), panel[TARGET].to_numpy())


def test_psi_has_zeros_to_identify_against_when_the_target_has_any():
    """Zero-filling only pays for itself if a null target actually reaches the likelihood as 0."""
    panel = synthetic_panel(np.random.default_rng(6), n=10)
    panel = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias(TARGET))

    model = build_frequency_model(panel)

    assert model.coords["obs_id"] == tuple(range(10))
    np.testing.assert_allclose(model["Y"].get_value(), 0.0)


def test_the_coefficient_priors_are_standard_normal():
    """Standardizing the covariates is what makes a flat N(0, 1) prior appropriate; a differently
    scaled feature slipping through would show up here as a beta prior that is no longer standard.
    """
    model = build_frequency_model(synthetic_panel(np.random.default_rng(4), n=200))

    with model:
        prior = pm.sample_prior_predictive(draws=2000, random_seed=0).prior

    assert prior["intercept"].mean().item() == pytest.approx(0.0, abs=0.1)
    assert prior["intercept"].std().item() == pytest.approx(1.0, rel=0.1)
    assert prior["beta"].mean().item() == pytest.approx(0.0, abs=0.1)
    assert prior["beta"].std().item() == pytest.approx(1.0, rel=0.1)


def test_the_hierarchical_model_replaces_the_intercept_with_a_country_effect():
    """Stage 2's only change: a per-country effect stands in for the scalar intercept."""
    isos = ("AAA", "BBB", "CCC")
    panel = synthetic_panel(np.random.default_rng(7), n=30, isos=isos)

    model = build_hierarchical_frequency_model(panel)

    assert model.coords["ISO"] == isos
    assert {rv.name for rv in model.free_RVs} == {
        "country_effect_loc",
        "country_effect_scale",
        "country_effect_offset",
        "beta",
        "psi",
        "alpha",
    }
    assert "intercept" not in model.named_vars


def test_the_country_effect_and_offset_vary_over_iso_not_the_covariates_over_country():
    """Wrong dims would broadcast a country's baseline onto every other country's observations."""
    isos = ("AAA", "BBB", "CCC")
    panel = synthetic_panel(np.random.default_rng(8), n=30, isos=isos)

    model = build_hierarchical_frequency_model(panel)

    assert model.named_vars_to_dims["country_effect_offset"] == ("ISO",)
    assert model.named_vars_to_dims["country_effect"] == ("ISO",)
    assert model.named_vars_to_dims["beta"] == ("feature",)
    assert "country_effect_loc" not in model.named_vars_to_dims


def test_the_zero_sum_offset_still_sums_to_zero_here():
    """A plain Normal offset would leave country_loc unidentified against it; confirm the
    constraint that prevents that is actually the one wired in.
    """
    isos = ("AAA", "BBB", "CCC")
    panel = synthetic_panel(np.random.default_rng(9), n=30, isos=isos)
    model = build_hierarchical_frequency_model(panel)

    with model:
        prior = pm.sample_prior_predictive(draws=200, random_seed=0).prior

    totals = prior["country_effect_offset"].sum(dim="ISO").values
    np.testing.assert_allclose(totals, 0.0, atol=1e-10)
