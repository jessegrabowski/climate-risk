import numpy as np
import pandas as pd
import pymc as pm
import pytest

from climate_risk.models.blocks import add_data, add_hierarchical_effect, compute_center

COORDS = {"ISO": ["AAA", "BBB", "CCC"], "obs_idx": [0, 1, 2, 3], "feature": ["precip", "temp"]}


@pytest.fixture
def observations():
    return pd.DataFrame(
        {
            "precip": [1.0, 2.0, 3.0, 4.0],
            "temp": [10.0, 20.0, 30.0, 40.0],
            "damage": [0.5, 1.5, 2.5, 3.5],
        }
    )


def prior_draws(build, draws=50):
    with pm.Model(coords=COORDS) as model:
        build()
        return pm.sample_prior_predictive(draws=draws, random_seed=0).prior, model


def test_a_hierarchical_effect_needs_the_dimension_it_varies_over():
    with pm.Model(coords=COORDS):
        with pytest.raises(ValueError, match="group_dim must be provided"):
            add_hierarchical_effect(name="country")


def test_the_effect_is_its_location_plus_the_scaled_offset():
    """The non-centered parameterisation is the whole point of the helper."""
    prior, _ = prior_draws(lambda: add_hierarchical_effect(name="country", group_dim="ISO"))

    expected = prior["country_effect_loc"] + prior["country_effect_scale"] * prior["country_effect_offset"]

    np.testing.assert_allclose(prior["country_effect"].values, expected.values)


def test_the_effect_varies_over_the_group_dimension():
    """Wrong dims broadcast the effect to the wrong observations without erroring."""
    _, model = prior_draws(lambda: add_hierarchical_effect(name="country", group_dim="ISO"))

    assert model.named_vars_to_dims["country_effect_offset"] == ("ISO",)
    assert model.named_vars_to_dims["country_effect"] == ("ISO",)
    assert "country_effect_loc" not in model.named_vars_to_dims


def test_zero_sum_offsets_sum_to_zero():
    """Without the constraint the group mean and the offsets are not jointly identified."""
    prior, _ = prior_draws(lambda: add_hierarchical_effect(name="country", group_dim="ISO", use_zerosum_offset=True))

    totals = prior["country_effect_offset"].sum(dim="ISO").values

    np.testing.assert_allclose(totals, 0.0, atol=1e-10)


def test_offsets_are_unconstrained_by_default():
    """The zero-sum constraint changes the prior, so it must stay opt-in."""
    prior, _ = prior_draws(lambda: add_hierarchical_effect(name="country", group_dim="ISO"))

    totals = prior["country_effect_offset"].sum(dim="ISO").values

    assert np.abs(totals).max() > 1e-6


def test_the_prior_parameters_reach_their_distributions():
    """Nothing else pins these five, so a swapped Gamma alpha and beta would go unnoticed."""
    prior, _ = prior_draws(
        lambda: add_hierarchical_effect(
            name="country",
            group_dim="ISO",
            loc_mu=100.0,
            loc_sigma=0.1,
            scale_alpha=200.0,
            scale_beta=100.0,
        ),
        draws=2000,
    )

    assert prior["country_effect_loc"].mean() == pytest.approx(100.0, abs=0.05)
    assert prior["country_effect_loc"].std() == pytest.approx(0.1, rel=0.1)
    assert prior["country_effect_scale"].mean() == pytest.approx(2.0, rel=0.05)


def test_the_effect_name_prefixes_every_variable_it_creates():
    """Downstream reads the posterior by name, so the prefix is the contract."""
    _, model = prior_draws(lambda: add_hierarchical_effect(name="rainfall", group_dim="ISO"))

    created = {rv.name for rv in model.free_RVs} | {rv.name for rv in model.deterministics}

    assert created == {"rainfall_effect_loc", "rainfall_effect_scale", "rainfall_effect_offset", "rainfall_effect"}


def test_data_without_a_target_returns_only_the_features(observations):
    with pm.Model(coords=COORDS) as model:
        result = add_data(["precip", "temp"], observations, dims=["obs_idx", "feature"])

    assert not isinstance(result, tuple)
    assert "Y" not in model.named_vars


def test_the_target_takes_only_the_batch_dimension(observations):
    """The features are two-dimensional and the target is one; reusing both dims fails to build."""
    with pm.Model(coords=COORDS) as model:
        add_data(["precip", "temp"], observations, target="damage", dims=["obs_idx", "feature"])

    assert tuple(model.named_vars_to_dims["X"]) == ("obs_idx", "feature")
    assert tuple(model.named_vars_to_dims["Y"]) == ("obs_idx",)


def test_named_data_is_suffixed(observations):
    with pm.Model(coords=COORDS) as model:
        add_data(["precip", "temp"], observations, target="damage", name="train", dims=["obs_idx", "feature"])

    assert {"X_train", "Y_train"} <= set(model.named_vars)


def test_the_center_is_the_midpoint_of_the_extremes():
    assert compute_center(np.array([[0.0, -5.0], [10.0, 5.0], [4.0, 1.0]])).tolist() == [5.0, 0.0]
