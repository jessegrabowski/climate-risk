import pymc as pm
import pytest
import xarray as xr

from climate_risk.sample import sample_or_load

SAMPLE_KWARGS = {"draws": 10, "tune": 10, "chains": 1, "progressbar": False}


@pytest.fixture
def observed_model():
    with pm.Model() as model:
        mu = pm.Normal("mu")
        pm.Normal("obs", mu=mu, sigma=1.0, observed=[0.1, -0.3, 0.7])
    return model


@pytest.mark.slow
def test_saves_and_reloads_without_resampling(tmp_path, observed_model):
    fp = tmp_path / "idata.nc"

    sampled = sample_or_load(fp, model=observed_model, sample_kwargs=SAMPLE_KWARGS)
    assert fp.exists()

    reloaded = sample_or_load(fp, model=observed_model, sample_kwargs=SAMPLE_KWARGS)
    xr.testing.assert_identical(reloaded["posterior"].to_dataset(), sampled["posterior"].to_dataset())


@pytest.mark.slow
def test_save_results_false_leaves_no_file(tmp_path, observed_model):
    fp = tmp_path / "idata.nc"

    sample_or_load(fp, model=observed_model, sample_kwargs=SAMPLE_KWARGS, save_results=False)

    assert not fp.exists()


@pytest.mark.slow
def test_nutpie_draws_survive_the_round_trip(tmp_path, observed_model):
    """Post-processing used to raise for this sampler after sampling had already finished, which threw
    away every draw it had just paid for.
    """
    idata = sample_or_load(
        tmp_path / "idata.nc",
        model=observed_model,
        sample_kwargs=SAMPLE_KWARGS | {"nuts_sampler": "nutpie"},
    )

    assert idata["posterior"].to_dataset().sizes["draw"] == SAMPLE_KWARGS["draws"]
