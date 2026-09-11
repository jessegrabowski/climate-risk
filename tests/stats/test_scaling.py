import numpy as np
import pandas as pd
import pytest

from climate_risk.stats.scaling import standardize


@pytest.fixture
def features():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "name": ["x", "y", "z"]})


def test_standardized_columns_are_added_not_replaced(features):
    """Plotting reads the original scale while the model reads the standardized one."""
    _, transformed = standardize(features, ["a"])

    assert transformed["a"].tolist() == [1.0, 2.0, 3.0]
    assert "a__standardized" in transformed.columns


def test_standardizing_centers_and_scales(features):
    _, transformed = standardize(features, ["a"])

    assert transformed["a__standardized"].mean() == pytest.approx(0.0)
    assert transformed["a__standardized"].std(ddof=0) == pytest.approx(1.0)


def test_unnamed_columns_are_left_alone(features):
    _, transformed = standardize(features, ["a"])

    assert "b__standardized" not in transformed.columns
    assert transformed["name"].tolist() == ["x", "y", "z"]


def test_a_fitted_scaler_applies_the_training_mean_and_scale(features):
    """Held-out data must not recenter on itself, or its coefficients mean something else."""
    scaler, _ = standardize(features, ["a"])

    _, shifted = standardize(features.assign(a=features["a"] + 1.0), ["a"], transformer_fitted=scaler)

    assert shifted["a__standardized"].tolist() == pytest.approx([np.sqrt(1.5) * x for x in (0.0, 1.0, 2.0)])
