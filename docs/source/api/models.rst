.. _api_models:

Model building blocks
=====================

.. currentmodule:: climate_risk.models

Aggregated Poisson likelihood
-----------------------------

Event counts are observed per administrative unit, while the latent intensity lives on the cell
grid. These reconcile the two, which geostatistics calls a change of support.

.. autosummary::
    :toctree: generated/

    ~aggregated_poisson.aggregated_poisson_elbo
    ~aggregated_poisson.expected_intensity
    ~aggregated_poisson.latent_moments
    ~aggregated_poisson.sample_log_intensity

Aggregation matrices
--------------------

.. autosummary::
    :toctree: generated/

    ~aggregation.Aggregation
    ~aggregation.build_aggregation
    ~aggregation.build_aggregation_from_overlaps

Reusable PyMC blocks
--------------------

.. autosummary::
    :toctree: generated/

    ~blocks.add_hierarchical_effect
    ~blocks.add_data
    ~blocks.compute_center
    ~blocks.set_plotting_data
    ~predictions.prediction_to_gpd_df
