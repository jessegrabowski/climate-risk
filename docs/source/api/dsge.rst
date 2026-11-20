.. _api_dsge:

CatDSGE
=======

.. currentmodule:: climate_risk.dsge.catdsge

The small open economy DSGE model with a capital-destruction shock, calibrated and estimated on the
India panel from :func:`~climate_risk.data_functions.india_panel.load_india_quarterly_panel`. The
figures these notebooks draw are in :ref:`api_plotting`.

Calibration
-----------

.. autosummary::
    :toctree: generated/

    configure_session
    compute_data_targets
    model_ratios
    calibrate
    calibrate_india

Estimation
----------

.. autosummary::
    :toctree: generated/

    growth_observation_equation
    configure_statespace
    load_estimation_statespace
    load_simulation_statespace
    build_observed
    noise_sd_scales
    error_scale
    fixed_parameters
    wage_calvo_probability
    build_pymc_model

Simulation and welfare
----------------------

.. autosummary::
    :toctree: generated/

    statespace_trace
    simulation_trace
    to_display_units
    responses_in_display_units
    response_quantiles
    bind_to_model
    build_welfare_model
    median_and_hdi
