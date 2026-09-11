.. _api_plotting:

Plotting
========

.. currentmodule:: climate_risk.plotting

Figure setup
------------

``configure_plot_style`` sets the rcParams every figure in the project shares, and ``PALETTE`` names
the colors by role, as ``PALETTE["primary"]``, ``PALETTE["secondary"]`` and ``PALETTE["observed"]``.
Naming a role rather than a color is what keeps one series the same color across two figures.

.. autosummary::
    :toctree: generated/

    configure_plot_style
    panel_grid
    prepare_gridspec_figure

Data and diagnostics
--------------------

.. autosummary::
    :toctree: generated/

    plot_descriptive
    plot_fan
    plot_aggregated_series
    plot_aggregated_series_by_region
    plot_ppc_loopit

Predictions
-----------

.. autosummary::
    :toctree: generated/

    plot_predicted_counts
    plot_predicted_damages
    attach_count_predictions
    attach_damage_predictions
