.. _api_sample:

Sampling and replication
========================

.. currentmodule:: climate_risk

Fitting a model, caching the trace, and resolving the CatDSGE model files. ``.gcn`` is gEconpy's
model format, and each variant is a separate file.

.. autosummary::
    :toctree: generated/

    ~sample.sample_or_load
    ~sample.drop_transformed
    ~replication_data.create_replication_data
    ~dsge.model_files.resolve_gcn_path
