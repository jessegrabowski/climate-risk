.. _api_sample:

Sampling and replication
========================

.. currentmodule:: climate_risk

Fitting a model, caching the trace, and resolving the CatDSGE model files. ``.gcn`` is gEconpy's
model format, and each variant is a separate file.

.. autosummary::
    :toctree: generated/

    ~sample.sample_or_load
    ~replication_data.create_replication_data
    ~replication_data.model_frame
    ~replication_data.load_model_frame
    ~dsge.model_files.resolve_gcn_path
