.. _api_geo:

Geometry, grids and rasters
===========================

.. currentmodule:: climate_risk.geo

Cell grids
----------

A model observes a continuous field through administrative geometries. These build the cell grid,
measure it, and map cells onto the units that observe them.

.. autosummary::
    :toctree: generated/

    ~raster.CellGrid
    ~raster.build_cell_grid
    ~raster.grid_axes
    ~raster.cell_areas_km2
    ~raster.cell_coverage
    ~raster.assign_cells_to_units
    ~raster.sample_onto_cells
    ~raster.dissolve_place_boundary
    ~grid.create_grid_from_shape

Distance and projection
-----------------------

.. autosummary::
    :toctree: generated/

    ~distance.get_distance_to
    ~crs.to_km
