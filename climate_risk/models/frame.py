import numpy as np
import pandas as pd
import polars as pl

from climate_risk.exceptions import DataValidationError
from climate_risk.geo.raster import ISO_COLUMN, CellGrid
from climate_risk.models.aggregation import Aggregation, build_aggregation, build_aggregation_from_overlaps

EVENT_KEY = "DisNo."


def region_aggregation(grid: CellGrid, cell_weights: np.ndarray, *, label: str = "place") -> Aggregation:
    """
    The one-row operator over every cell, giving the intensity across the whole place.

    Parameters
    ----------
    grid : CellGrid
        The lattice the model integrates over.
    cell_weights : ndarray
        Weight of each cell, in the order of ``grid.cells``.
    label : str, optional
        Row label. Default 'place'.

    Returns
    -------
    Aggregation
        One row spanning every cell.
    """
    if cell_weights.shape[0] != len(grid.cells):
        raise DataValidationError(
            f"{cell_weights.shape[0]} weights against {len(grid.cells)} cells; they index the same grid."
        )

    return build_aggregation(unit_of_cell=[label] * len(grid.cells), weights=cell_weights)


def _cells_of_named_windows(windows: pl.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    """Every cell each event's named units reach, with the share of the cell the window covers."""
    # Empty windows are the country tier and are handled separately, so none reach the explode.
    named = windows.filter(pl.col("gids").list.len() > 0).select(EVENT_KEY, "gids").explode("gids", empty_as_null=False)

    return coverage.merge(
        named.rename({"gids": "gid"}).to_pandas(),
        on="gid",
        how="inner",
    )[[EVENT_KEY, "cell_id", "coverage"]]


def _cells_of_whole_country_windows(windows: pl.DataFrame, grid: CellGrid) -> pd.DataFrame:
    """Every cell of an event's country, for the events no source placed inside one."""
    unplaced = windows.filter(pl.col("gids").list.len() == 0).select(EVENT_KEY, "ISO")
    if unplaced.is_empty():
        return pd.DataFrame(
            {EVENT_KEY: pd.Series(dtype=str), "cell_id": pd.Series(dtype=int), "coverage": pd.Series(dtype=float)}
        )

    cells = grid.cells[["cell_id", ISO_COLUMN]].rename(columns={ISO_COLUMN: "ISO"})

    return (
        unplaced.to_pandas()
        .merge(cells, on="ISO", how="inner")
        .assign(coverage=1.0)[[EVENT_KEY, "cell_id", "coverage"]]
    )


def window_aggregation(
    windows: pl.DataFrame,
    coverage: pd.DataFrame,
    grid: CellGrid,
    *,
    cell_weights: np.ndarray,
) -> Aggregation:
    """
    Build the operator taking a cell-level intensity to the total each event was observed through.

    An event's window is the union of the units its geography names, so a cell held by two of them
    counts once. An event no source placed inside a unit is observed through its whole country.
    Events whose window reaches no cell of the grid are dropped, since the model has nowhere to put
    them; the caller compares the row count against the frame to see how many.

    Parameters
    ----------
    windows : DataFrame
        One row per event, from :func:`~climate_risk.data.model_frame.event_windows`.
    coverage : DataFrame
        One row per cell and unit, from :func:`~climate_risk.geo.raster.assign_cells_to_units`.
    grid : CellGrid
        The lattice the model integrates over.
    cell_weights : ndarray
        Weight of each cell, in the order of ``grid.cells``. Population gives an exposure-weighted
        total, area an unweighted one.

    Returns
    -------
    Aggregation
        One row per event that reaches the grid, labelled by ``DisNo.``.
    """
    if cell_weights.shape[0] != len(grid.cells):
        raise DataValidationError(
            f"{cell_weights.shape[0]} weights against {len(grid.cells)} cells; they index the same grid."
        )

    reached = pd.concat(
        [_cells_of_named_windows(windows, coverage), _cells_of_whole_country_windows(windows, grid)],
        ignore_index=True,
    )
    if reached.empty:
        raise DataValidationError("No event window reaches a cell of this grid.")

    # Units inside one window can nest, so their shares of a cell are a union rather than a sum.
    union = reached.groupby([EVENT_KEY, "cell_id"], as_index=False)["coverage"].sum()
    union["coverage"] = union["coverage"].clip(upper=1.0)

    columns = grid.column_of_cell(union["cell_id"].to_numpy())

    return build_aggregation_from_overlaps(
        unit_of_overlap=union[EVENT_KEY].tolist(),
        cell_of_overlap=columns,
        weights=union["coverage"].to_numpy() * cell_weights[columns],
        n_cells=len(grid.cells),
    )
