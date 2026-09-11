from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytensor.tensor as pt

from pytensor.tensor import TensorVariable
from scipy import sparse

from climate_risk.exceptions import DataValidationError


@dataclass(frozen=True, slots=True)
class Aggregation:
    r"""
    The linear operator taking a cell-level field to the polygon totals it is observed through.

    .. math::

        y_A = \sum_{i \in A} w_i \lambda_i

    ``units`` fixes the row order, which the model reads by position: it must agree with the order
    the observations arrive in.

    Parameters
    ----------
    units : tuple of str
        Unit labels, in row order.
    rows : ndarray
        Unit index of each contributing cell, one entry per nonzero.
    columns : ndarray
        Cell index of each contributing cell, one entry per nonzero.
    weights : ndarray
        Weight of each contributing cell, one entry per nonzero.
    n_cells : int
        Width of the operator, counting cells that contribute to no unit.
    """

    # The triplet is primary because a scipy matrix cannot multiply a pytensor variable.
    units: tuple[str, ...]
    rows: np.ndarray
    columns: np.ndarray
    weights: np.ndarray
    n_cells: int

    @property
    def n_units(self) -> int:
        """Rows of the operator, and the length of an aggregated vector."""
        return len(self.units)

    @property
    def matrix(self) -> sparse.csr_array:
        """The operator as a sparse matrix, shape ``(n_units, n_cells)``."""
        return sparse.csr_array(
            (self.weights, (self.rows, self.columns)),
            shape=(self.n_units, self.n_cells),
        )

    def aggregate(self, cell_values: np.ndarray) -> np.ndarray:
        """
        Sum a cell-level field to unit totals, outside a graph.

        Use this on drawn or observed arrays. Inside a model, use :meth:`aggregate_symbolic`.

        Parameters
        ----------
        cell_values : ndarray
            Shape ``(n_cells,)`` for one field, or ``(n_cells, k)`` to aggregate ``k`` of them at
            once, which is how posterior draws are carried through.

        Returns
        -------
        aggregated : ndarray
            Shape ``(n_units,)`` or ``(n_units, k)``, in the row order of ``units``.
        """
        self._check_width(cell_values.shape[0])

        return np.asarray(self.matrix @ cell_values)

    def aggregate_symbolic(self, cell_values: TensorVariable) -> TensorVariable:
        """
        Sum a cell-level field to unit totals, as a graph.

        Cells sharing a unit accumulate, and the reverse pass is the operator's transpose.

        Parameters
        ----------
        cell_values : TensorVariable
            Shape ``(n_cells,)`` for one field, or ``(n_cells, k)`` to aggregate ``k`` of them at
            once, each column independently.

        Returns
        -------
        aggregated : TensorVariable
            Shape ``(n_units,)`` or ``(n_units, k)``, in the row order of ``units``.
        """
        weights = pt.as_tensor_variable(self.weights)
        if cell_values.ndim == 2:
            weights = weights[:, None]
            base = pt.zeros((self.n_units, cell_values.shape[1]))
        else:
            base = pt.zeros((self.n_units,))

        contributions = weights * cell_values[self.columns]

        totals: TensorVariable = pt.inc_subtensor(base[self.rows], contributions)

        return totals

    def _check_width(self, width: int) -> None:
        if width != self.n_cells:
            raise DataValidationError(f"The field has {width} cells but the operator was built for {self.n_cells}.")


def build_aggregation(
    *,
    unit_of_cell: Sequence[str | None],
    weights: np.ndarray,
    units: Sequence[str] | None = None,
) -> Aggregation:
    """
    Build the cell-to-unit aggregation operator.

    Parameters
    ----------
    unit_of_cell : sequence of str or None
        The unit each cell belongs to. ``None`` marks a cell inside the gridded extent but outside
        every unit, which is dropped: it contributes to no total.
    weights : ndarray
        Each cell's weight in its unit, one per entry of ``unit_of_cell``. Cell area gives an
        integral over the polygon. Population gives an exposure-weighted one.
    units : sequence of str, optional
        Row order of the operator. Every assigned label must appear, and no label twice. Default
        None, which orders the labels that appear in ``unit_of_cell``, sorted.

    Returns
    -------
    aggregation : Aggregation
        The operator and its row labels.

    Examples
    --------
    Two cells in one unit, one in another:

    .. code-block:: python

        import numpy as np

        from climate_risk.models.aggregation import build_aggregation

        aggregation = build_aggregation(
            unit_of_cell=["LAO.1_1", "LAO.1_1", "LAO.2_1"],
            weights=np.array([1.0, 1.0, 2.0]),
        )
    """
    weights = _validated_weights(weights)
    if len(unit_of_cell) != weights.shape[0]:
        raise DataValidationError(
            f"{len(unit_of_cell)} cell assignments against {weights.shape[0]} weights; they index the same cells."
        )

    assigned = np.array([label is not None for label in unit_of_cell], dtype=bool)
    labels = [label for label in unit_of_cell if label is not None]
    row_order, rows = _rows_for_units(labels, units)

    return Aggregation(
        units=row_order,
        rows=rows,
        columns=np.flatnonzero(assigned),
        weights=weights[assigned],
        n_cells=weights.shape[0],
    )


def build_aggregation_from_overlaps(
    *,
    unit_of_overlap: Sequence[str],
    cell_of_overlap: np.ndarray,
    weights: np.ndarray,
    n_cells: int,
    units: Sequence[str] | None = None,
) -> Aggregation:
    """
    Build the operator from cell-unit overlaps, where a cell may belong to several units.

    Parameters
    ----------
    unit_of_overlap : sequence of str
        The unit each overlap belongs to, one entry per overlap.
    cell_of_overlap : ndarray
        The cell each overlap belongs to, indexing the grid, one entry per overlap.
    weights : ndarray
        The weight each overlap contributes, one entry per overlap. Overlapping area gives an
        integral over the polygon. Overlapping population gives an exposure-weighted one.
    n_cells : int
        Width of the operator. Cells named in no overlap contribute to no total.
    units : sequence of str, optional
        Row order of the operator. Every named unit must appear, and no unit twice. Default None,
        which orders the units that appear in ``unit_of_overlap``, sorted.

    Returns
    -------
    aggregation : Aggregation
        The operator and its row labels.
    """
    weights = _validated_weights(weights)
    cells = np.asarray(cell_of_overlap, dtype=int)
    lengths = {len(unit_of_overlap), cells.shape[0], weights.shape[0]}
    if len(lengths) != 1:
        raise DataValidationError(
            f"{len(unit_of_overlap)} units, {cells.shape[0]} cells and {weights.shape[0]} weights; one of each "
            f"describes one overlap."
        )

    out_of_range = cells[(cells < 0) | (cells >= n_cells)]
    if out_of_range.size:
        raise DataValidationError(
            f"Overlaps name cells outside the grid of {n_cells}: {sorted(set(out_of_range.tolist()))[:5]}."
        )

    row_order, rows = _rows_for_units(list(unit_of_overlap), units)

    # A repeated pair is summed silently by the sparse matrix, so the same overlap listed twice
    # would double the ground it stands for.
    pairs, counts = np.unique(np.column_stack([rows, cells]), axis=0, return_counts=True)
    if np.any(counts > 1):
        repeated = [(row_order[row], int(cell)) for row, cell in pairs[counts > 1][:5]]
        raise DataValidationError(f"The same unit and cell appear in more than one overlap: {repeated}.")

    return Aggregation(units=row_order, rows=rows, columns=cells, weights=weights, n_cells=n_cells)


def _validated_weights(weights: np.ndarray) -> np.ndarray:
    validated = np.asarray(weights, dtype=float)

    if not np.all(np.isfinite(validated)):
        raise DataValidationError("A weight is non-finite, which would poison every total it enters.")

    if np.any(validated < 0.0):
        raise DataValidationError("A weight is negative, so a polygon's total would not be a sum of parts.")

    return validated


def _rows_for_units(labels: list[str], units: Sequence[str] | None) -> tuple[tuple[str, ...], np.ndarray]:
    """The row order, and the row each label occupies in it."""
    non_strings = sorted({type(label).__name__ for label in labels if not isinstance(label, str)})
    if non_strings:
        raise DataValidationError(
            f"Cell assignments carry non-string labels ({', '.join(non_strings)}). A spatial join marks a cell it "
            f"matched to nothing with NaN; mark it with None instead, which is what drops it."
        )

    if units is None:
        units = sorted(set(labels))

    repeated = sorted(unit for unit, count in Counter(units).items() if count > 1)
    if repeated:
        raise DataValidationError(f"The row order repeats units, so their rows would not be reachable: {repeated}.")

    row_of_unit = {unit: row for row, unit in enumerate(units)}
    unknown = sorted(set(labels) - row_of_unit.keys())
    if unknown:
        raise DataValidationError(f"Cells are assigned to units absent from the row order: {unknown}.")

    return tuple(units), np.array([row_of_unit[label] for label in labels], dtype=int)
