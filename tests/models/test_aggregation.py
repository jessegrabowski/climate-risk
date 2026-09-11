import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from climate_risk.exceptions import DataValidationError
from climate_risk.models.aggregation import build_aggregation, build_aggregation_from_overlaps

# A unit square split into a left and a right half, so both units have a known analytic integral.
SPLIT = 0.5


def square_grid(cells_per_axis: int):
    """Cell centroids and areas for a regular grid over the unit square, split into two units."""
    edges = np.linspace(0.0, 1.0, cells_per_axis + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    x, y = (axis.ravel() for axis in np.meshgrid(centers, centers, indexing="ij"))

    area = (1.0 / cells_per_axis) ** 2
    unit_of_cell = ["left" if position < SPLIT else "right" for position in x]

    return x, y, np.full(x.shape, area), unit_of_cell


def integrate_plane(a: float, b: float, c: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """The exact integral of ``a + b*x + c*y`` over a rectangle."""
    return a * (x1 - x0) * (y1 - y0) + b * (x1**2 - x0**2) / 2.0 * (y1 - y0) + c * (y1**2 - y0**2) / 2.0 * (x1 - x0)


def test_a_plane_integrates_exactly():
    """The midpoint rule is exact for a linear field, so this pins the operator against a closed
    form rather than against a tolerance. An error in the weights or the row order shows up here.
    """
    a, b, c = 2.0, 3.0, -1.5
    x, y, areas, unit_of_cell = square_grid(cells_per_axis=8)

    aggregation = build_aggregation(unit_of_cell=unit_of_cell, weights=areas)
    totals = aggregation.aggregate(a + b * x + c * y)

    assert aggregation.units == ("left", "right")
    assert totals == pytest.approx(
        [
            integrate_plane(a, b, c, 0.0, SPLIT, 0.0, 1.0),
            integrate_plane(a, b, c, SPLIT, 1.0, 0.0, 1.0),
        ]
    )


def test_refining_the_grid_converges_on_a_curved_field():
    """A sinusoid is not integrated exactly by the midpoint rule, so the claim is convergence.
    A fixed tolerance would pass on an operator that is wrong by a constant factor.
    """
    # Integral of sin(pi x) sin(pi y) over the unit square, which separates into (2 / pi) squared.
    exact = 4.0 / np.pi**2

    errors = []
    for cells_per_axis in (8, 32):
        x, y, areas, unit_of_cell = square_grid(cells_per_axis)
        aggregation = build_aggregation(unit_of_cell=unit_of_cell, weights=areas)
        totals = aggregation.aggregate(np.sin(np.pi * x) * np.sin(np.pi * y))
        errors.append(abs(totals.sum() - exact))

    assert errors[1] < errors[0] / 10.0, errors


def test_weights_place_each_cell_in_exactly_one_unit():
    """Every cell's weight appears once in the operator, so the column sums are the weights back.
    A cell double-counted into two units inflates one polygon's total and is invisible in a
    row-wise check.
    """
    _, _, areas, unit_of_cell = square_grid(cells_per_axis=6)

    aggregation = build_aggregation(unit_of_cell=unit_of_cell, weights=areas)

    assert np.asarray(aggregation.matrix.sum(axis=0)) == pytest.approx(areas)


def test_a_cell_outside_every_unit_contributes_to_nothing():
    """The gridded extent is a bounding box, so cells over the sea or a neighbouring country are
    normal. They must drop out rather than join whichever unit is nearest.
    """
    aggregation = build_aggregation(
        unit_of_cell=["left", None, "right"],
        weights=np.array([1.0, 99.0, 1.0]),
    )

    totals = aggregation.aggregate(np.array([1.0, 1.0, 1.0]))

    assert totals == pytest.approx([1.0, 1.0])
    assert aggregation.matrix.sum() == pytest.approx(2.0)


def test_a_grid_that_misses_every_unit_gives_an_empty_operator():
    """A bounding box can miss the geometries altogether: an all-sea tile, or a place whose
    polygons failed to load. Both paths have to return no rows rather than raise, so the caller
    sees an empty result instead of a traceback from inside the operator. The field is still the
    full width, so aggregating one of the right length must not trip the width check.
    """
    aggregation = build_aggregation(unit_of_cell=[None, None], weights=np.ones(2))
    field = np.array([1.0, 2.0])

    cells = pt.vector("cells")
    symbolic = pytensor.function([cells], aggregation.aggregate_symbolic(cells))

    assert aggregation.units == ()
    assert aggregation.aggregate(field).shape == (0,)
    assert symbolic(field).shape == (0,)


def test_draws_aggregate_columnwise():
    """Posterior draws come through as (n_cells, n_draws), and each column is its own field."""
    aggregation = build_aggregation(unit_of_cell=["a", "a", "b"], weights=np.ones(3))

    totals = aggregation.aggregate(np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))

    np.testing.assert_allclose(totals, [[3.0, 30.0], [3.0, 30.0]])


def test_the_symbolic_path_integrates_the_plane_exactly():
    """The graph is a scatter-add over index arrays while the array path is a sparse matrix
    product, so they are two implementations of one operator. Pinning the graph against the same
    closed form as :func:`test_a_plane_integrates_exactly` checks it on its own terms rather than
    checking that the two agree, which they would if a shared index array were wrong.
    """
    a, b, c = 2.0, 3.0, -1.5
    x, y, areas, unit_of_cell = square_grid(cells_per_axis=8)

    aggregation = build_aggregation(unit_of_cell=unit_of_cell, weights=areas)
    cells = pt.vector("cells")
    totals = pytensor.function([cells], aggregation.aggregate_symbolic(cells))(a + b * x + c * y)

    assert totals == pytest.approx(
        [
            integrate_plane(a, b, c, 0.0, SPLIT, 0.0, 1.0),
            integrate_plane(a, b, c, SPLIT, 1.0, 0.0, 1.0),
        ]
    )


def test_the_symbolic_path_handles_repeats_gaps_and_empty_units():
    """The scatter-add has to accumulate over cells sharing a unit, skip cells assigned to none,
    and leave an empty unit's row at zero. A plain assignment rather than an increment would keep
    only the last cell of each unit and pass a test where every unit holds one cell.
    """
    aggregation = build_aggregation(
        unit_of_cell=["b", "a", None, "b", "a"],
        weights=np.array([1.5, 2.0, 99.0, 0.5, 3.0]),
        units=["a", "empty", "b"],
    )
    field = np.array([1.0, 2.0, 7.0, 3.0, 4.0])

    cells = pt.vector("cells")
    totals = pytensor.function([cells], aggregation.aggregate_symbolic(cells))(field)

    assert totals == pytest.approx([16.0, 0.0, 3.0])


@pytest.mark.parametrize("n_columns", [1, 3])
def test_the_symbolic_path_aggregates_a_stack_of_fields_columnwise(n_columns):
    """Draws and covariance factors arrive as (n_cells, k) blocks. Each column is its own field,
    so a base array of the wrong rank would sum them together and return one column.
    """
    weights = np.array([2.0, 3.0, 4.0])
    aggregation = build_aggregation(unit_of_cell=["a", "a", "b"], weights=weights)
    block = np.arange(3 * n_columns, dtype=float).reshape(3, n_columns)

    cells = pt.matrix("cells")
    totals = pytensor.function([cells], aggregation.aggregate_symbolic(cells))(block)

    np.testing.assert_allclose(totals, np.stack([weights[:2] @ block[:2], weights[2] * block[2]]))


def test_the_symbolic_gradient_is_the_operator_transpose():
    """The model differentiates through this, and a gather paired with the wrong scatter can give
    the right totals with the wrong sensitivities. The operator is linear, so its Jacobian is the
    matrix itself and the reverse pass is exactly its transpose.
    """
    _, _, areas, unit_of_cell = square_grid(cells_per_axis=4)
    aggregation = build_aggregation(unit_of_cell=unit_of_cell, weights=areas)

    cotangent = np.array([2.0, -3.0])
    cells = pt.vector("cells")
    seeded = pt.sum(aggregation.aggregate_symbolic(cells) * pt.as_tensor_variable(cotangent))
    pullback = pytensor.function([cells], pt.grad(seeded, cells))

    np.testing.assert_allclose(pullback(np.zeros_like(areas)), aggregation.matrix.T @ cotangent)


def test_the_default_row_order_is_sorted():
    """Sorted, not first-appearance: a caller joining totals back to a sorted unit table would get
    silently transposed rows if the default followed the order the cells happened to arrive in.
    """
    aggregation = build_aggregation(unit_of_cell=["b", "a"], weights=np.array([1.0, 2.0]))

    assert aggregation.units == ("a", "b")
    assert aggregation.aggregate(np.array([10.0, 20.0])) == pytest.approx([40.0, 10.0])


def test_the_row_order_can_be_given():
    """The model reads totals by position, so the caller has to be able to fix the order rather
    than depend on the sort of whatever labels happened to appear.
    """
    aggregation = build_aggregation(
        unit_of_cell=["b", "a"],
        weights=np.ones(2),
        units=["b", "a"],
    )

    assert aggregation.units == ("b", "a")
    assert aggregation.aggregate(np.array([1.0, 2.0])) == pytest.approx([1.0, 2.0])


def test_a_unit_with_no_cells_keeps_its_row():
    """A polygon smaller than a cell is real, and its row is all zeros. Dropping the row instead
    would shift every later unit up one and misalign the observations.
    """
    aggregation = build_aggregation(
        unit_of_cell=["a", "a"],
        weights=np.ones(2),
        units=["a", "empty", "b"],
    )

    totals = aggregation.aggregate(np.array([1.0, 1.0]))

    assert aggregation.units == ("a", "empty", "b")
    assert totals == pytest.approx([2.0, 0.0, 0.0])


def test_a_unit_absent_from_the_row_order_is_named():
    """Silently dropping the cells of an unlisted unit would lose part of the field with no sign."""
    with pytest.raises(DataValidationError, match="ghost"):
        build_aggregation(unit_of_cell=["a", "ghost"], weights=np.ones(2), units=["a"])


def test_a_nan_assignment_is_rejected_by_name():
    """A spatial join marks a cell it matched to nothing with NaN rather than None, and NaN is not
    None, so it would otherwise be taken for a real unit label.
    """
    with pytest.raises(DataValidationError, match="None"):
        build_aggregation(unit_of_cell=["a", np.nan, "b"], weights=np.ones(3))  # type: ignore[list-item]


def test_a_repeated_unit_in_the_row_order_is_rejected():
    """A duplicated label makes the row lookup last-wins, which leaves the earlier row permanently
    zero and shifts that unit's total onto a row the caller thinks belongs to something else. A
    repeated gid in an upstream geometry table is an ordinary way to arrive here.
    """
    with pytest.raises(DataValidationError, match="repeats units"):
        build_aggregation(unit_of_cell=["a", "b"], weights=np.ones(2), units=["a", "a", "b"])


def test_mismatched_assignments_and_weights_are_rejected():
    with pytest.raises(DataValidationError, match="same cells"):
        build_aggregation(unit_of_cell=["a", "b"], weights=np.ones(3))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_non_finite_weight_is_rejected(bad):
    """One NaN weight silently turns a polygon's total into NaN and the fit into a wall of errors
    a long way from the cause.
    """
    with pytest.raises(DataValidationError, match="non-finite"):
        build_aggregation(unit_of_cell=["a", "b"], weights=np.array([1.0, bad]))


def test_a_negative_weight_is_rejected():
    with pytest.raises(DataValidationError, match="negative"):
        build_aggregation(unit_of_cell=["a", "b"], weights=np.array([1.0, -1.0]))


def test_a_field_of_the_wrong_length_is_rejected():
    """The operator is built once and applied many times, so a mismatch here means the cell table
    changed underneath it."""
    aggregation = build_aggregation(unit_of_cell=["a", "b"], weights=np.ones(2))

    with pytest.raises(DataValidationError, match="built for 2"):
        aggregation.aggregate(np.ones(3))


def test_a_cell_split_between_units_contributes_to_both():
    """The reason this constructor exists. A cell on a border is one piece of ground shared between
    two units, and each must receive the part that falls inside it. The per-cell constructor cannot
    express that at all, since it takes one label per cell.
    """
    aggregation = build_aggregation_from_overlaps(
        unit_of_overlap=["a", "a", "b", "b"],
        cell_of_overlap=np.array([0, 1, 1, 2]),
        weights=np.array([1.0, 0.25, 0.75, 1.0]),
        n_cells=3,
    )

    totals = aggregation.aggregate(np.array([10.0, 100.0, 1000.0]))

    assert aggregation.units == ("a", "b")
    assert totals == pytest.approx([10.0 + 25.0, 75.0 + 1000.0])


def test_the_two_constructors_agree_when_no_cell_is_shared():
    """One overlap per cell is the case the per-cell constructor already covers, so the operators
    must be identical there. A divergence means the shared validation has drifted.
    """
    per_cell = build_aggregation(unit_of_cell=["b", None, "a", "b"], weights=np.array([1.0, 9.0, 2.0, 3.0]))
    from_overlaps = build_aggregation_from_overlaps(
        unit_of_overlap=["b", "a", "b"],
        cell_of_overlap=np.array([0, 2, 3]),
        weights=np.array([1.0, 2.0, 3.0]),
        n_cells=4,
    )

    field = np.array([1.0, 2.0, 4.0, 8.0])

    assert from_overlaps.units == per_cell.units
    assert from_overlaps.n_cells == per_cell.n_cells
    np.testing.assert_allclose(from_overlaps.aggregate(field), per_cell.aggregate(field))


def test_a_cell_named_in_no_overlap_keeps_its_column():
    """Cells outside every unit still sit in the grid, so the operator has to stay as wide as the
    field the model evaluates. A narrower operator would silently misalign against it.
    """
    aggregation = build_aggregation_from_overlaps(
        unit_of_overlap=["a"], cell_of_overlap=np.array([0]), weights=np.array([1.0]), n_cells=4
    )

    assert aggregation.n_cells == 4
    assert aggregation.aggregate(np.arange(4.0)) == pytest.approx([0.0])


def test_the_same_unit_and_cell_twice_is_rejected():
    """A duplicated pair is summed silently by the sparse matrix, so a double-counted overlap would
    inflate one unit's total with nothing to show for it.
    """
    with pytest.raises(DataValidationError, match="more than one overlap"):
        build_aggregation_from_overlaps(
            unit_of_overlap=["a", "a"],
            cell_of_overlap=np.array([1, 1]),
            weights=np.array([0.5, 0.5]),
            n_cells=2,
        )


@pytest.mark.parametrize("cell", [-1, 5])
def test_an_overlap_naming_a_cell_outside_the_grid_is_rejected(cell):
    """The cell index comes from a raster pass that knows the whole lattice, while the operator is
    built for the clipped field. An index past the end would raise far away from the mismatch.
    """
    with pytest.raises(DataValidationError, match="outside the grid"):
        build_aggregation_from_overlaps(
            unit_of_overlap=["a"], cell_of_overlap=np.array([cell]), weights=np.array([1.0]), n_cells=3
        )


def test_mismatched_overlap_columns_are_rejected():
    with pytest.raises(DataValidationError, match="one of each"):
        build_aggregation_from_overlaps(
            unit_of_overlap=["a", "b"], cell_of_overlap=np.array([0]), weights=np.array([1.0]), n_cells=2
        )


def test_units_that_miss_the_grid_entirely_give_an_empty_operator():
    """The raster layer hands over whatever overlaps it found, and a place whose units all miss the
    grid finds none. That has to come back as an operator with no rows rather than raise, and it
    still has to be as wide as the field the model evaluates.
    """
    aggregation = build_aggregation_from_overlaps(
        unit_of_overlap=[], cell_of_overlap=np.array([], dtype=int), weights=np.array([]), n_cells=4
    )

    assert aggregation.units == ()
    assert aggregation.n_cells == 4
    assert aggregation.aggregate(np.arange(4.0)).shape == (0,)
