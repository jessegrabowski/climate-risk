import math

from collections.abc import Mapping, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.mplot3d import Axes3D

BORDER_WIDTHS = {"ADM2": 0.5, "ADM1": 1.2, "ADM0": 2.4}
LATTICE_EXTENT = (0.0, 1.0, 0.0, 1.0)
RATE_FLOOR = 1e-2
POSTERIOR_COLOR = "#c1443c"
TRUTH_COLOR = "0.25"


def floored_log10(rate: np.ndarray) -> np.ndarray:
    """Events per atom on a log10 scale, with empty atoms held at ``RATE_FLOOR``."""
    return np.log10(np.maximum(rate, RATE_FLOOR))


def spread_over_units(unit_counts: np.ndarray, units: np.ndarray) -> np.ndarray:
    """
    Divide each unit's count equally among its atoms.

    Parameters
    ----------
    unit_counts : ndarray
        Count reported at each unit, indexed by unit.
    units : ndarray
        Unit of each atom.

    Returns
    -------
    density : ndarray
        Count per atom, one entry per atom.
    """
    return (unit_counts / np.bincount(units))[units]


def draw_borders(axis: Axes, borders: Mapping[str, gpd.GeoSeries], *, color: str) -> None:
    """Draw every level's borders, heavier the coarser the level."""
    for level, boundary in borders.items():
        boundary.plot(ax=axis, color=color, linewidth=BORDER_WIDTHS[level])


def plot_atoms(axis: Axes,
               values: np.ndarray,
               borders: Mapping[str, gpd.GeoSeries],
               *,
               title: str,
               cmap: str = "magma",
               vmin: float | None = None,
               vmax: float | None = None) -> AxesImage:
    """
    Draw a field over a square lattice of atoms, with every level's borders over it.

    Parameters
    ----------
    axis : Axes
        Where to draw.
    values : ndarray
        One value per atom, row-major over the lattice.
    borders : mapping of str to GeoSeries
        Boundary of every unit, keyed by level.
    title : str
        Title of the panel.
    cmap : str, optional
        Colormap for ``values``. Default "magma".
    vmin, vmax : float, optional
        Limits of the color scale. Default None, which takes them from ``values``.

    Returns
    -------
    image : AxesImage
        The drawn field, for a colorbar.
    """
    grid_side = math.isqrt(len(values))
    image = axis.imshow(np.asarray(values).reshape(grid_side, grid_side),
                        origin="lower",
                        extent=LATTICE_EXTENT,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax)
    draw_borders(axis, borders, color="white")
    axis.set_axis_off()
    axis.set_title(title, loc="left")

    return image


def plot_fields(fields: Mapping[str, np.ndarray], borders: Mapping[str, gpd.GeoSeries]) -> Figure:
    """
    Draw several fields over the lattice in one row, on a shared diverging scale centered at zero.

    Parameters
    ----------
    fields : mapping of str to ndarray
        One value per atom for each panel, keyed by the panel's title.
    borders : mapping of str to GeoSeries
        Boundary of every unit, keyed by level.

    Returns
    -------
    figure : Figure
        The drawn row.
    """
    limit = max(np.abs(values).max() for values in fields.values())
    figure, axes = plt.subplots(1, len(fields), figsize=(4.0 * len(fields), 4.4))

    images = [
        plot_atoms(axis,
                   values,
                   borders,
                   title=title,
                   cmap="RdBu_r",
                   vmin=-limit,
                   vmax=limit)
        for axis, (title, values) in zip(axes, fields.items(), strict=True)
    ]

    figure.colorbar(images[0],
                    ax=axes,
                    shrink=0.8,
                    pad=0.01,
                    label="field, log rate")

    return figure


def draw_stack(axis: Axes3D,
               layers: Sequence[tuple[str, np.ndarray, Sequence[str]]],
               borders: Mapping[str, gpd.GeoSeries],
               *,
               vmin: float,
               vmax: float,
               label_size: float,
               zoom: float,
               line_scale: float) -> None:
    """
    Draw atom-level densities as horizontal sheets stacked top to bottom, on a log10 scale.

    Parameters
    ----------
    axis : Axes3D
        An orthographic 3d axis to draw in.
    layers : sequence of (str, ndarray, sequence of str)
        One sheet each, top first: its label, its density per atom, and the levels whose borders
        it carries.
    borders : mapping of str to GeoSeries
        Boundary of every unit, keyed by level.
    vmin, vmax : float
        Limits of the log10 color scale.
    label_size : float
        Font size of the sheet labels.
    zoom : float
        Scale of the stack inside its axis.
    line_scale : float
        Factor on ``BORDER_WIDTHS``.
    """
    grid_side = math.isqrt(len(layers[0][1]))
    edges = np.linspace(LATTICE_EXTENT[0], LATTICE_EXTENT[1], grid_side + 1)
    grid_x, grid_y = np.meshgrid(edges, edges)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for depth, (label, density, outlined_levels) in enumerate(layers):
        height = -float(depth)
        colors = plt.get_cmap("magma")(norm(floored_log10(density).reshape(grid_side, grid_side)))
        axis.plot_surface(grid_x,
                          grid_y,
                          np.full_like(grid_x, height),
                          facecolors=colors,
                          rstride=1,
                          cstride=1,
                          shade=False,
                          linewidth=0,
                          antialiased=False)

        for level in outlined_levels:
            for part in shapely.get_parts(borders[level].to_numpy()):
                x, y = shapely.get_coordinates(part).T
                axis.plot(x,
                          y,
                          height,
                          color="white",
                          linewidth=BORDER_WIDTHS[level] * line_scale,
                          zorder=10)

        axis.text(1.08,
                  0.0,
                  height,
                  label,
                  fontsize=label_size)

    axis.view_init(elev=28, azim=-55)
    axis.set_box_aspect((1.0, 1.0, 2.2), zoom=zoom)
    axis.set_axis_off()
