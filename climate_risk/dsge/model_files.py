from pathlib import Path

GCN_DIR = Path(__file__).parent / "gcn"

# CatDSGE ships as a matched pair: the model as written, and the first-order sibling that collapses
# each Calvo recursion to one Phillips curve. They share a calibration and a shock set, so a result
# from one is comparable with a result from the other.
GCN_FILES = {
    "nonlinear": "catdsge_v3.gcn",
    "approx": "catdsge_v3_linear_approx.gcn",
}


def resolve_gcn_path(variant: str) -> Path:
    """
    Return the path to the CatDSGE ``.gcn`` file for ``variant``.

    Parameters
    ----------
    variant : str
        Which of the pair to load. ``'nonlinear'`` is the model as written; ``'approx'`` is the
        first-order sibling used for estimation.

    Returns
    -------
    path : Path
        Location of the file inside the installed package.
    """
    if variant not in GCN_FILES:
        known = ", ".join(sorted(GCN_FILES))
        raise ValueError(f"unknown CatDSGE variant {variant!r}, expected one of {known}")

    return GCN_DIR / GCN_FILES[variant]
