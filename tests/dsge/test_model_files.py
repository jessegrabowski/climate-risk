import re

import pytest

from climate_risk.dsge.model_files import GCN_FILES, resolve_gcn_path

# A shock is declared as `epsilon_name[] ~ Distribution(...)` inside a block's `shocks` section.
SHOCK_DECLARATION = re.compile(r"^\s*(\w+)\[\]\s*~", re.MULTILINE)


def shocks_declared_in(variant: str) -> set[str]:
    return set(SHOCK_DECLARATION.findall(resolve_gcn_path(variant).read_text()))


@pytest.mark.parametrize("variant", sorted(GCN_FILES))
def test_every_declared_variant_is_on_disk(variant):
    """The files are package data, so a rename upstream leaves the mapping pointing at nothing."""
    assert resolve_gcn_path(variant).is_file()


def test_an_unknown_variant_says_which_ones_exist():
    with pytest.raises(ValueError, match="nonlinear"):
        resolve_gcn_path("m8")


def test_the_pair_are_different_files():
    assert resolve_gcn_path("nonlinear") != resolve_gcn_path("approx")


def test_both_variants_declare_the_same_shocks():
    """The pair is only comparable while the shock sets agree. Refreshing one file from upstream and
    not the other is the way that stops being true, and it is invisible until an estimate disagrees.
    """
    nonlinear = shocks_declared_in("nonlinear")

    assert nonlinear, "no shock declarations matched; the .gcn syntax has moved on"
    assert nonlinear == shocks_declared_in("approx")


def test_the_approximation_drops_the_calvo_auxiliaries():
    """The collapsed Phillips curves are what makes `approx` cheaper to estimate. If the recursion's
    auxiliary variables reappear, the file is no longer the first-order sibling it claims to be.
    """
    nonlinear = resolve_gcn_path("nonlinear").read_text()
    approx = resolve_gcn_path("approx").read_text()

    # Asserting absence alone passes on a typo'd name, so each one is checked present where it belongs.
    for auxiliary in ("x_1[]", "p_H_tilde[]"):
        assert auxiliary in nonlinear, auxiliary
        assert auxiliary not in approx, auxiliary
