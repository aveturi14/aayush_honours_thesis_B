import numpy as np
import pytest 

from sandwich.clt import (effective_modulus, extensional_stiffness,
                          ply_boundaries, reduced_stiffness,
                          transverse_strain_ratio)

# fixtures providing CFRP / GFRP / AL Material objects live in conftest.py


def test_isotropic_Q(AL):
    """Al must give Q11 = Q22 = E/(1-nu^2) and Q66 = E/(2(1+nu))."""
    Q = reduced_stiffness(AL)
    expected = AL.E1 / (1 - AL.nu12 ** 2)
    assert Q[0, 0] == pytest.approx(expected)
    assert Q[1, 1] == pytest.approx(expected)


def test_z_array_closes(FML_LAYUP):
    z = ply_boundaries(FML_LAYUP)
    h = sum(m.t_ply for m in FML_LAYUP)
    assert z[-1] == pytest.approx(h / 2.0)
    assert z[0] == pytest.approx(-h / 2.0)


def test_no_extension_shear_coupling(FML_LAYUP):
    A = extensional_stiffness(FML_LAYUP)
    assert A[0, 2] == pytest.approx(0.0)
    assert A[1, 2] == pytest.approx(0.0)


@pytest.mark.parametrize("fixture,expected", [("CFRP", 130_000.0),
                                              ("AL", 70_000.0)])
def test_homogeneous_returns_own_modulus(request, fixture, expected):
    """A stack of one material has that material's own E1."""
    m = request.getfixturevalue(fixture)
    assert effective_modulus([m] * 6) == pytest.approx(expected, rel=1e-9)


def test_homogeneous_strain_ratio_is_poisson(CFRP):
    A = extensional_stiffness([CFRP] * 6)
    assert transverse_strain_ratio(A) == pytest.approx(-CFRP.nu12)


def test_fml_modulus(FML_LAYUP):
    """Hand calculation, 2026-10-08.  See docs/hand_calcs_week2.md."""
    assert effective_modulus(FML_LAYUP) == pytest.approx(89_111.0, rel=1e-4)