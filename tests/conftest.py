"""Classical lamination theory tests.

The homogeneous cases are theorems, not recorded numbers: a stack of one
material must return that material's own E1.  Those catch real errors.
The FML values are hand calculations (2026-10-08, docs/hand_calcs_week2.md)
and only lock in arithmetic.
"""

import pytest

from sandwich.clt import (effective_modulus, extensional_stiffness,
                          ply_boundaries, reduced_stiffness,
                          transverse_strain_ratio)


# --- inputs ----------------------------------------------------------------

def test_face_sheet_thicknesses(MATERIALS):
    """t_f must match Table 6 for all three key designs.

    Verifies both the ply thicknesses from Table 1 and the layup code
    mapping (1 = CFRP, 2 = GFRP, 3 = Al) in one assertion.
    """
    cases = [((1, 1, 1, 1, 1, 1), 0.75),     # totally CFRP
             ((3, 3, 3, 3, 3, 3), 1.20),     # totally Al
             ((3, 1, 1, 2, 1, 2), 0.825)]    # FML knee point
    for codes, expected in cases:
        t_f = sum(MATERIALS[c].t_ply for c in codes)
        assert t_f == pytest.approx(expected), codes


# --- ply level -------------------------------------------------------------

def test_isotropic_Q(AL):
    """Aluminium must give Q11 = Q22 = E/(1-nu^2) and Q66 = E/(2(1+nu))."""
    Q = reduced_stiffness(AL)
    assert Q[0, 0] == pytest.approx(AL.E1 / (1 - AL.nu12 ** 2))
    assert Q[1, 1] == pytest.approx(AL.E1 / (1 - AL.nu12 ** 2))
    assert Q[2, 2] == pytest.approx(AL.E1 / (2 * (1 + AL.nu12)), rel=1e-5)


def test_Q_exceeds_E1(CFRP):
    """The Poisson correction raises Q11 above E1, never lowers it."""
    assert reduced_stiffness(CFRP)[0, 0] > CFRP.E1


# --- geometry --------------------------------------------------------------

def test_z_array_closes(FML_LAYUP):
    z = ply_boundaries(FML_LAYUP)
    h = sum(m.t_ply for m in FML_LAYUP)
    assert z[0] == pytest.approx(-h / 2.0)
    assert z[-1] == pytest.approx(+h / 2.0)
    assert len(z) == len(FML_LAYUP) + 1


def test_z_differences_are_ply_thicknesses(FML_LAYUP):
    z = ply_boundaries(FML_LAYUP)
    for k, m in enumerate(FML_LAYUP):
        assert z[k + 1] - z[k] == pytest.approx(m.t_ply)


# --- A matrix --------------------------------------------------------------

def test_no_extension_shear_coupling(FML_LAYUP):
    """All plies at 0 deg, so A16 = A26 = 0."""
    A = extensional_stiffness(FML_LAYUP)
    assert A[0, 2] == pytest.approx(0.0)
    assert A[1, 2] == pytest.approx(0.0)


def test_A_is_symmetric(FML_LAYUP):
    A = extensional_stiffness(FML_LAYUP)
    assert A[0, 1] == pytest.approx(A[1, 0])


def test_A_independent_of_stacking_order(MATERIALS):
    """Membrane stiffness depends on composition, not sequence."""
    a = extensional_stiffness([MATERIALS[c] for c in (3, 1, 1, 2, 1, 2)])
    b = extensional_stiffness([MATERIALS[c] for c in (1, 1, 1, 2, 2, 3)])
    assert a == pytest.approx(b)


# --- effective modulus -----------------------------------------------------

@pytest.mark.parametrize("fixture,expected", [("CFRP", 130_000.0),
                                              ("AL", 70_000.0)])
def test_homogeneous_returns_own_modulus(request, fixture, expected):
    """A single-material stack has that material's own E1, exactly."""
    m = request.getfixturevalue(fixture)
    assert effective_modulus([m] * 6) == pytest.approx(expected, rel=1e-9)


def test_homogeneous_strain_ratio_is_poisson(CFRP):
    """eps_y / eps_x reduces to -nu12 when all plies share one Q."""
    A = extensional_stiffness([CFRP] * 6)
    assert transverse_strain_ratio(A) == pytest.approx(-CFRP.nu12)


def test_fml_modulus(FML_LAYUP):
    """Hand calculation: A11 = 75633, A12 = 6747, A22 = 21507."""
    assert effective_modulus(FML_LAYUP) == pytest.approx(89_111.0, rel=1e-4)


def test_fml_strain_ratio(FML_LAYUP):
    """Lies between the constituent Poisson ratios (0.28 and 0.33)."""
    A = extensional_stiffness(FML_LAYUP)
    ratio = transverse_strain_ratio(A)
    assert ratio == pytest.approx(-0.31370, rel=1e-4)
    assert -0.33 < ratio < -0.28


def test_fml_modulus_brackets_constituents(FML_LAYUP, CFRP, AL):
    """Mixed layup must sit between the two homogeneous extremes."""
    E = effective_modulus(FML_LAYUP)
    assert effective_modulus([AL] * 6) < E < effective_modulus([CFRP] * 6)