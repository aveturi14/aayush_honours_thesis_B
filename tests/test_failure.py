"""Tsai-Wu first-ply-failure tests.

As in test_clt.py, the homogeneous cases are theorems (a single-material
stack fails at that material's Xt) and the FML value is a hand
calculation.
"""

import pytest

from sandwich.failure import coefficients, laminate_strength


# --- coefficients ----------------------------------------------------------

@pytest.mark.parametrize("fixture", ["CFRP", "GFRP", "AL"])
def test_envelope_is_closed(request, fixture):
    """Kaw problem 2.41: H12^2 < H11*H22, else the surface is a hyperboloid."""
    assert coefficients(request.getfixturevalue(fixture)).closed_envelope()


def test_isotropic_linear_terms_vanish(AL):
    """Equal tensile and compressive strengths give H1 = H2 = 0."""
    c = coefficients(AL)
    assert c.H1 == pytest.approx(0.0)
    assert c.H2 == pytest.approx(0.0)


def test_h12_models_all_negative(CFRP):
    """Every empirical H12 convention (Kaw Eq. 2.171) is non-positive."""
    for model in ("mises_hencky", "tsai_hill", "hoffman", "zero"):
        assert coefficients(CFRP, h12_model=model).H12 <= 0.0


# --- laminate strength -----------------------------------------------------

@pytest.mark.parametrize("fixture", ["CFRP", "AL"])
def test_homogeneous_returns_own_strength(request, fixture):
    """A single-material stack fails at that material's Xt, exactly."""
    m = request.getfixturevalue(fixture)
    sigma_fx, _, _ = laminate_strength([m] * 6)
    assert sigma_fx == pytest.approx(m.Xt, rel=1e-9)


def test_homogeneous_all_plies_equally_critical(CFRP):
    """Identical plies see identical stress, so all R are equal."""
    _, _, ratios = laminate_strength([CFRP] * 6)
    assert ratios == pytest.approx([ratios[0]] * 6)


def test_fml_strength(FML_LAYUP):
    """Hand calculation, 2026-10-08: sigma_fx = 237.49 MPa."""
    sigma_fx, k, _ = laminate_strength(FML_LAYUP)
    assert sigma_fx == pytest.approx(237.49, rel=1e-4)
    assert FML_LAYUP[k].name == "Al"       # the ductile ply governs


def test_fml_aluminium_is_most_critical(FML_LAYUP):
    """Al sits far closer to failure than the composite plies."""
    _, _, ratios = laminate_strength(FML_LAYUP)
    al_R = min(ratios)
    assert al_R == pytest.approx(0.0026651, rel=1e-4)
    assert max(ratios) / al_R > 5.0


def test_fml_strength_exceeds_governing_ply_strength(FML_LAYUP):
    """Stiff plies carry more, so the laminate stress exceeds Al's 186 MPa."""
    sigma_fx, k, _ = laminate_strength(FML_LAYUP)
    assert sigma_fx > FML_LAYUP[k].Xt


def test_h12_sensitivity(FML_LAYUP):
    """GR-002: quantify how much the H12 convention moves sigma_fx."""
    results = {m: laminate_strength(FML_LAYUP, h12_model=m)[0]
               for m in ("mises_hencky", "tsai_hill", "hoffman", "zero")}
    spread = (max(results.values()) - min(results.values())) / min(results.values())
    assert spread < 0.25, results      # tighten once the real spread is known