import pytest

from sandwich.failure import coefficients, laminate_strength


@pytest.mark.parametrize("fixture", ["CFRP", "GFRP", "AL"])
def test_envelope_is_closed(request, fixture):
    """Kaw problem 2.41: H12^2 < H11*H22, else the surface is a hyperboloid."""
    assert coefficients(request.getfixturevalue(fixture)).closed_envelope()


def test_isotropic_linear_terms_vanish(AL):
    c = coefficients(AL)
    assert c.H1 == pytest.approx(0.0)
    assert c.H2 == pytest.approx(0.0)


@pytest.mark.parametrize("fixture", ["CFRP", "AL"])
def test_homogeneous_returns_own_strength(request, fixture):
    """A single-material stack fails at that material's Xt."""
    m = request.getfixturevalue(fixture)
    sigma_fx, _, _ = laminate_strength([m] * 6)
    assert sigma_fx == pytest.approx(m.Xt, rel=1e-9)


def test_fml_strength(FML_LAYUP):
    """Hand calculation, 2026-10-08.  See docs/hand_calcs_week2.md."""
    sigma_fx, k, _ = laminate_strength(FML_LAYUP)
    assert sigma_fx == pytest.approx(237.5, rel=1e-3)
    assert FML_LAYUP[k].name == "Al"       # the ductile ply governs


def test_h12_sensitivity(FML_LAYUP):
    """GR-002: all three conventions plus zero should bracket each other."""
    results = {m: laminate_strength(FML_LAYUP, h12_model=m)[0]
               for m in ("mises_hencky", "tsai_hill", "hoffman", "zero")}
    spread = (max(results.values()) - min(results.values())) / min(results.values())
    assert spread < 0.25, results      # tighten once you know the real spread