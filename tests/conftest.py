"""Shared fixtures.

Material values from Table 1 of Sahib & Kovacs (2024), with two
corrections (see docs/gap_register.md):
  - GFRP E1 printed as 430,000 MPa; implausible for glass fibre and
    stiffer than the CFRP alongside it.  Corrected to 43,000 MPa.
  - Al G12 printed as 260,000 MPa; the isotropic relation E/(2(1+nu))
    gives 26,316 MPa.  Corrected.

Hardcoded here rather than read from config/materials.yaml so the tests
do not depend on a YAML loader that does not exist yet.  Move to the
loader in Week 3.
"""

import pytest

from sandwich.clt import Material


@pytest.fixture
def CFRP() -> Material:
    return Material(
        name="CFRP",
        E1=130_000.0, E2=10_000.0, nu12=0.28, G12=5_000.0,
        t_ply=0.125, rho=1600.0,
        Xt=2000.0, Xc=1300.0, Yt=78.0, Yc=246.0, S=68.0,
        cost_ratio=1.5,
    )


@pytest.fixture
def GFRP() -> Material:
    return Material(
        name="GFRP",
        E1=43_000.0, E2=8_000.0, nu12=0.25, G12=4_300.0,
        t_ply=0.125, rho=1800.0,
        Xt=1140.0, Xc=620.0, Yt=39.0, Yc=128.0, S=60.0,
        cost_ratio=1.0,
    )


@pytest.fixture
def AL() -> Material:
    return Material(
        name="Al",
        E1=70_000.0, E2=70_000.0, nu12=0.33, G12=26_315.8,
        t_ply=0.2, rho=2780.0,
        Xt=186.0, Xc=186.0, Yt=186.0, Yc=186.0, S=110.0,
        cost_ratio=0.31,
    )


@pytest.fixture
def MATERIALS(CFRP, GFRP, AL) -> dict[int, Material]:
    """Layup codes per config/problem.yaml: 1 = CFRP, 2 = GFRP, 3 = Al."""
    return {1: CFRP, 2: GFRP, 3: AL}


@pytest.fixture
def FML_LAYUP(MATERIALS) -> list[Material]:
    """Table 6 knee point: [3,1,1,2,1,2], t_f = 0.825 mm."""
    return [MATERIALS[c] for c in (3, 1, 1, 2, 1, 2)]