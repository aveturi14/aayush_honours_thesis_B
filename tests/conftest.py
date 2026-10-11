
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
        E1=70_000.0, E2=70_000.0, nu12=0.33, G12=26_000.0,
        t_ply=0.2, rho=2780.0,
        Xt=186.0, Xc=186.0, Yt=186.0, Yc=186.0, S=110.0,
        cost_ratio=0.31,
    )


@pytest.fixture
def MATERIALS(CFRP, GFRP, AL) -> dict[int, Material]:
    return {1: CFRP, 2: GFRP, 3: AL}


@pytest.fixture
def FML_LAYUP(MATERIALS) -> list[Material]:
    return [MATERIALS[c] for c in (3, 1, 1, 2, 1, 2)]