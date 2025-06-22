import numpy as np
import pytest
import tempfile
import os

from hiten.system.center import CenterManifold
from hiten.system.poincare import PoincareMap, poincareMapConfig
from hiten.system.base import System, systemConfig
from hiten.system.body import Body
from hiten.utils.constants import Constants

TEST_MAX_DEG = 6
TEST_L_POINT_IDX = 1

TEST_H0 = 0.2
TEST_N_SEEDS = 3
TEST_N_ITER = 20
TEST_DT = 0.01
TEST_SEED_AXIS = "q2"

@pytest.fixture(scope="module")
def poincare_test_setup():
    Earth = Body("Earth", Constants.bodies["earth"]["mass"], Constants.bodies["earth"]["radius"], "blue")
    Moon = Body("Moon", Constants.bodies["moon"]["mass"], Constants.bodies["moon"]["radius"], "gray", Earth)
    distance = Constants.get_orbital_distance("earth", "moon")
    system = System(systemConfig(Earth, Moon, distance))
    libration_point = system.get_libration_point(TEST_L_POINT_IDX)

    cm = CenterManifold(libration_point, TEST_MAX_DEG)
    cm.compute()

    pmConfig = poincareMapConfig(
        dt=TEST_DT,
        method="rk",
        integrator_order=4,
        c_omega_heuristic=20.0,
        n_seeds=TEST_N_SEEDS,
        n_iter=TEST_N_ITER,
        seed_axis=TEST_SEED_AXIS,
        compute_on_init=True,
        use_gpu=False,
    )

    pm = PoincareMap(cm, TEST_H0, pmConfig)
    return cm, pmConfig, pm

def test_poincaremap_save_load(poincare_test_setup):
    cm, pmConfig, pm = poincare_test_setup

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmpfile:
        filepath = tmpfile.name
    try:
        pm.save(filepath)

        # Create a new map instance (with same CM, config, but do not compute)
        pm_loaded = PoincareMap(cm, 0.0, pmConfig)  # energy will be overwritten by load
        pm_loaded.load(filepath)

        # Check energy
        assert np.isclose(pm.energy, pm_loaded.energy), f"Energy mismatch: {pm.energy} vs {pm_loaded.energy}"
        # Check config
        assert pm.config == pm_loaded.config, f"Config mismatch: {pm.config} vs {pm_loaded.config}"
        # Check points
        np.testing.assert_allclose(pm.points, pm_loaded.points, atol=1e-12, rtol=1e-12, err_msg="Loaded points do not match saved points")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
