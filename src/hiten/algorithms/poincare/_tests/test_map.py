import numpy as np
import pytest

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

    pmGPUConfig = poincareMapConfig(
        dt=TEST_DT,
        method="rk",
        integrator_order=4,
        c_omega_heuristic=20.0,
        n_seeds=TEST_N_SEEDS,
        n_iter=TEST_N_ITER,
        seed_axis=TEST_SEED_AXIS,
        compute_on_init=True,
        use_gpu=True,
    )

    pmCPUConfig = poincareMapConfig(
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

    pmGPU = PoincareMap(cm, TEST_H0, pmGPUConfig)
    pmCPU = PoincareMap(cm, TEST_H0, pmCPUConfig)

    return pmGPU, pmCPU


def test_map_cpu_vs_gpu(poincare_test_setup):
    pmGPU, pmCPU = poincare_test_setup

    pts_gpu, pts_cpu = pmGPU.points, pmCPU.points

    assert pts_cpu.shape[0] == pts_gpu.shape[0], f"Number of generated points differ: CPU={pts_cpu.shape[0]}, GPU={pts_gpu.shape[0]}"

    if pts_cpu.shape[0] > 0:
        pts_cpu_sorted = pts_cpu[np.lexsort((pts_cpu[:, 1], pts_cpu[:, 0]))]
        pts_gpu_sorted = pts_gpu[np.lexsort((pts_gpu[:, 1], pts_gpu[:, 0]))]

        np.testing.assert_allclose(
            pts_cpu_sorted,
            pts_gpu_sorted,
            atol=1e-6,
            rtol=1e-6,
            err_msg="Poincaré map points differ between CPU and GPU implementations"
        )
