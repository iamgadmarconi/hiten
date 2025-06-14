import numpy as np
import pytest

# System and L-point setup
from system.base import System, systemConfig
from system.body import Body
from utils.constants import Constants

# Center manifold reduction
from algorithms.center.manifold import center_manifold_real
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               init_index_tables)

# CPU Poincare map generation
from algorithms.center.poincare.map import generate_iterated_poincare_map

# GPU Poincare map generation
from algorithms.center.poincare.cuda.map import generate_iterated_poincare_map_gpu


# Test constants
TEST_MAX_DEG = 6  # Keep low for faster CM reduction in tests
TEST_L_POINT_IDX = 1
TEST_SYSTEM_NAME = "EM"  # Earth-Moon system

TEST_H0 = 0.2  # Example energy level
TEST_N_SEEDS = 3
TEST_N_ITER = 20
TEST_DT = 0.01
TEST_SEED_AXIS = "q2"


@pytest.fixture(scope="module")
def poincare_test_setup():
    """
    Sets up the system, L-point, and performs center manifold reduction.
    This is a module-scoped fixture to avoid recomputing for multiple tests if added later.
    """
    # 1. Initialize polynomial indexing tables
    psi, clmo = init_index_tables(TEST_MAX_DEG)
    encode_dict_list = _create_encode_dict_from_clmo(clmo)

    # 2. Set up the system (Earth-Moon)
    Sun = Body("Sun", Constants.bodies["sun"]["mass"], Constants.bodies["sun"]["radius"], "yellow")
    Earth = Body("Earth", Constants.bodies["earth"]["mass"], Constants.bodies["earth"]["radius"], "blue", Sun)
    Moon = Body("Moon", Constants.bodies["moon"]["mass"], Constants.bodies["moon"]["radius"], "gray", Earth)
    d_EM = Constants.get_orbital_distance("earth", "moon")
    system_EM = System(systemConfig(Earth, Moon, d_EM))

    selected_system = system_EM
    selected_l_point = selected_system.get_libration_point(TEST_L_POINT_IDX)

    # 3. Perform centre-manifold reduction
    # This can be time-consuming, hence TEST_MAX_DEG is kept low.
    H_cm_rn_full = center_manifold_real(selected_l_point, psi, clmo, TEST_MAX_DEG)

    return {
        "H_blocks": H_cm_rn_full,
        "max_degree": TEST_MAX_DEG,
        "psi_table": psi,
        "clmo_table": clmo,
        "encode_dict_list": encode_dict_list,
        "h0": TEST_H0,
        "n_seeds": TEST_N_SEEDS,
        "n_iter": TEST_N_ITER,
        "dt": TEST_DT,
        "seed_axis": TEST_SEED_AXIS,
    }


def test_generate_iterated_poincare_map_cpu_vs_gpu(poincare_test_setup):
    """
    Compares the output of generate_iterated_poincare_map (CPU)
    with generate_iterated_poincare_map_gpu (GPU).
    """
    params = poincare_test_setup

    # Common parameters for both CPU and GPU calls
    common_args = {
        "h0": params["h0"],
        "H_blocks": params["H_blocks"],
        "max_degree": params["max_degree"],
        "psi_table": params["psi_table"],
        "clmo_table": params["clmo_table"],
        "encode_dict_list": params["encode_dict_list"],
        "n_seeds": params["n_seeds"],
        "n_iter": params["n_iter"],
        "dt": params["dt"],
        "seed_axis": params["seed_axis"],
    }

    # Generate points using CPU version (ensure RK4 is used for fair comparison)
    # generate_iterated_poincare_map uses _poincare_step_jit, which defaults to RK4
    # if use_symplectic is False (its default value matches for map.py).
    # We'll explicitly set use_symplectic=False. integrator_order and c_omega_heuristic
    # are relevant only if use_symplectic=True for the CPU version.
    pts_cpu = generate_iterated_poincare_map(
        **common_args,
        use_symplectic=False, 
        integrator_order=4 # Not used if use_symplectic=False, but provide a value
    )

    # Generate points using GPU version
    # generate_iterated_poincare_map_gpu currently only supports RK4
    pts_gpu = generate_iterated_poincare_map_gpu(
        **common_args
        # use_symplectic defaults to False and is handled internally
    )

    # Compare the results
    assert pts_cpu.shape[0] == pts_gpu.shape[0], \
        f"Number of generated points differ: CPU={pts_cpu.shape[0]}, GPU={pts_gpu.shape[0]}"

    if pts_cpu.shape[0] > 0:
        # Sort points for consistent comparison
        # Lexicographical sort (by first column, then by second)
        pts_cpu_sorted = pts_cpu[np.lexsort((pts_cpu[:, 1], pts_cpu[:, 0]))]
        pts_gpu_sorted = pts_gpu[np.lexsort((pts_gpu[:, 1], pts_gpu[:, 0]))]

        np.testing.assert_allclose(
            pts_cpu_sorted,
            pts_gpu_sorted,
            atol=1e-6,  # Absolute tolerance
            rtol=1e-6,  # Relative tolerance
            err_msg="Poincaré map points differ between CPU and GPU implementations"
        )
