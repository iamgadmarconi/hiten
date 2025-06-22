import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

from hiten.system.center import CenterManifold
from hiten.system.base import System, systemConfig
from hiten.system.body import Body
from hiten.utils.constants import Constants

TEST_MAX_DEG = 6
TEST_L_POINT_IDX = 1


@pytest.fixture(scope="module")
def reduction_test_setup():
    Earth = Body("Earth", Constants.bodies["earth"]["mass"], Constants.bodies["earth"]["radius"], "blue")
    Moon = Body("Moon", Constants.bodies["moon"]["mass"], Constants.bodies["moon"]["radius"], "gray", Earth)
    Sun = Body("Sun", Constants.bodies["sun"]["mass"], Constants.bodies["sun"]["radius"], "yellow")
    distance_em = Constants.get_orbital_distance("earth", "moon")
    distance_se = Constants.get_orbital_distance("sun", "earth")

    system_em = System(systemConfig(Earth, Moon, distance_em))
    system_se = System(systemConfig(Sun, Earth, distance_se))

    libration_point_em = system_em.get_libration_point(TEST_L_POINT_IDX) # L1
    libration_point_se = system_se.get_libration_point(TEST_L_POINT_IDX) # L1

    cm_em = CenterManifold(libration_point_em, TEST_MAX_DEG)
    cm_se = CenterManifold(libration_point_se, TEST_MAX_DEG)
    cm_em.compute()
    cm_se.compute()

    return cm_em, cm_se


def calculate_and_plot_ratios(ax, H_cm_real, max_n_to_plot, plot_title):
    """
    Calculate and plot coefficient radius ratios for a given Hamiltonian.
    
    Args:
        ax: matplotlib axis object
        H_cm_real: Hamiltonian coefficients from center manifold
        max_n_to_plot: maximum degree to plot
        plot_title: title for the plot
    """
    if len(H_cm_real) <= max_n_to_plot:
        raise ValueError(
            f"Hamiltonian coefficients available up to degree {len(H_cm_real)-1}, "
            f"but plotting requires up to degree {max_n_to_plot}."
        )

    H_norms_1 = {}  # Keyed by degree n
    # We need norms for H_n where n ranges from 2 to max_n_to_plot.
    for n_deg in range(2, max_n_to_plot + 1):
        if H_cm_real[n_deg] is not None and len(H_cm_real[n_deg]) > 0:
            H_norms_1[n_deg] = np.sum(np.abs(H_cm_real[n_deg]))
        else:
            # If coefficients for a degree are None or empty, its norm is 0.
            H_norms_1[n_deg] = 0.0

    n_values_for_plot_axis = list(range(3, max_n_to_plot + 1))
    r1_values = []
    r2_values = []

    for n_val in n_values_for_plot_axis:
        # r_n^(1) = ||H_n||_1 / ||H_{n-1}||_1
        if H_norms_1.get(n_val - 1, 0) == 0: # Check H_{n-1} norm
            r1 = np.nan  # Avoid division by zero; paper's plots suggest non-zero denominators
        else:
            r1 = H_norms_1.get(n_val, 0) / H_norms_1[n_val - 1]
        r1_values.append(r1)
        
        # r_n^(2) = ||H_n||_1^(1/n)
        norm_Hn = H_norms_1.get(n_val, 0)
        if norm_Hn < 0: # L1 norm should be non-negative
            r2 = np.nan
        elif norm_Hn == 0:
            r2 = 0.0
        else:
            r2 = norm_Hn**(1.0 / n_val)
        r2_values.append(r2)

    ax.plot(n_values_for_plot_axis, r1_values, 'o', label='$r_n^{(1)}$', markersize=3)
    ax.plot(n_values_for_plot_axis, r2_values, '+', label='$r_n^{(2)}$', markersize=5)
    ax.set_xlabel('$n$')
    ax.set_title(plot_title)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Set y-axis limits and x-tics based on the plot requirements
    ax.set_ylim(0.4, 1.8)
    ax.set_xticks(np.arange(0, max_n_to_plot + 1, 1))
    ax.set_xlim(2, max_n_to_plot + 1)


def test_hamiltonian_norms_calculation(reduction_test_setup):
    """
    Test that Hamiltonian norms can be calculated for both systems.
    """
    cm_em, cm_se = reduction_test_setup

    # Test Earth-Moon system
    H_cm_real_em = cm_em.cache_get(('hamiltonian', TEST_MAX_DEG, 'center_manifold_real'))
    assert H_cm_real_em is not None, "Earth-Moon Hamiltonian coefficients should be available"
    assert len(H_cm_real_em) > TEST_MAX_DEG, f"Expected at least {TEST_MAX_DEG+1} degrees of coefficients"

    # Test Sun-Earth system  
    H_cm_real_se = cm_se.cache_get(('hamiltonian', TEST_MAX_DEG, 'center_manifold_real'))
    assert H_cm_real_se is not None, "Sun-Earth Hamiltonian coefficients should be available"
    assert len(H_cm_real_se) > TEST_MAX_DEG, f"Expected at least {TEST_MAX_DEG+1} degrees of coefficients"

    # Calculate norms for both systems
    for system_name, H_cm_real in [("Earth-Moon", H_cm_real_em), ("Sun-Earth", H_cm_real_se)]:
        H_norms = {}
        for n_deg in range(2, TEST_MAX_DEG + 1):
            if H_cm_real[n_deg] is not None and len(H_cm_real[n_deg]) > 0:
                H_norms[n_deg] = np.sum(np.abs(H_cm_real[n_deg]))
            else:
                H_norms[n_deg] = 0.0
        
        # Verify norms are non-negative
        for n_deg, norm in H_norms.items():
            assert norm >= 0, f"{system_name} Hamiltonian norm at degree {n_deg} should be non-negative"
        
        print(f"{system_name} Hamiltonian norms: {H_norms}")


def test_coefficient_radius_ratios(reduction_test_setup):
    """
    Test that coefficient radius ratios can be calculated for both systems.
    """
    cm_em, cm_se = reduction_test_setup

    # Test both systems
    for system_name, cm in [("Earth-Moon", cm_em), ("Sun-Earth", cm_se)]:
        H_cm_real = cm.cache_get(('hamiltonian', TEST_MAX_DEG, 'center_manifold_real'))
        
        # Calculate norms
        H_norms = {}
        for n_deg in range(2, TEST_MAX_DEG + 1):
            if H_cm_real[n_deg] is not None and len(H_cm_real[n_deg]) > 0:
                H_norms[n_deg] = np.sum(np.abs(H_cm_real[n_deg]))
            else:
                H_norms[n_deg] = 0.0

        # Calculate ratios
        for n_val in range(3, TEST_MAX_DEG + 1):
            # r_n^(1) = ||H_n||_1 / ||H_{n-1}||_1
            if H_norms.get(n_val - 1, 0) > 0:
                r1 = H_norms.get(n_val, 0) / H_norms[n_val - 1]
                assert r1 >= 0, f"{system_name} r^(1) ratio at degree {n_val} should be non-negative"
            
            # r_n^(2) = ||H_n||_1^(1/n)
            norm_Hn = H_norms.get(n_val, 0)
            if norm_Hn > 0:
                r2 = norm_Hn**(1.0 / n_val)
                assert r2 >= 0, f"{system_name} r^(2) ratio at degree {n_val} should be non-negative"

        print(f"{system_name} coefficient radius ratios calculated successfully")


def test_plot_coefficient_radius_diagnostics(reduction_test_setup):
    """
    Generate and save coefficient-radius diagnostics plots for both systems.
    """
    cm_em, cm_se = reduction_test_setup

    # Get Hamiltonian coefficients for both systems
    H_cm_real_em = cm_em.cache_get(('hamiltonian', TEST_MAX_DEG, 'center_manifold_real'))
    H_cm_real_se = cm_se.cache_get(('hamiltonian', TEST_MAX_DEG, 'center_manifold_real'))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Earth-Moon L1
    print("Plotting coefficient ratios for Earth-Moon L1...")
    calculate_and_plot_ratios(axes[0], H_cm_real_em, TEST_MAX_DEG, "Earth-Moon L1")

    # Plot Sun-Earth L1
    print("Plotting coefficient ratios for Sun-Earth L1...")
    calculate_and_plot_ratios(axes[1], H_cm_real_se, TEST_MAX_DEG, "Sun-Earth L1")

    fig.suptitle("Coefficient-Radius Diagnostics", fontsize=14)
    fig.supylabel("$r_n$ values", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "coefficient_radius_diagnostics.png")
    try:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
        plt.close()  # Close the figure to free memory
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close()
