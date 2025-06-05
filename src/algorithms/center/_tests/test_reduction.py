import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from algorithms.center.manifold import center_manifold_real
from algorithms.center.polynomial.base import init_index_tables
from system.base import System, systemConfig
from system.body import Body
from utils.constants import Constants


def get_hamiltonian_coefficients(
    system_key: str,  # "SE" for Sun-Earth, "EM" for Earth-Moon
    libration_point_number: int,  # 1 or 2
    max_deg_coeffs: int
):
    """
    Computes Hamiltonian coefficients for a given system and libration point.
    """
    psi, clmo = init_index_tables(max_deg_coeffs)
    
    system_to_use = None
    if system_key == "SE":  # Sun-Earth system
        # Sun is m1 (primary), Earth is m2 (secondary)
        sun_body = Body("Sun", Constants.bodies["sun"]["mass"], Constants.bodies["sun"]["radius"], "yellow")
        earth_body = Body("Earth", Constants.bodies["earth"]["mass"], Constants.bodies["earth"]["radius"], "blue", sun_body) # Earth's parent is Sun
        d_SE = Constants.get_orbital_distance("sun", "earth")
        system_to_use = System(systemConfig(sun_body, earth_body, d_SE))
        
    elif system_key == "EM":  # Earth-Moon system
        # Earth is m1 (primary), Moon is m2 (secondary)
        # In this specific RTBP context, Earth does not have Sun as parent.
        earth_body_for_EM = Body("Earth", Constants.bodies["earth"]["mass"], Constants.bodies["earth"]["radius"], "blue") # No parent
        moon_body = Body("Moon", Constants.bodies["moon"]["mass"], Constants.bodies["moon"]["radius"], "gray", earth_body_for_EM) # Moon's parent is Earth
        d_EM = Constants.get_orbital_distance("earth", "moon")
        system_to_use = System(systemConfig(earth_body_for_EM, moon_body, d_EM))
    else:
        raise ValueError(f"Unsupported system key: {system_key}. Choose 'SE' or 'EM'.")

    l_point_obj = system_to_use.get_libration_point(libration_point_number)
    
    # H_cm_coeffs[k] are coefficients for degree k. Starts from H_0, typically H_0, H_1 are zero/not used.
    # Useful terms H_n start from n=2.
    H_cm_coeffs = center_manifold_real(l_point_obj, psi, clmo, max_deg_coeffs)
    return H_cm_coeffs


def calculate_and_plot_ratios(
    ax,  # Matplotlib axis object
    H_cm_full_coeffs,  # List of coefficient arrays, H_cm_full_coeffs[k] is for degree k
    max_n_to_plot: int,  # Max n for r_n (e.g., 32)
    plot_title: str
):
    """
    Calculates r_n^(1) and r_n^(2) and plots them.
    H_n means coefficients of degree n. The list H_cm_full_coeffs is 0-indexed by degree.
    So, H_cm_full_coeffs[k] corresponds to degree k.
    The paper plots for n=3...32. This requires norms of H_n for n=2...32.
    """
    if len(H_cm_full_coeffs) <= max_n_to_plot:
        raise ValueError(
            f"Hamiltonian coefficients available up to degree {len(H_cm_full_coeffs)-1}, "
            f"but plotting requires up to degree {max_n_to_plot}."
        )

    H_norms_1 = {}  # Keyed by degree n
    # We need norms for H_n where n ranges from 2 to max_n_to_plot.
    for n_deg in range(2, max_n_to_plot + 1):
        if H_cm_full_coeffs[n_deg] is not None and len(H_cm_full_coeffs[n_deg]) > 0:
            H_norms_1[n_deg] = np.sum(np.abs(H_cm_full_coeffs[n_deg]))
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
    # ax.set_ylabel('$r_n$ values') # Common Y label can be set on figure
    ax.set_title(plot_title)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Set y-axis limits and x-ticks based on figures in the paper
    if "L1" in plot_title or "L₁" in plot_title:  # Fig 4 (Sun-Earth L1)
        ax.set_ylim(0.9, 1.4)
    elif "L2" in plot_title or "L₂" in plot_title:  # Fig 5 (Earth-Moon L2)
        ax.set_ylim(0.4, 1.8) # Original image looks like 0.4 to 1.8
    
    ax.set_xticks(np.arange(0, max_n_to_plot + 1, 5))
    ax.set_xlim(0, max_n_to_plot + 2)


def test_plot_coefficient_radius_diagnostics():
    """
    Generates and plots coefficient-radius diagnostics for SE L1 and EM L2.
    """
    TEST_MAX_DEG_N = 16

    fig, axes = plt.subplots(2, 1, figsize=(7, 9)) # Adjusted for typical paper figure style

    # --- Fig. 4: Sun-Earth L1 ---
    print(f"Calculating Hamiltonian for Sun-Earth L1 up to degree {TEST_MAX_DEG_N}...")
    H_cm_SE_L1 = get_hamiltonian_coefficients("SE", 1, TEST_MAX_DEG_N)
    print("Plotting for Sun-Earth L1...")
    calculate_and_plot_ratios(axes[0], H_cm_SE_L1, TEST_MAX_DEG_N, "Fig. 4 Replica: Sun-Earth $L_1$")

    # --- Fig. 5: Earth-Moon L2 ---
    print(f"Calculating Hamiltonian for Earth-Moon L2 up to degree {TEST_MAX_DEG_N}...")
    H_cm_EM_L2 = get_hamiltonian_coefficients("EM", 2, TEST_MAX_DEG_N)
    print("Plotting for Earth-Moon L2...")
    calculate_and_plot_ratios(axes[1], H_cm_EM_L2, TEST_MAX_DEG_N, "Fig. 5 Replica: Earth-Moon $L_2$")

    fig.suptitle("Coefficient-Radius Diagnostics", fontsize=14)
    fig.supylabel("$r_n$ values", fontsize=10) # Common Y label for the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title/label overlap
    
    # Save the plot to the same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(script_dir, "coefficient_radius_diagnostics.png")
    try:
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
        # To display the plot if in an interactive environment:
        # plt.show()
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == '__main__':
    # This block allows running the test script directly.
    # Ensure that the project's src directory is in PYTHONPATH if you run this directly,
    # or that the relative paths for imports work from your current execution directory.
    print("Running coefficient-radius diagnostic test...")
    
    # Simple check for src path addition if run directly from _tests folder
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Expected structure: cr3bpv2/src/algorithms/center/poincare/_tests/test_flow.py
    # We want cr3bpv2/src to be in path
    src_dir_guess = os.path.abspath(os.path.join(current_script_dir, "..", "..", "..", "..", "src"))

    # A more general way to find the project root (assuming 'cr3bpv2' is a unique part of the path)
    path_parts = current_script_dir.split(os.sep)
    try:
        project_root_index = path_parts.index("cr3bpv2")
        project_root_path = os.sep.join(path_parts[:project_root_index+1])
        src_main_path = os.path.join(project_root_path, "src")
        if src_main_path not in sys.path:
            print(f"Adding to sys.path for direct execution: {src_main_path}")
            sys.path.insert(0, src_main_path)
            # Re-evaluate imports if they failed initially (Python caches import failures sometimes)
            import importlib
            if 'system.base' in sys.modules: importlib.reload(sys.modules['system.base'])
            if 'system.body' in sys.modules: importlib.reload(sys.modules['system.body'])
            if 'utils.constants' in sys.modules: importlib.reload(sys.modules['utils.constants'])
            if 'algorithms.center.manifold' in sys.modules: importlib.reload(sys.modules['algorithms.center.manifold'])
            if 'algorithms.center.polynomial.base' in sys.modules: importlib.reload(sys.modules['algorithms.center.polynomial.base'])
            
            from algorithms.center.manifold import center_manifold_real
            from algorithms.center.polynomial.base import init_index_tables
            from system.base import System, systemConfig
            from system.body import Body
            from utils.constants import Constants

    except ValueError:
        print("Could not automatically determine project root 'cr3bpv2' from path. Imports might fail.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"sys.path: {sys.path}")


    test_plot_coefficient_radius_diagnostics()
    print("Test finished. Check for 'coefficient_radius_diagnostics.png'.")
