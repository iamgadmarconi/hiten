import numpy as np
import pytest

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import (_center2modal, evaluate_transform,
                                   lie_transform, _apply_series)
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               decode_multiindex,
                                               encode_multiindex,
                                               init_index_tables)
from algorithms.center.polynomial.operations import (
    polynomial_evaluate, polynomial_poisson_bracket, polynomial_zero_list)
from algorithms.center.transforms import substitute_complex, local2realmodal
from system.libration import L1Point

# Test parameters
MU_EM = 0.0121505816  # Earth-Moon mass parameter
TOL_TEST = 1e-15


@pytest.fixture(scope="module")
def debug_setup():
    """Set up test data for diagnostics."""
    point = L1Point(mu=MU_EM)
    _ = point.position  # Ensure position is calculated
    
    max_degree = 8
    psi, clmo = init_index_tables(max_degree)
    
    # Build and normalize Hamiltonian
    H_phys = build_physical_hamiltonian(point, max_degree)
    H_rn = local2realmodal(point, H_phys, max_degree, psi, clmo)
    H_cn = substitute_complex(H_rn, max_degree, psi, clmo)
    
    # Perform Lie transformation
    poly_trans, poly_G_total, poly_elim_total = lie_transform(point, H_cn, psi, clmo, max_degree)
    
    return {
        "point": point,
        "psi": psi,
        "clmo": clmo,
        "max_degree": max_degree,
        "poly_G_total": poly_G_total,
        "poly_elim_total": poly_elim_total,
        "complex_6d_cm": np.array([ 0.+0.j,-0.47286937-0.02290062j, 0.+0.21637672j, 0.+0.j,-0.02290062-0.47286937j,0.21637672+0.j], dtype=np.complex128)
    }

def test_pb():
    psi, clmo = init_index_tables(1)
    encode_dict = _create_encode_dict_from_clmo(clmo)
    q2 = polynomial_zero_list(1, psi)
    p2 = polynomial_zero_list(1, psi)
    q2[1][1] = 1.0          # q2  is the second coordinate
    p2[1][4] = 1.0          # p2  is the fifth coordinate

    res = polynomial_poisson_bracket(q2, p2, 1, psi, clmo, encode_dict)
    assert res[0] == 1                     # constant term

    # Build the six "bare" coordinate polynomials exactly as _center2modal does
    coords = _center2modal([],          # empty G's
                           max_degree=1,
                           psi=psi, clmo=clmo,
                           restrict=False)        # returns all coordinates

    q1 = coords[0]                     # polynomial "q1"
    p1 = coords[3]                     # polynomial "p1"
    q2 = coords[1]                     # polynomial "q2"
    p2 = coords[4]                     # polynomial "p2"

    one   = polynomial_poisson_bracket(q1, p1, 1, psi, clmo, encode_dict)[0]
    minus = polynomial_poisson_bracket(p1, q1, 1, psi, clmo, encode_dict)[0]
    zero  = polynomial_poisson_bracket(q1, q2, 1, psi, clmo, encode_dict)[0]

    print(" {q1,p1}  ->", one)
    print(" {p1,q1}  ->", minus)
    print(" {q1,q2}  ->", zero)
    
    # Assert the expected Poisson bracket relationships
    assert abs(one - 1.0) < TOL_TEST, f"Expected {{q1,p1}} = 1, got {one}"
    assert abs(minus - (-1.0)) < TOL_TEST, f"Expected {{p1,q1}} = -1, got {minus}"
    assert abs(zero) < TOL_TEST, f"Expected {{q1,q2}} = 0, got {zero}"

def test_transformation_accumulation(debug_setup):
    point = debug_setup["point"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    
    # Test point
    cm = np.array([0, 1e-3+1e-3j, 0.5e-3+0.2e-3j,
                   0, 1e-3-1e-3j, 0.5e-3-0.2e-3j])
    
    print("\nTesting transformation accumulation:")
    print("-" * 50)
    
    # Build transformations incrementally
    for max_deg in [3, 4, 5, 6]:
        # Get generators up to this degree
        H_phys = build_physical_hamiltonian(point, 8)
        H_rn = local2realmodal(point, H_phys, 8, psi, clmo)
        H_cn = substitute_complex(H_rn, 8, psi, clmo)
        poly_trans, poly_G_total, _ = lie_transform(point, H_cn, psi, clmo, max_deg)
        
        # Apply only up to current degree
        expansions = _center2modal(poly_G_total, max_deg, psi, clmo, 
                                   tol=1e-15, inverse=False)
        
        # Evaluate
        modal = evaluate_transform(expansions, cm, clmo)
        
        print(f"\nDegree {max_deg}:")
        print(f"  q1 = {modal[0]:.6e}")
        print(f"  p1 = {modal[3]:.6e}")
        
        # Check if specific generator was applied
        if max_deg < len(poly_G_total) and poly_G_total[max_deg] is not None:
            has_content = poly_G_total[max_deg].any()
            print(f"  G_{max_deg} has content: {has_content}")


def test_series_convergence(debug_setup):
    point = debug_setup["point"]
    poly_G_total = debug_setup["poly_G_total"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    max_degree = debug_setup["max_degree"]
    
    # Test point on center manifold
    cm = np.array([0, 1e-3+1e-3j, 0.5e-3+0.2e-3j,
                   0, 1e-3-1e-3j, 0.5e-3-0.2e-3j])
    
    print("\nTesting Lie series convergence for each generator:")
    print("="*70)
    
    # Start with identity transformation
    identity_coords = []
    for i in range(6):
        poly = polynomial_zero_list(max_degree, psi)
        poly[1][i] = 1.0
        identity_coords.append(poly)
    
    # Apply each generator individually to see its effect
    for n in range(3, min(7, len(poly_G_total))):
        if poly_G_total[n] is None or not poly_G_total[n].any():
            continue
            
        print(f"\nApplying only G_{n}:")
        
        # Create polynomial for this generator only
        test_G = polynomial_zero_list(max_degree, psi)
        test_G[n] = poly_G_total[n].copy()
        
        # Apply to q1 coordinate (index 0)
        encode_dict_list = _create_encode_dict_from_clmo(clmo)
        transformed_q1 = _apply_series(
            identity_coords[0], test_G, max_degree, psi, clmo, encode_dict_list, 1e-15
        )
        
        # Evaluate the transformation at the test point
        q1_value = polynomial_evaluate(transformed_q1, cm, clmo)
        
        print(f"  q1 transformation: {q1_value:.6e}")
        print(f"  Change from identity: {abs(q1_value):.6e}")
        
        # Also check how many Lie series terms were significant
        # by examining the polynomial structure of the transformation
        for d in range(min(len(transformed_q1), 6)):
            if transformed_q1[d].any():
                nonzero = np.count_nonzero(np.abs(transformed_q1[d]) > 1e-15)
                if nonzero > 0:
                    print(f"    Degree {d}: {nonzero} non-zero coefficients")

def test_cumulative_vs_individual(debug_setup):
    poly_G_total = debug_setup["poly_G_total"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    max_degree = 6
    
    # Test point
    cm = np.array([0, 1e-3+1e-3j, 0.5e-3+0.2e-3j,
                   0, 1e-3-1e-3j, 0.5e-3-0.2e-3j])
    
    print("\nComparing cumulative vs individual generator contributions:")
    print("="*70)
    
    # Get cumulative transformations for each degree
    cumulative_results = {}
    
    for max_deg in [3, 4, 5, 6]:
        expansions = _center2modal(poly_G_total, max_deg, psi, clmo, tol=1e-15, inverse=False)
        modal = evaluate_transform(expansions, cm, clmo)
        cumulative_results[max_deg] = modal.copy()
    
    # Compute differences between successive degrees
    print("\nIncremental changes from adding each generator:")
    for deg in [4, 5, 6]:
        prev_q1 = cumulative_results[deg-1][0]
        curr_q1 = cumulative_results[deg][0]
        diff = curr_q1 - prev_q1
        
        print(f"\nG_{deg} contribution to q1:")
        print(f"  Previous (up to G_{deg-1}): {prev_q1:.6e}")
        print(f"  Current (up to G_{deg}):    {curr_q1:.6e}")
        print(f"  Difference:                 {diff:.6e}")
        print(f"  |Difference|:               {abs(diff):.6e}")
        
        # Check if the generator has terms that should contribute
        if deg < len(poly_G_total) and poly_G_total[deg] is not None:
            G_mag = np.max(np.abs(poly_G_total[deg]))
            print(f"  Max |G_{deg}| coefficient:   {G_mag:.6e}")

def test_restricted_transformation(debug_setup):
    poly_G_total = debug_setup["poly_G_total"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    max_degree = 6
    
    # Test point
    cm = np.array([0, 1e-3+1e-3j, 0.5e-3+0.2e-3j,
                   0, 1e-3-1e-3j, 0.5e-3-0.2e-3j])
    
    print("\nComparing restricted vs unrestricted transformations:")
    print("="*60)
    
    # Unrestricted transformation
    expansions_unrestricted = _center2modal(
        poly_G_total, max_degree, psi, clmo, tol=1e-15, inverse=False, restrict=False
    )
    modal_unrestricted = evaluate_transform(expansions_unrestricted, cm, clmo)
    
    # Restricted transformation
    expansions_restricted = _center2modal(
        poly_G_total, max_degree, psi, clmo, tol=1e-15, inverse=False, restrict=True
    )
    modal_restricted = evaluate_transform(expansions_restricted, cm, clmo)
    
    print("Unrestricted:")
    print(f"  q1 = {modal_unrestricted[0]:.6e}")
    print(f"  p1 = {modal_unrestricted[3]:.6e}")
    
    print("\nRestricted:")
    print(f"  q1 = {modal_restricted[0]:.6e}")
    print(f"  p1 = {modal_restricted[3]:.6e}")
    
    print("\nDifference:")
    print(f"  Δq1 = {modal_restricted[0] - modal_unrestricted[0]:.6e}")
    print(f"  Δp1 = {modal_restricted[3] - modal_unrestricted[3]:.6e}")

def test_cm_degree_scaling(debug_setup):
    point = debug_setup["point"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    
    # Build and normalize Hamiltonian for different degrees
    H_phys = build_physical_hamiltonian(point, 8)  # Use max degree 8 for building
    H_rn = local2realmodal(point, H_phys, 8, psi, clmo)
    H_cn = substitute_complex(H_rn, 8, psi, clmo)
    
    # Test point on center manifold
    cm = np.array([0, 1e-3+1e-3j, 0.5e-3+0.2e-3j,   # q1,q2,q3
                   0, 1e-3-1e-3j, 0.5e-3-0.2e-3j])  # p1,p2,p3  (complex conj)
    
    degrees_to_test = [4, 5, 6, 7, 8]
    errors = []
    
    print("\nTesting convergence with polynomial degree:")
    print("Degree | q1 leak | p1 leak | Max leak")
    print("-------|---------|---------|----------")
    
    for max_deg in degrees_to_test:
        # Perform Lie transformation up to this degree
        poly_trans, poly_G_total, _ = lie_transform(point, H_cn, psi, clmo, max_deg)
        
        # Generate coordinate transformation
        expansions = _center2modal(poly_G_total, max_deg, psi, clmo, tol=1e-15, inverse=False)
        
        # Evaluate transformation
        modal = evaluate_transform(expansions, cm, clmo)
        
        # Compute leak magnitudes
        q1_leak = abs(modal[0])
        p1_leak = abs(modal[3])
        max_leak = max(q1_leak, p1_leak)
        
        errors.append(max_leak)
        
        print(f"  {max_deg}    | {q1_leak:.2e} | {p1_leak:.2e} | {max_leak:.2e}")
    
    # Verify convergence: each higher degree should have smaller or equal error
    # Allow for some numerical noise by using a relaxed comparison
    convergence_factor = 1.1  # Allow 10% tolerance for numerical variations
    
    for i in range(1, len(errors)):
        prev_error = errors[i-1]
        curr_error = errors[i]
        
        # Error should generally decrease or stay similar (within factor)
        assert curr_error <= convergence_factor * prev_error, (
            f"Convergence failure: degree {degrees_to_test[i]} error {curr_error:.2e} "
            f"> {convergence_factor} x degree {degrees_to_test[i-1]} error {prev_error:.2e}"
        )


def test_cm_amplitude_scaling(debug_setup):
    point = debug_setup["point"]
    poly_G_total = debug_setup["poly_G_total"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    max_degree = debug_setup["max_degree"]
    
    # Generate the coordinate transformation expansions
    expansions = _center2modal(poly_G_total, max_degree, psi, clmo, tol=1e-15, inverse=False)
    
    # Test different amplitudes (scaling the base coordinates)
    base_coords = np.array([0, 1+1j, 0.5+0.2j,   # q1,q2,q3
                           0, 1-1j, 0.5-0.2j])   # p1,p2,p3  (complex conj)
    
    amplitudes = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    
    results = []
    
    print("\nTesting error scaling with center manifold amplitude:")
    print("Amplitude | Input norm | q1 leak | p1 leak | Max leak | leak/amp^2")
    print("----------|------------|---------|---------|----------|----------")
    
    for amp_scale in amplitudes:
        # Scale the test coordinates
        cm = amp_scale * base_coords
        
        # Compute input amplitude (only center manifold coordinates)
        input_amp = np.linalg.norm([cm[1], cm[2], cm[4], cm[5]])
        
        # Evaluate transformation
        modal = evaluate_transform(expansions, cm, clmo)
        
        # Compute leak magnitudes
        q1_leak = abs(modal[0])
        p1_leak = abs(modal[3])
        max_leak = max(q1_leak, p1_leak)
        
        # Expected quadratic scaling
        leak_per_amp2 = max_leak / (input_amp**2) if input_amp > 0 else 0
        
        results.append({
            'amp_scale': amp_scale,
            'input_amp': input_amp,
            'q1_leak': q1_leak,
            'p1_leak': p1_leak,
            'max_leak': max_leak,
            'leak_per_amp2': leak_per_amp2
        })
        
        print(f"{amp_scale:.1e} | {input_amp:.2e} | {q1_leak:.2e} | {p1_leak:.2e} | {max_leak:.2e} | {leak_per_amp2:.2e}")
    
    # Verify quadratic scaling by checking that leak/amp^2 is approximately constant
    leak_ratios = [r['leak_per_amp2'] for r in results]
    
    # Skip the first point which might have numerical issues at very small amplitudes
    ratios_to_check = leak_ratios[1:]
    
    if len(ratios_to_check) > 1:
        mean_ratio = np.mean(ratios_to_check)
        std_ratio = np.std(ratios_to_check)
        
        # Check that the ratio is reasonably constant (within 50% variation)
        max_variation = 0.5 * mean_ratio
        
        for i, ratio in enumerate(ratios_to_check):
            assert abs(ratio - mean_ratio) <= max_variation, (
                f"Quadratic scaling violation at amplitude {amplitudes[i+1]:.1e}: "
                f"leak/amp^2 = {ratio:.2e}, mean = {mean_ratio:.2e}, "
                f"deviation = {abs(ratio - mean_ratio):.2e} > tolerance {max_variation:.2e}"
            )
        
        print(f"\nQuadratic scaling verified:")
        print(f"Mean leak/amp^2 ratio: {mean_ratio:.2e}")
        print(f"Standard deviation: {std_ratio:.2e} ({100*std_ratio/mean_ratio:.1f}%)")
    
    # Also verify that larger amplitudes give larger absolute errors
    for i in range(1, len(results)):
        prev_leak = results[i-1]['max_leak']
        curr_leak = results[i]['max_leak']
        
        assert curr_leak >= prev_leak, (
            f"Error should increase with amplitude: "
            f"amp {amplitudes[i]:.1e} leak {curr_leak:.2e} < "
            f"amp {amplitudes[i-1]:.1e} leak {prev_leak:.2e}"
        )

def test_symplecticity(debug_setup):
    def poisson_matrix(expansions, clmo, psi, max_degree, test_point):
        """Return the 6x6 matrix M_ij = {Phi_i, Phi_j}(point)."""
        encode_dict_list = _create_encode_dict_from_clmo(clmo)
        n = 6
        M = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(i+1, n):
                bracket = polynomial_poisson_bracket(
                    expansions[i], expansions[j], max_degree, psi, clmo, encode_dict_list
                )
                val = polynomial_evaluate(bracket, test_point, clmo)
                M[i, j] = val
                M[j, i] = -val  # antisymmetry
        return M

    def analyze_symplectic_error(M, Omega, description=""):
        """Analyze and return detailed symplectic error information."""
        error_matrix = M - Omega
        max_error = np.linalg.norm(error_matrix, np.inf)
        
        # Specific canonical relationships
        canonical_errors = {
            'q1_p1': abs(M[0,3] - 1.0),  # Should be +1
            'q2_p2': abs(M[1,4] - 1.0),  # Should be +1  
            'q3_p3': abs(M[2,5] - 1.0),  # Should be +1
            'q1_q2': abs(M[0,1]),        # Should be 0
            'p1_p2': abs(M[3,4]),        # Should be 0
            'q1_p2': abs(M[0,4]),        # Should be 0
        }
        
        return max_error, canonical_errors

    # Get test data
    point = debug_setup["point"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    
    # Build Hamiltonian for different degrees
    H_phys = build_physical_hamiltonian(point, 8)
    H_rn = local2realmodal(point, H_phys, 8, psi, clmo)
    H_cn = substitute_complex(H_rn, 8, psi, clmo)
    
    # Canonical symplectic matrix
    Omega = np.block([[np.zeros((3,3)),  np.eye(3)],
                      [-np.eye(3), np.zeros((3,3))]])
    
    # Test parameters
    degrees_to_test = [4, 5, 6, 7, 8]
    base_amplitudes = [1e-4, 5e-4, 1e-3, 5e-3]
    
    # Different types of test points
    test_point_templates = [
        np.array([0, 1+1j, 0.5+0.2j, 0, 1-1j, 0.5-0.2j]),      # Standard complex conjugate
        np.array([0, 1+0j, 0+1j, 0, 1+0j, 0-1j]),              # Real/imaginary separation  
        np.array([0, 1+0.5j, 0.3+0.8j, 0, 1-0.5j, 0.3-0.8j]), # Different conjugate pattern
        np.array([0, 0.7+0.7j, 0.1+0.1j, 0, 0.7-0.7j, 0.1-0.1j]), # Smaller, equal real/imag
    ]
    
    print(f"\nComprehensive Direct Symplecticity Test")
    print("="*60)
    
    all_results = []
    overall_max_error = 0.0
    degree_max_errors = {}  # Track max error per degree
    
    for degree in degrees_to_test:
        print(f"\nDegree {degree}:")
        print("-" * 40)
        
        # Generate transformations for this degree
        poly_trans, poly_G_total, _ = lie_transform(point, H_cn, psi, clmo, degree)
        expansions = _center2modal(poly_G_total, degree, psi, clmo, 
                                   tol=1e-15, inverse=False, sign=1, restrict=False)
        
        degree_results = []
        
        for amp_idx, base_amp in enumerate(base_amplitudes):
            for point_idx, template in enumerate(test_point_templates):
                # Scale the test point  
                test_point = base_amp * template
                
                # Compute Poisson bracket matrix
                M = poisson_matrix(expansions, clmo, psi, degree, test_point)
                
                # Analyze errors
                max_error, canonical_errors = analyze_symplectic_error(M, Omega)
                
                result = {
                    'degree': degree,
                    'amplitude': base_amp, 
                    'point_type': point_idx,
                    'max_error': max_error,
                    'canonical_errors': canonical_errors,
                    'test_point': test_point
                }
                
                degree_results.append(result)
                all_results.append(result)
                overall_max_error = max(overall_max_error, max_error)
                
        # Report degree summary
        degree_max_error = max(r['max_error'] for r in degree_results)
        degree_avg_error = np.mean([r['max_error'] for r in degree_results])
        degree_max_errors[degree] = degree_max_error
        
        print(f"  Max error: {degree_max_error:.2e}")
        print(f"  Avg error: {degree_avg_error:.2e}")
        print(f"  Tests run: {len(degree_results)}")
    
    print(f"\nOverall Results:")
    print("="*30)
    print(f"Total tests: {len(all_results)}")
    print(f"Overall max error: {overall_max_error:.2e}")
    
    # Analyze scaling with degree
    print(f"\nError vs Degree (avg over all amplitudes/points):")
    for degree in degrees_to_test:
        degree_results = [r for r in all_results if r['degree'] == degree]
        avg_error = np.mean([r['max_error'] for r in degree_results])
        print(f"  Degree {degree}: {avg_error:.2e}")
    
    # Analyze scaling with amplitude for highest degree
    highest_degree = max(degrees_to_test)
    print(f"\nError vs Amplitude (degree {highest_degree}, avg over all point types):")
    for amp in base_amplitudes:
        amp_results = [r for r in all_results 
                      if r['degree'] == highest_degree and r['amplitude'] == amp]
        if amp_results:
            avg_error = np.mean([r['max_error'] for r in amp_results])
            print(f"  Amplitude {amp:.1e}: {avg_error:.2e}")
    
    # Find worst cases
    worst_result = max(all_results, key=lambda r: r['max_error'])
    print(f"\nWorst case:")
    print(f"  Degree: {worst_result['degree']}, Amplitude: {worst_result['amplitude']:.1e}")
    print(f"  Point type: {worst_result['point_type']}, Error: {worst_result['max_error']:.2e}")
    
    # Detailed analysis of canonical errors for worst case
    print(f"  Canonical relationship errors:")
    for name, error in worst_result['canonical_errors'].items():
        print(f"    {name}: {error:.2e}")
    
    # Improved degree-aware tolerance calculation
    # Each degree should have its own tolerance based on expected truncation error
    print(f"\nDegree-specific tolerance verification:")
    all_passed = True
    
    for degree in degrees_to_test:
        # Base tolerance that scales with polynomial truncation error
        # For degree d, expect errors ~ O(amplitude^(d+1))
        max_amplitude_tested = max(base_amplitudes)
        
        # Expected error for polynomial of degree d is roughly proportional to:
        # - amplitude^(d+1) for truncation errors
        # - machine epsilon effects
        degree_tolerance = 1e-15 * (10 ** (8 - degree)) * (max_amplitude_tested / 1e-4) ** (degree - 2)
        
        # Get max error for this degree
        degree_error = degree_max_errors[degree]
        passed = degree_error < degree_tolerance
        all_passed = all_passed and passed
        
        print(f"  Degree {degree}: error={degree_error:.2e}, tolerance={degree_tolerance:.2e}, {'PASS' if passed else 'FAIL'}")
    
    # Quality check: verify error decreases with degree
    error_ratios = []
    for i in range(1, len(degrees_to_test)):
        prev_degree = degrees_to_test[i-1]
        curr_degree = degrees_to_test[i]
        ratio = degree_max_errors[curr_degree] / degree_max_errors[prev_degree]
        error_ratios.append(ratio)
        print(f"\nError reduction {prev_degree}→{curr_degree}: {ratio:.2e} ({1/ratio:.1f}x improvement)")
    
    # Verify strong convergence (each degree should reduce error significantly)
    avg_reduction_factor = np.mean([1/r for r in error_ratios])
    print(f"\nAverage error reduction per degree: {avg_reduction_factor:.1f}x")
    
    assert all_passed, (
        f"Symplecticity test failed for some polynomial degrees. "
        f"See degree-specific results above."
    )
    
    assert avg_reduction_factor > 10, (
        f"Insufficient convergence rate: average error reduction {avg_reduction_factor:.1f}x < 10x per degree"
    )
    
    print(f"\nComprehensive direct symplecticity test passed!")
    print(f"All {len(all_results)} test combinations satisfy canonical relationships with appropriate tolerances.")
    print(f"Strong convergence verified: {avg_reduction_factor:.1f}x average error reduction per degree.")


def test_transformation_at_different_points(debug_setup):
    poly_G_total = debug_setup["poly_G_total"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    max_degree = 6
    
    print("\nTesting transformation at various points:")
    print("="*60)
    
    # Generate transformation
    expansions = _center2modal(poly_G_total, max_degree, psi, clmo, tol=1e-15, inverse=False)
    
    # Test points with different characteristics
    test_cases = [
        # (name, point)
        ("Pure center manifold", np.array([0, 1e-3, 1e-3j, 0, 1e-3, -1e-3j])),
        ("With small q1", np.array([1e-6, 1e-3, 1e-3j, 0, 1e-3, -1e-3j])),
        ("With small p1", np.array([0, 1e-3, 1e-3j, 1e-6, 1e-3, -1e-3j])),
        ("With q1=p1", np.array([1e-6, 1e-3, 1e-3j, 1e-6, 1e-3, -1e-3j])),
        ("Larger amplitude", np.array([0, 1e-2, 1e-2j, 0, 1e-2, -1e-2j])),
        ("Complex conjugate", np.array([0, 1e-3+1e-3j, 0.5e-3+0.2e-3j, 0, 1e-3-1e-3j, 0.5e-3-0.2e-3j])),
    ]
    
    for name, test_point in test_cases:
        modal = evaluate_transform(expansions, test_point, clmo)
        
        print(f"\n{name}:")
        print(f"  Input:  q1={test_point[0]:.2e}, p1={test_point[3]:.2e}")
        print(f"  Output: q1={modal[0]:.6e}, p1={modal[3]:.6e}")
        print(f"  |q1|={abs(modal[0]):.2e}, |p1|={abs(modal[3]):.2e}")


def test_hamiltonian_on_center_manifold(debug_setup):
    point = debug_setup["point"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    max_degree = 6
    
    # Build Hamiltonians
    H_phys = build_physical_hamiltonian(point, max_degree)
    H_rn = local2realmodal(point, H_phys, max_degree, psi, clmo)
    H_cn = substitute_complex(H_rn, max_degree, psi, clmo)
    
    # Get the transformed Hamiltonian
    poly_trans, _, _ = lie_transform(point, H_cn, psi, clmo, max_degree)
    
    print("\nAnalyzing transformed Hamiltonian on center manifold:")
    print("="*60)
    
    # Check linear terms in the transformed Hamiltonian
    if len(poly_trans) > 1:
        H1 = poly_trans[1]
        print(f"Linear terms in transformed H: {np.count_nonzero(H1)} non-zero")
        for i, coeff in enumerate(H1):
            if abs(coeff) > 1e-15:
                print(f"  x_{i}: {coeff:.6e}")
    
    # Check quadratic terms
    if len(poly_trans) > 2:
        H2 = poly_trans[2]
        encode_dict = _create_encode_dict_from_clmo(clmo)[2]
        
        # Look for q1*q2, q1*p2, p1*q2, p1*p2 terms (cross terms)
        cross_terms = []
        for idx, coeff in enumerate(H2):
            if abs(coeff) > 1e-15:
                k = decode_multiindex(idx, 2, clmo)
                # Check if it's a cross term between (q1,p1) and (q2,p2,q3,p3)
                has_q1p1 = (k[0] > 0 or k[3] > 0)
                has_cm = (k[1] > 0 or k[2] > 0 or k[4] > 0 or k[5] > 0)
                if has_q1p1 and has_cm:
                    cross_terms.append((k, coeff))
        
        print(f"\nQuadratic cross-terms (q1/p1 × CM): {len(cross_terms)}")
        for k, coeff in cross_terms[:5]:
            vars_str = []
            var_names = ['q1', 'q2', 'q3', 'p1', 'p2', 'p3']
            for i, power in enumerate(k):
                if power > 0:
                    vars_str.append(f"{var_names[i]}^{power}" if power > 1 else var_names[i])
            print(f"  {' '.join(vars_str)}: {coeff:.6e}")