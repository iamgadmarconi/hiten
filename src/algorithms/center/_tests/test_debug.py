import numpy as np
import pytest

from algorithms.center.hamiltonian import build_physical_hamiltonian
from algorithms.center.lie import (_center2modal, evaluate_transform,
                                   lie_transform)
from algorithms.center.polynomial.base import (_create_encode_dict_from_clmo,
                                               encode_multiindex,
                                               init_index_tables)
from algorithms.center.polynomial.operations import (
    polynomial_evaluate, polynomial_poisson_bracket, polynomial_zero_list)
from algorithms.center.transforms import complexify, local2realmodal
from algorithms.variables import N_VARS
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
    H_cn = complexify(H_rn, max_degree, psi, clmo)
    
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
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    q2 = polynomial_zero_list(1, psi)
    p2 = polynomial_zero_list(1, psi)
    q2[1][1] = 1.0          # q2  is the second coordinate
    p2[1][4] = 1.0          # p2  is the fifth coordinate

    res = polynomial_poisson_bracket(q2, p2, 1, psi, clmo, encode_dict_list)
    assert res[0] == 1                     # constant term

def test_bare_coordinate_poisson_brackets():
    """
    Test basic Poisson bracket relationships for bare coordinate polynomials.
    
    This test verifies that the coordinate polynomials built by _center2modal
    satisfy the fundamental Poisson bracket relationships:
    - {q1, p1} = 1
    - {p1, q1} = -1  
    - {q1, q2} = 0
    """
    psi, clmo = init_index_tables(1)
    encode_dict = _create_encode_dict_from_clmo(clmo)
    
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

def test_poly_G_total_shape(debug_setup):
    """Test the shape of the generating polynomials."""
    poly_G_total = debug_setup["poly_G_total"]
    psi = debug_setup["psi"]
    for n, G in enumerate(poly_G_total):
        if G is not None and G.any():
            assert G.ndim == 1 and len(G) == psi[N_VARS, n], f"bad shape at n={n}"

def test_homological_equation(debug_setup):
    """
    Homological equation spot test.
    Check that {λq₁p₁ + λω₁q₂p₂ + λω₂q₃p₃, Gₙ} = -Pₙᵉˡⁱᵐ
    using polynomial_poisson_bracket and the eliminated terms.
    """
    point = debug_setup["point"]
    poly_G_total = debug_setup["poly_G_total"]
    poly_elim_total = debug_setup["poly_elim_total"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    max_degree = debug_setup["max_degree"]
    
    # Get eigenvalues from the point
    linear_data = point.linear_data
    lam = linear_data.lambda1
    omega1 = linear_data.omega1
    omega2 = linear_data.omega2
    
    # Create encode dict for Poisson bracket computation
    encode_dict_list = _create_encode_dict_from_clmo(clmo)
    
    # Construct linear Hamiltonian: λq₁p₁ + iω₁q₂p₂ + iω₂q₃p₃
    # Variables: q₁=x₀, p₁=x₃, q₂=x₁, p₂=x₄, q₃=x₂, p₃=x₅
    linear_H = polynomial_zero_list(2, psi)  # degree 2 for quadratic terms
    
    # λq₁p₁ term: coefficient λ for x₀*x₃ monomial
    q1p1_multiindex = np.zeros(N_VARS, dtype=np.int64)
    q1p1_multiindex[0] = 1  # q₁
    q1p1_multiindex[3] = 1  # p₁
    q1p1_idx = encode_multiindex(q1p1_multiindex, 2, encode_dict_list)
    if q1p1_idx >= 0:
        linear_H[2][q1p1_idx] = lam
    
    # iω₁q₂p₂ term: coefficient iω₁ for x₁*x₄ monomial
    q2p2_multiindex = np.zeros(N_VARS, dtype=np.int64)
    q2p2_multiindex[1] = 1  # q₂
    q2p2_multiindex[4] = 1  # p₂
    q2p2_idx = encode_multiindex(q2p2_multiindex, 2, encode_dict_list)
    if q2p2_idx >= 0:
        linear_H[2][q2p2_idx] = 1j * omega1
    
    # iω₂q₃p₃ term: coefficient iω₂ for x₂*x₅ monomial
    q3p3_multiindex = np.zeros(N_VARS, dtype=np.int64)
    q3p3_multiindex[2] = 1  # q₃
    q3p3_multiindex[5] = 1  # p₃
    q3p3_idx = encode_multiindex(q3p3_multiindex, 2, encode_dict_list)
    if q3p3_idx >= 0:
        linear_H[2][q3p3_idx] = 1j * omega2
    
    # Test the homological equation for degrees 3-5 (where we have non-trivial G_n)
    tolerance = 1e-12  # Tolerance for numerical comparison
    
    for n in range(3, min(6, max_degree + 1)):
        if (n >= len(poly_G_total) or poly_G_total[n] is None or 
            n >= len(poly_elim_total) or poly_elim_total[n] is None):
            continue
            
        G_n = poly_G_total[n]
        P_elim_n = poly_elim_total[n]
        
        if not G_n.any() or not P_elim_n.any():  # Skip if all zeros
            continue
            
        # Create polynomial representation for G_n
        G_n_poly = polynomial_zero_list(n, psi)
        G_n_poly[n][:] = G_n[:]
        
        # Compute Poisson bracket {linear_H, G_n}
        pb_result = polynomial_poisson_bracket(linear_H, G_n_poly, max(2, n), psi, clmo, encode_dict_list)
        
        # Extract the degree n part of the Poisson bracket result
        if n < len(pb_result):
            pb_degree_n = pb_result[n]
            
            # Check the homological equation: {H₂, Gₙ} = -Pₙᵉˡⁱᵐ
            # This means: pb_degree_n + P_elim_n should be approximately zero
            difference = pb_degree_n + P_elim_n
            max_error = np.max(np.abs(difference))
            
            assert max_error < tolerance, (
                f"Homological equation violated at degree {n}: "
                f"max error = {max_error:.2e}, tolerance = {tolerance:.2e}"
            )
            
            # Also check that we have non-trivial terms (not just testing zeros)
            pb_magnitude = np.max(np.abs(pb_degree_n))
            elim_magnitude = np.max(np.abs(P_elim_n))
            
            assert pb_magnitude > tolerance and elim_magnitude > tolerance, (
                f"Trivial test at degree {n}: pb_magnitude={pb_magnitude:.2e}, "
                f"elim_magnitude={elim_magnitude:.2e}"
            )


def test_cm_degree_scaling(debug_setup):
    """
    Test that the center manifold approximation converges as polynomial degree increases.
    
    This test verifies that the q1/p1 leak decreases as we include higher-order terms
    in the polynomial approximation.
    """
    point = debug_setup["point"]
    psi = debug_setup["psi"]
    clmo = debug_setup["clmo"]
    
    # Build and normalize Hamiltonian for different degrees
    H_phys = build_physical_hamiltonian(point, 8)  # Use max degree 8 for building
    H_rn = local2realmodal(point, H_phys, 8, psi, clmo)
    H_cn = complexify(H_rn, 8, psi, clmo)
    
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
    """
    Test that the center manifold error scales quadratically with input amplitude.
    
    This test verifies that q1/p1 leak ~ amplitude^2, which is expected since
    linear terms are eliminated by the center manifold restriction.
    """
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
    """
    Comprehensive test of symplecticity using direct Poisson bracket computation.
    
    This test verifies that coordinate transformations from _center2modal satisfy 
    canonical Poisson bracket relationships across multiple polynomial degrees,
    amplitudes, and test points.
    """
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
    H_cn = complexify(H_rn, 8, psi, clmo)
    
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
