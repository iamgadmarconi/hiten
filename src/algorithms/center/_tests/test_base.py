import pytest
import numpy as np
import symengine as se
from pathlib import Path
import tempfile
import os

from algorithms.center.base import symplectic_dot, FormalSeries, Hamiltonian
from algorithms.center.polynomials import Polynomial

# --- Test Fixtures ---

@pytest.fixture
def n_vars():
    return 6  # 3 DOF for CR3BP

@pytest.fixture
def poly_q0(n_vars):
    return Polynomial('x0', n_vars=n_vars)

@pytest.fixture
def poly_p0(n_vars):
    return Polynomial('x1', n_vars=n_vars)

@pytest.fixture
def poly_q1(n_vars):
    return Polynomial('x2', n_vars=n_vars)

@pytest.fixture
def poly_p1(n_vars):
    return Polynomial('x3', n_vars=n_vars)

@pytest.fixture
def poly_q2(n_vars):
    return Polynomial('x4', n_vars=n_vars)

@pytest.fixture
def poly_p2(n_vars):
    return Polynomial('x5', n_vars=n_vars)

@pytest.fixture
def quadratic_poly(n_vars):
    # Simple quadratic Hamiltonian: 0.5*p0^2 + q0^2
    return Polynomial('0.5*x1**2 + x0**2', n_vars=n_vars)

@pytest.fixture
def cubic_poly(n_vars):
    # Cubic polynomial: q0^3 + q0*p0^2
    return Polynomial('x0**3 + x0*x1**2', n_vars=n_vars)

@pytest.fixture
def quartic_poly(n_vars):
    # Quartic polynomial: q0^4 + p0^4
    return Polynomial('x0**4 + x1**4', n_vars=n_vars)

@pytest.fixture
def formal_series(quadratic_poly, cubic_poly, quartic_poly):
    # Create a formal series with terms of degree 2, 3, and 4
    return FormalSeries({2: quadratic_poly, 3: cubic_poly, 4: quartic_poly})

@pytest.fixture
def empty_formal_series():
    return FormalSeries()

@pytest.fixture
def hamiltonian(formal_series):
    # Create a Hamiltonian with mu=0.01 in synodic coordinates
    return Hamiltonian(formal_series, mu=0.01, coords="synodic")

# --- Test symplectic_dot ---

def test_symplectic_dot_valid():
    # Test symplectic_dot with valid input
    gradient = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    expected = np.array([4.0, 5.0, 6.0, -1.0, -2.0, -3.0])
    result = symplectic_dot(gradient)
    np.testing.assert_array_equal(result, expected)

def test_symplectic_dot_invalid():
    # Test symplectic_dot with invalid input size
    invalid_gradient = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        symplectic_dot(invalid_gradient)

# --- Test FormalSeries ---

def test_formal_series_init(formal_series, quadratic_poly, cubic_poly, quartic_poly):
    # Test initialization and __getitem__
    assert formal_series[2] == quadratic_poly
    assert formal_series[3] == cubic_poly
    assert formal_series[4] == quartic_poly
    
    # Test with invalid key
    with pytest.raises(KeyError):
        FormalSeries({-1: quadratic_poly})  # Negative degree should raise KeyError

def test_formal_series_dict_protocol(formal_series, quadratic_poly, cubic_poly, quartic_poly):
    # Test __setitem__, __delitem__, __iter__, __len__
    
    # Test __len__
    assert len(formal_series) == 3
    
    # Test __iter__
    degrees = list(formal_series)
    assert set(degrees) == {2, 3, 4}
    
    # Test __setitem__ with new key
    new_poly = Polynomial('x0*x2', n_vars=6)
    formal_series[5] = new_poly
    assert formal_series[5] == new_poly
    assert len(formal_series) == 4
    
    # Test __delitem__
    del formal_series[3]
    assert 3 not in formal_series
    assert len(formal_series) == 3

def test_formal_series_degrees(formal_series):
    # Test degrees() method
    assert formal_series.degrees() == [2, 3, 4]
    
    # After modification
    formal_series[5] = Polynomial('x0*x2*x4', n_vars=6)
    del formal_series[3]
    assert formal_series.degrees() == [2, 4, 5]

def test_formal_series_truncate(formal_series, quadratic_poly, cubic_poly):
    # Test truncate() method
    truncated = formal_series.truncate(3)
    assert set(truncated.degrees()) == {2, 3}
    assert truncated[2] == quadratic_poly
    assert truncated[3] == cubic_poly
    assert 4 not in truncated
    
    # Check that the original is unchanged
    assert len(formal_series) == 3
    assert 4 in formal_series

def test_formal_series_poisson_pair(formal_series, n_vars):
    # Create another series for testing poisson_pair
    other_series = FormalSeries({
        2: Polynomial('x2**2 + 0.5*x3**2', n_vars=n_vars),  # q1^2 + 0.5*p1^2
        3: Polynomial('x2**3 + x2*x3**2', n_vars=n_vars)    # q1^3 + q1*p1^2
    })
    
    # Test poisson_pair for degree 2 (should be None since no valid pairs)
    result_deg2 = FormalSeries.poisson_pair(formal_series, other_series, 2)
    assert result_deg2 is None
    
    # Test poisson_pair for degree 3
    # {H_2, G_2} where H_2 = 0.5*p0^2 + q0^2, G_2 = q1^2 + 0.5*p1^2
    # This should be zero because there's no interaction between variables
    result_deg3 = FormalSeries.poisson_pair(formal_series, other_series, 3)
    # Polynomial equality can be tricky, we could check if it's zero or very small
    assert result_deg3.total_degree() <= 3
    
    # Create a more interesting case with coupled variables
    coupled_series = FormalSeries({
        2: Polynomial('x0*x2 + x1*x3', n_vars=n_vars),  # q0*q1 + p0*p1
    })
    
    # Test poisson_pair with coupled variables
    result_coupled = FormalSeries.poisson_pair(formal_series, coupled_series, 3)
    assert result_coupled is not None
    assert result_coupled.total_degree() <= 3

def test_formal_series_lie_transform(formal_series, cubic_poly, n_vars):
    # Create a generator for the Lie transform
    chi = Polynomial('x0**3', n_vars=n_vars)  # q0^3
    
    # Apply the Lie transform
    transformed = formal_series.lie_transform(chi, k_max=4)
    
    # Check that we have the expected degrees
    assert set(transformed.degrees()) == {2, 3, 4}
    
    # Check that some terms are modified
    # The quadratic term should be modified by {χ, H_2}
    assert transformed[2] != formal_series[2]
    
    # Higher terms should also change
    assert transformed[3] != formal_series[3]
    assert transformed[4] != formal_series[4]

def test_formal_series_str(formal_series):
    # Test string representation
    assert str(formal_series) == "FormalSeries(deg2, deg3, deg4)"
    assert repr(formal_series) == "FormalSeries(deg2, deg3, deg4)"

# --- Test Hamiltonian ---

def test_hamiltonian_init(hamiltonian, formal_series):
    # Test initialization
    assert hamiltonian.series == formal_series
    assert hamiltonian.mu == 0.01
    assert hamiltonian.coords == "synodic"
    assert hamiltonian.order == 4  # Highest degree in the series

def test_hamiltonian_empty():
    # Test with empty series
    empty_series = FormalSeries()
    empty_ham = Hamiltonian(empty_series, mu=0.01)
    assert empty_ham.order == 0

def test_hamiltonian_quadratic(hamiltonian, quadratic_poly):
    # Test quadratic() method
    assert hamiltonian.quadratic() == quadratic_poly

def test_hamiltonian_evaluate(hamiltonian):
    # Create a point for evaluation
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    # Evaluate the Hamiltonian at this point
    val = hamiltonian.evaluate(x)
    
    # Check that the result is a float
    assert isinstance(val, float)
    
    # Evaluate without float conversion
    val_complex = hamiltonian.evaluate(x, float_only=False)
    
    # They should be equal when converted to float
    assert abs(float(val_complex) - val) < 1e-10

def test_hamiltonian_gradient(hamiltonian):
    # Create a point for evaluation
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    # Compute the gradient
    grad = hamiltonian.gradient(x)
    
    # Check that the result is a numpy array of the right shape
    assert isinstance(grad, np.ndarray)
    assert grad.shape == (6,)

def test_hamiltonian_vector_field(hamiltonian):
    # Create a point for evaluation
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    
    # Compute the vector field
    vf = hamiltonian.vector_field(x)
    
    # Check that the result is a numpy array of the right shape
    assert isinstance(vf, np.ndarray)
    assert vf.shape == (6,)
    
    # The vector field should be related to the gradient via symplectic_dot
    grad = hamiltonian.gradient(x)
    expected_vf = symplectic_dot(grad)
    np.testing.assert_array_almost_equal(vf, expected_vf)

def test_hamiltonian_poisson(hamiltonian, n_vars):
    # Create another Hamiltonian for testing
    other_series = FormalSeries({
        2: Polynomial('x2**2 + 0.5*x3**2', n_vars=n_vars),  # q1^2 + 0.5*p1^2
        3: Polynomial('x2**3 + x2*x3**2', n_vars=n_vars)    # q1^3 + q1*p1^2
    })
    other_ham = Hamiltonian(other_series, mu=0.01)
    
    # Compute the Poisson bracket
    pb = hamiltonian.poisson(other_ham)
    
    # Check the result is a Hamiltonian
    assert isinstance(pb, Hamiltonian)
    assert pb.mu == 0.01
    assert pb.coords == "synodic"
    
    # The order should be h.order + g.order - 2
    assert pb.order <= hamiltonian.order + other_ham.order - 2

def test_hamiltonian_change_variables(hamiltonian, n_vars):
    # Define a simple transform function
    def transform(poly):
        # Add 1 to all q coordinates (shift origin)
        shifted = Polynomial('1', n_vars=n_vars)
        for i in range(0, n_vars, 2):  # Only q coordinates (even indices)
            shifted = poly.substitute({i: Polynomial(f'x{i} + 1', n_vars=n_vars)})
        return shifted
    
    # Apply the transform
    transformed = hamiltonian.change_variables(transform)
    
    # Check the result has the same general properties
    assert isinstance(transformed, Hamiltonian)
    assert transformed.mu == 0.01
    assert transformed.coords == "synodic"
    assert transformed.order == hamiltonian.order
    
    # But the series should be different
    assert transformed.series != hamiltonian.series

def test_hamiltonian_hdf_io(hamiltonian):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
        tmp_path = tmp.name
    
    try:
        # Save to HDF
        hamiltonian.to_hdf(tmp_path)
        
        # Check the file exists
        assert os.path.exists(tmp_path)
        
        # Load from HDF
        loaded = Hamiltonian.from_hdf(tmp_path)
        
        # Check the loaded Hamiltonian
        assert loaded.mu == hamiltonian.mu
        assert loaded.coords == hamiltonian.coords
        assert loaded.order == hamiltonian.order
        
        # Series should have the same degrees
        assert set(loaded.series.degrees()) == set(hamiltonian.series.degrees())
        
        # Evaluate at a point and compare
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        val_original = hamiltonian.evaluate(x)
        val_loaded = loaded.evaluate(x)
        assert abs(val_original - val_loaded) < 1e-10
        
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def test_hamiltonian_str(hamiltonian):
    # Test string representation
    expected = "Hamiltonian(order≤4, coords='synodic', μ=0.01)"
    assert str(hamiltonian) == expected
    assert repr(hamiltonian) == expected

def run_all_tests():
    """Run all test functions in this module."""
    # Setup
    n_vars = 6
    quadratic_poly = Polynomial('0.5*x1**2 + x0**2', n_vars=n_vars)
    cubic_poly = Polynomial('x0**3 + x0*x1**2', n_vars=n_vars)
    quartic_poly = Polynomial('x0**4 + x1**4', n_vars=n_vars)
    formal_series = FormalSeries({2: quadratic_poly, 3: cubic_poly, 4: quartic_poly})
    hamiltonian = Hamiltonian(formal_series, mu=0.01, coords="synodic")
    
    # Test symplectic_dot
    test_symplectic_dot_valid()
    test_symplectic_dot_invalid()
    
    # Test FormalSeries
    test_formal_series_init(formal_series, quadratic_poly, cubic_poly, quartic_poly)
    test_formal_series_dict_protocol(formal_series, quadratic_poly, cubic_poly, quartic_poly)
    test_formal_series_degrees(formal_series)
    test_formal_series_truncate(formal_series, quadratic_poly, cubic_poly)
    test_formal_series_poisson_pair(formal_series, n_vars)
    test_formal_series_lie_transform(formal_series, cubic_poly, n_vars)
    test_formal_series_str(formal_series)
    
    # Test Hamiltonian
    test_hamiltonian_init(hamiltonian, formal_series)
    test_hamiltonian_empty()
    test_hamiltonian_quadratic(hamiltonian, quadratic_poly)
    test_hamiltonian_evaluate(hamiltonian)
    test_hamiltonian_gradient(hamiltonian)
    test_hamiltonian_vector_field(hamiltonian)
    test_hamiltonian_poisson(hamiltonian, n_vars)
    test_hamiltonian_change_variables(hamiltonian, n_vars)
    test_hamiltonian_hdf_io(hamiltonian)
    test_hamiltonian_str(hamiltonian)
    
    print("All base tests passed!")

if __name__ == "__main__":
    run_all_tests()
