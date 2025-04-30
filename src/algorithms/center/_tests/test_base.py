import pytest
import numpy as np
import symengine as se
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from typing import List, Tuple

from algorithms.center.base import (
    CenterModel, 
    hamiltonian_expr, 
    taylor_expand, 
    apply_linear_change, 
    kill_saddle_terms, 
    build_center_model
)
from algorithms.center.core import FormalSeries, Polynomial
from system.libration import LinearData


MU = 0.01
POINT = "L1"

# --- Test Fixtures ---

@pytest.fixture
def mu():
    return MU  # Small mass parameter for testing

@pytest.fixture
def libration_point():
    return POINT  # Use L1 for testing

@pytest.fixture
def order():
    return 4  # Lower order for faster tests

@pytest.fixture
def mock_linear_data():
    # Mock LinearData for testing
    mu=MU
    point=POINT
    lambda1 = 1.0
    omega1 = 2.0
    omega2 = 3.0
    C = np.eye(6)  # Identity matrix for simplicity
    Cinv = np.eye(6)
    return LinearData(mu=mu, point=point, lambda1=lambda1, omega1=omega1, omega2=omega2, C=C, Cinv=Cinv)

@pytest.fixture
def simple_formal_series():
    # Create a simple formal series with terms of degree 2, 3, and 4
    n_vars = 6
    degree2 = Polynomial('q0^2 + 0.5*p0^2', n_vars=n_vars)
    degree3 = Polynomial('q0^3', n_vars=n_vars)
    degree4 = Polynomial('q0^4 + p0^4', n_vars=n_vars)
    return FormalSeries({2: degree2, 3: degree3, 4: degree4})

@pytest.fixture
def simple_center_model(mock_linear_data, simple_formal_series, mu, libration_point):
    # Create a mock generator list
    generators = [
        Polynomial('q0*p0', n_vars=6),  # Simple generator for degree 3
        Polynomial('q0^2*p0', n_vars=6)  # Simple generator for degree 4
    ]
    
    return CenterModel(
        mu=mu,
        point=libration_point,
        series=simple_formal_series,
        generators=generators,
        linear=mock_linear_data
    )

# --- Test CenterModel ---

def test_center_model_init(simple_center_model, mu, libration_point):
    # Test initialization
    assert simple_center_model.mu == mu
    assert simple_center_model.point == libration_point
    assert len(simple_center_model.series) == 3  # Should have 3 terms
    assert len(simple_center_model.generators) == 2  # Should have 2 generators
    
def test_center_model_persistence(simple_center_model):
    # Test to_hdf and from_hdf methods
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
        tmp_path = tmp.name
    
    try:
        # Save to HDF
        simple_center_model.to_hdf(tmp_path)
        
        # Check the file exists
        assert os.path.exists(tmp_path)
        
        # Load from HDF
        loaded = CenterModel.from_hdf(tmp_path)
        
        # Check the loaded model
        assert loaded.mu == simple_center_model.mu
        assert loaded.point == simple_center_model.point
        
        # Series should have the same degrees
        assert set(loaded.series.degrees()) == set(simple_center_model.series.degrees())
        
        # Should have the same number of generators
        assert len(loaded.generators) == len(simple_center_model.generators)
        
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# --- Test hamiltonian_expr ---

def test_hamiltonian_expr():
    # Test the Hamiltonian expression in synodic coordinates
    mu = 0.01
    H, vars_symbols = hamiltonian_expr(mu)
    
    # Check that H is a symengine expression
    assert isinstance(H, se.Basic)
    
    # Check that vars_symbols has 6 symbols (X, Y, Z, PX, PY, PZ)
    assert len(vars_symbols) == 6
    
    # Check that the Hamiltonian has the expected form
    # Instead of using .has() which has issues with Pow expressions, 
    # convert to string and check for substrings
    H_str = str(H)
    
    # Check for kinetic energy terms
    assert "PX**2" in H_str
    assert "PY**2" in H_str
    assert "PZ**2" in H_str
    
    # Check for Coriolis terms
    assert "Y*PX" in H_str or "PX*Y" in H_str
    assert "X*PY" in H_str or "PY*X" in H_str
    
    # Test with a different value of mu
    mu2 = 0.1
    H2, _ = hamiltonian_expr(mu2)
    assert str(H2) != str(H)  # Should be different for different mu values

# --- Test taylor_expand ---

@patch('algorithms.center.base.get_equilibrium_point')
def test_taylor_expand(mock_get_eq_point, mu, libration_point, order):
    # Mock the equilibrium point function
    mock_get_eq_point.return_value = (0.1, 0.0, 0.0)  # Typical L1 location
    
    # Run taylor_expand
    fs = taylor_expand(mu, libration_point, order)
    
    # Check the result is a FormalSeries
    assert isinstance(fs, FormalSeries)
    
    # Check the degrees go from 0 to order
    assert set(fs.degrees()).issubset(set(range(order+1)))
    
    # Check the second-order term exists (quadratic term is always there)
    assert 2 in fs

# --- Test apply_linear_change ---

def test_apply_linear_change(simple_formal_series, mock_linear_data):
    # Test applying a linear change of variables
    result = apply_linear_change(simple_formal_series, mock_linear_data)
    
    # Should be a FormalSeries with the same degrees
    assert isinstance(result, FormalSeries)
    assert set(result.degrees()) == set(simple_formal_series.degrees())
    
    # With identity matrix, should be the same expressions
    for deg in simple_formal_series.degrees():
        assert result[deg].expr == simple_formal_series[deg].expr

# --- Test kill_saddle_terms ---

def test_kill_saddle_terms():
    # Create a polynomial with mixed terms
    poly = Polynomial('q0^2 + p0^2 + q1^2 + p1^2 + q2^2 + p2^2', n_vars=6)
    
    # Apply kill_saddle_terms
    Zk, R_perp = kill_saddle_terms(poly)
    
    # Zk should have only terms without q1 or p1
    assert not Zk.expr.has(poly.variables[0])  # q1
    assert not Zk.expr.has(poly.variables[1])  # p1
    
    # R_perp should have only terms with q1 or p1
    assert R_perp.expr.has(poly.variables[0]) or R_perp.expr.has(poly.variables[1])
    
    # Original polynomial should equal Zk + R_perp
    combined = Zk + R_perp
    assert combined.expr == poly.expr

# --- Test build_center_model ---

@patch('algorithms.center.base.compute_linear_data')
@patch('algorithms.center.base.taylor_expand')
def test_build_center_model(mock_taylor_expand, mock_compute_linear_data, 
                           mock_linear_data, simple_formal_series, mu, libration_point, order):
    # Mock the dependencies
    mock_compute_linear_data.return_value = mock_linear_data
    mock_taylor_expand.return_value = simple_formal_series
    
    # Run build_center_model
    model = build_center_model(mu, libration_point, order)
    
    # Check the result is a CenterModel
    assert isinstance(model, CenterModel)
    assert model.mu == mu
    assert model.point == libration_point
    assert model.linear == mock_linear_data
    
    # Check the series contains terms up to the requested order
    assert max(model.series.degrees()) <= order
    
    # Check that we have the expected number of generators
    # We should have one generator for each degree from 3 to order
    assert len(model.generators) == (order - 2)
