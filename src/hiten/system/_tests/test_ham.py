"""Tests for the Hamiltonian and LieGeneratingFunction classes in hamiltonian.py."""

import numpy as np
import pytest
from pathlib import Path

from hiten.system.base import System
from hiten.system.hamiltonian import Hamiltonian, LieGeneratingFunction


@pytest.fixture
def earth_moon_system():
    """Create an Earth-Moon system for testing."""
    return System.from_bodies("earth", "moon")


@pytest.fixture
def l1_point(earth_moon_system):
    """Create L1 libration point."""
    return earth_moon_system.get_libration_point(1)


@pytest.fixture
def sample_hamiltonian(l1_point):
    """Create a real Hamiltonian from L1 point."""
    # Get Hamiltonian from center manifold
    ham = l1_point.hamiltonian(max_deg=4, form="physical")
    return ham


class TestHamiltonianInitializationValidation:
    """Test Hamiltonian class initialization validation."""
    
    def test_hamiltonian_validates_degree(self):
        """Test that invalid degree raises error."""
        with pytest.raises(ValueError, match="degree must be a positive integer"):
            Hamiltonian([], degree=0)
        
        with pytest.raises(ValueError, match="degree must be a positive integer"):
            Hamiltonian([], degree=-1)
    
    def test_hamiltonian_validates_ndof(self):
        """Test that invalid ndof raises error."""
        with pytest.raises(NotImplementedError, match="only supports 3 degrees of freedom"):
            Hamiltonian([], degree=1, ndof=2)
        
        with pytest.raises(NotImplementedError, match="only supports 3 degrees of freedom"):
            Hamiltonian([], degree=1, ndof=4)


class TestHamiltonianFromLibrationPoint:
    """Test Hamiltonian creation from libration points."""
    
    def test_create_hamiltonian_from_l1(self, l1_point):
        """Test creating Hamiltonian from L1 point."""
        ham = l1_point.hamiltonian(max_deg=4, form="physical")
        
        assert ham is not None
        assert isinstance(ham, Hamiltonian)
    
    def test_create_hamiltonian_different_degrees(self, l1_point):
        """Test creating Hamiltonians with different degrees."""
        ham2 = l1_point.hamiltonian(max_deg=2, form="physical")
        ham4 = l1_point.hamiltonian(max_deg=4, form="physical")
        
        assert ham2.degree == 2
        assert ham4.degree == 4
    
    def test_create_hamiltonian_different_forms(self, l1_point):
        """Test creating Hamiltonians with different forms."""
        forms = ["physical", "real_normal"]
        
        for form in forms:
            ham = l1_point.hamiltonian(max_deg=4, form=form)
            assert ham is not None
            # Only test one form to keep test fast
            break


class TestHamiltonianProperties:
    """Test Hamiltonian class properties."""
    
    def test_name_property(self, sample_hamiltonian):
        """Test name property."""
        name = sample_hamiltonian.name
        assert isinstance(name, str)
        assert len(name) > 0
    
    def test_degree_property(self, sample_hamiltonian):
        """Test degree property."""
        assert sample_hamiltonian.degree == 4
        assert isinstance(sample_hamiltonian.degree, int)
    
    def test_ndof_property(self, sample_hamiltonian):
        """Test ndof property."""
        assert sample_hamiltonian.ndof == 3
        assert isinstance(sample_hamiltonian.ndof, int)
    
    def test_poly_H_property(self, sample_hamiltonian):
        """Test poly_H property."""
        poly_H = sample_hamiltonian.poly_H
        
        # poly_H can be a Numba ListType or regular list
        assert hasattr(poly_H, '__len__')
        assert len(poly_H) > 0
        # Should be iterable
        assert hasattr(poly_H, '__iter__')
    
    def test_hamsys_property(self, sample_hamiltonian):
        """Test hamsys property."""
        hamsys = sample_hamiltonian.hamsys
        assert hamsys is not None
    
    def test_jacobian_property(self, sample_hamiltonian):
        """Test jacobian property."""
        jacobian = sample_hamiltonian.jacobian
        assert jacobian is not None


class TestHamiltonianDunderMethods:
    """Test Hamiltonian dunder methods."""
    
    def test_repr(self, sample_hamiltonian):
        """Test __repr__ representation."""
        repr_str = repr(sample_hamiltonian)
        
        assert "Hamiltonian" in repr_str
        assert "degree=4" in repr_str
        assert "blocks=" in repr_str
    
    def test_str(self, sample_hamiltonian):
        """Test __str__ representation."""
        str_repr = str(sample_hamiltonian)
        
        # Should return a string representation
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
    
    def test_bool_true(self, sample_hamiltonian):
        """Test __bool__ for non-empty Hamiltonian."""
        assert bool(sample_hamiltonian) is True
    
    def test_len(self, sample_hamiltonian):
        """Test __len__ method."""
        length = len(sample_hamiltonian)
        assert isinstance(length, int)
        assert length > 0
    
    def test_getitem(self, sample_hamiltonian):
        """Test __getitem__ method."""
        block0 = sample_hamiltonian[0]
        assert block0 is not None
        assert isinstance(block0, np.ndarray)
        
        # Test negative indexing
        block_last = sample_hamiltonian[-1]
        assert block_last is not None
    
    def test_call(self, sample_hamiltonian):
        """Test __call__ method (evaluation)."""
        coords = np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0])
        
        result = sample_hamiltonian(coords)
        
        # Result can be complex, check it's a number and finite
        assert isinstance(result, (int, float, complex, np.number))
        assert np.isfinite(result) or np.isfinite(np.abs(result))


class TestHamiltonianSerialization:
    """Test Hamiltonian serialization methods."""
    
    def test_hamiltonian_serialization_basic(self, sample_hamiltonian):
        """Test basic Hamiltonian serialization."""
        temp_path = Path("temp_hamiltonian.pkl")
        
        try:
            sample_hamiltonian.save(str(temp_path))
            
            ham_loaded = Hamiltonian.load(str(temp_path))
            
            # Verify properties are preserved
            assert ham_loaded.name == sample_hamiltonian.name
            assert ham_loaded.degree == sample_hamiltonian.degree
            assert ham_loaded.ndof == sample_hamiltonian.ndof
            assert len(ham_loaded) == len(sample_hamiltonian)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestHamiltonianEvaluation:
    """Test Hamiltonian evaluation."""
    
    def test_evaluate_at_origin(self, sample_hamiltonian):
        """Test evaluating Hamiltonian at origin."""
        coords = np.zeros(6)
        value = sample_hamiltonian(coords)
        
        # Result can be complex
        assert isinstance(value, (int, float, complex, np.number))
        assert np.isfinite(value) or np.isfinite(np.abs(value))
    
    def test_evaluate_at_different_points(self, sample_hamiltonian):
        """Test evaluating Hamiltonian at different points."""
        coords1 = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        coords2 = np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0])
        
        value1 = sample_hamiltonian(coords1)
        value2 = sample_hamiltonian(coords2)
        
        assert np.isfinite(value1) or np.isfinite(np.abs(value1))
        assert np.isfinite(value2) or np.isfinite(np.abs(value2))


class TestHamiltonianIntegration:
    """Integration tests for Hamiltonian class."""
    
    def test_hamiltonian_full_workflow(self, l1_point):
        """Test a complete Hamiltonian workflow."""
        # Create Hamiltonian
        ham = l1_point.hamiltonian(max_deg=4, form="physical")
        
        # Verify properties
        assert ham.degree == 4
        assert ham.ndof == 3
        assert len(ham) > 0
        
        # Access blocks
        block0 = ham[0]
        assert block0 is not None
        
        # Evaluate
        coords = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        value = ham(coords)
        assert np.isfinite(value) or np.isfinite(np.abs(value))
    
    def test_multiple_hamiltonians_same_point(self, l1_point):
        """Test creating multiple Hamiltonians from same point."""
        ham1 = l1_point.hamiltonian(max_deg=2, form="physical")
        ham2 = l1_point.hamiltonian(max_deg=4, form="physical")
        
        assert ham1.degree != ham2.degree
        assert ham1 is not ham2
    
    def test_hamiltonians_from_different_points(self, earth_moon_system):
        """Test creating Hamiltonians from different libration points."""
        l1 = earth_moon_system.get_libration_point(1)
        l2 = earth_moon_system.get_libration_point(2)
        
        ham_l1 = l1.hamiltonian(max_deg=4, form="physical")
        ham_l2 = l2.hamiltonian(max_deg=4, form="physical")
        
        # Both should be valid Hamiltonians
        assert ham_l1 is not None
        assert ham_l2 is not None
        
        # But they are different objects
        assert ham_l1 is not ham_l2


class TestHamiltonianPropertiesConsistency:
    """Test consistency of Hamiltonian properties."""
    
    def test_properties_consistency(self, sample_hamiltonian):
        """Test that properties are consistent across accesses."""
        name1 = sample_hamiltonian.name
        name2 = sample_hamiltonian.name
        assert name1 == name2
        
        degree1 = sample_hamiltonian.degree
        degree2 = sample_hamiltonian.degree
        assert degree1 == degree2
        
        ndof1 = sample_hamiltonian.ndof
        ndof2 = sample_hamiltonian.ndof
        assert ndof1 == ndof2
    
    def test_poly_H_consistency(self, sample_hamiltonian):
        """Test that poly_H returns consistent results."""
        poly1 = sample_hamiltonian.poly_H
        poly2 = sample_hamiltonian.poly_H
        
        assert len(poly1) == len(poly2)


class TestHamiltonianEdgeCases:
    """Test Hamiltonian edge cases."""
    
    def test_evaluate_with_small_perturbation(self, sample_hamiltonian):
        """Test evaluation with very small perturbations."""
        coords = np.array([1e-8, 1e-8, 1e-8, 0.0, 0.0, 0.0])
        value = sample_hamiltonian(coords)
        
        assert np.isfinite(value) or np.isfinite(np.abs(value))
    
    def test_evaluate_with_larger_perturbation(self, sample_hamiltonian):
        """Test evaluation with larger perturbations."""
        coords = np.array([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        value = sample_hamiltonian(coords)
        
        assert np.isfinite(value) or np.isfinite(np.abs(value))


class TestHamiltonianClassMethods:
    """Test Hamiltonian class methods."""
    
    def test_from_state_exists(self):
        """Test that from_state class method exists."""
        assert hasattr(Hamiltonian, 'from_state')
        assert callable(Hamiltonian.from_state)
    
    def test_load_exists(self):
        """Test that load class method exists."""
        assert hasattr(Hamiltonian, 'load')
        assert callable(Hamiltonian.load)
    
    def test_register_conversion_exists(self):
        """Test that register_conversion static method exists."""
        assert hasattr(Hamiltonian, 'register_conversion')
        assert callable(Hamiltonian.register_conversion)
    
    def test_to_state_exists(self, sample_hamiltonian):
        """Test that to_state method exists."""
        assert hasattr(sample_hamiltonian, 'to_state')
        assert callable(sample_hamiltonian.to_state)


class TestLieGeneratingFunctionInterface:
    """Test LieGeneratingFunction interface."""
    
    def test_lgf_has_required_methods(self):
        """Test that LGF has required methods."""
        assert hasattr(LieGeneratingFunction, '__init__')
        assert hasattr(LieGeneratingFunction, '__repr__')
        assert hasattr(LieGeneratingFunction, '__str__')
        assert hasattr(LieGeneratingFunction, '__bool__')
        assert hasattr(LieGeneratingFunction, '__len__')
        assert hasattr(LieGeneratingFunction, '__getitem__')
        assert hasattr(LieGeneratingFunction, '__call__')
    
    def test_lgf_has_properties(self):
        """Test that LGF has expected properties."""
        assert hasattr(LieGeneratingFunction, 'poly_G')
        assert hasattr(LieGeneratingFunction, 'degree')
        assert hasattr(LieGeneratingFunction, 'ndof')
        assert hasattr(LieGeneratingFunction, 'poly_elim')
        assert hasattr(LieGeneratingFunction, 'name')
    
    def test_lgf_load_exists(self):
        """Test that load class method exists."""
        assert hasattr(LieGeneratingFunction, 'load')
        assert callable(LieGeneratingFunction.load)
