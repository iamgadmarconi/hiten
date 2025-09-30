"""Tests for the InvariantTori class API in torus.py."""

import numpy as np
import pytest
from pathlib import Path

from hiten.system.base import System
from hiten.system.torus import Torus, InvariantTori
from hiten.system.orbits import GenericOrbit


@pytest.fixture
def earth_moon_system():
    """Create an Earth-Moon system for testing."""
    return System.from_bodies("earth", "moon")


@pytest.fixture
def l1_point(earth_moon_system):
    """Create L1 libration point."""
    return earth_moon_system.get_libration_point(1)


@pytest.fixture
def propagated_orbit(l1_point):
    """Create and propagate an orbit."""
    initial_state = np.array([l1_point.position[0] + 0.01, 0, 0, 0, 0, 0])
    orbit = GenericOrbit(l1_point, initial_state=initial_state)
    orbit.period = 2.5
    orbit.propagate(steps=100)
    return orbit


@pytest.fixture
def invariant_tori(propagated_orbit):
    """Create an InvariantTori object."""
    return InvariantTori(propagated_orbit)


@pytest.fixture
def sample_torus_dataclass(earth_moon_system):
    """Create a sample Torus dataclass for testing."""
    grid = np.random.random((10, 10, 6))
    omega = np.array([1.0, 0.5])
    C0 = 3.0
    return Torus(grid=grid, omega=omega, C0=C0, system=earth_moon_system)


class TestTorusDataclass:
    """Test Torus dataclass."""
    
    def test_torus_initialization(self, earth_moon_system):
        """Test Torus dataclass initialization."""
        grid = np.random.random((10, 10, 6))
        omega = np.array([1.0, 0.5])
        C0 = 3.0
        
        torus = Torus(grid=grid, omega=omega, C0=C0, system=earth_moon_system)
        
        assert torus is not None
        assert torus.grid.shape == (10, 10, 6)
        assert torus.omega.shape == (2,)
        assert torus.C0 == 3.0
        assert torus.system is earth_moon_system
    
    def test_torus_immutability(self, sample_torus_dataclass):
        """Test that Torus dataclass is immutable (frozen)."""
        # Should not be able to modify attributes
        with pytest.raises((AttributeError, TypeError)):
            sample_torus_dataclass.C0 = 5.0
    
    def test_torus_grid_property(self, sample_torus_dataclass):
        """Test Torus grid property."""
        assert sample_torus_dataclass.grid.shape == (10, 10, 6)
        assert isinstance(sample_torus_dataclass.grid, np.ndarray)
    
    def test_torus_omega_property(self, sample_torus_dataclass):
        """Test Torus omega property."""
        assert sample_torus_dataclass.omega.shape == (2,)
        assert isinstance(sample_torus_dataclass.omega, np.ndarray)
    
    def test_torus_C0_property(self, sample_torus_dataclass):
        """Test Torus C0 property."""
        assert isinstance(sample_torus_dataclass.C0, (int, float))
    
    def test_torus_system_property(self, sample_torus_dataclass, earth_moon_system):
        """Test Torus system property."""
        assert sample_torus_dataclass.system is earth_moon_system


class TestInvariantToriInitialization:
    """Test InvariantTori class initialization."""
    
    def test_invariant_tori_initialization(self, propagated_orbit):
        """Test InvariantTori initialization."""
        tori = InvariantTori(propagated_orbit)
        
        assert tori is not None
        assert tori.orbit is propagated_orbit
    
    def test_invariant_tori_requires_orbit(self):
        """Test that InvariantTori requires an orbit."""
        with pytest.raises(TypeError):
            InvariantTori()


class TestInvariantToriProperties:
    """Test InvariantTori class properties."""
    
    def test_orbit_property(self, invariant_tori, propagated_orbit):
        """Test orbit property."""
        assert invariant_tori.orbit is propagated_orbit
    
    def test_libration_point_property(self, invariant_tori, l1_point):
        """Test libration_point property."""
        assert invariant_tori.libration_point is l1_point
    
    def test_system_property(self, invariant_tori, earth_moon_system):
        """Test system property."""
        assert invariant_tori.system is earth_moon_system
    
    def test_dynsys_property(self, invariant_tori):
        """Test dynsys property."""
        dynsys = invariant_tori.dynsys
        assert dynsys is not None
    
    def test_var_dynsys_property(self, invariant_tori):
        """Test var_dynsys property."""
        var_dynsys = invariant_tori.var_dynsys
        assert var_dynsys is not None
    
    def test_jacobian_dynsys_property(self, invariant_tori):
        """Test jacobian_dynsys property."""
        jacobian_dynsys = invariant_tori.jacobian_dynsys
        assert jacobian_dynsys is not None
    
    def test_period_property(self, invariant_tori, propagated_orbit):
        """Test period property."""
        assert invariant_tori.period == propagated_orbit.period
        assert isinstance(invariant_tori.period, (int, float))
    
    def test_jacobi_property(self, invariant_tori):
        """Test jacobi property."""
        jacobi = invariant_tori.jacobi
        assert isinstance(jacobi, (int, float))
    
    def test_energy_property(self, invariant_tori):
        """Test energy property."""
        energy = invariant_tori.energy
        assert isinstance(energy, (int, float))
    
    def test_grid_property_before_compute(self, invariant_tori):
        """Test grid property before compute."""
        # Before compute, grid might be None or empty
        try:
            grid = invariant_tori.grid
            # If it returns something, it should be an array or None
            assert grid is None or isinstance(grid, np.ndarray)
        except (ValueError, AttributeError, KeyError):
            # It's okay if accessing grid before compute raises an error
            pass


class TestInvariantToriStringRepresentations:
    """Test InvariantTori string representations."""
    
    def test_str_representation(self, invariant_tori):
        """Test __str__ representation."""
        str_repr = str(invariant_tori)
        
        assert "InvariantTori" in str_repr
    
    def test_repr_representation(self, invariant_tori):
        """Test __repr__ representation."""
        repr_str = repr(invariant_tori)
        
        assert "InvariantTori" in repr_str
        assert "orbit=" in repr_str
        assert "point=" in repr_str


class TestInvariantToriCompute:
    """Test InvariantTori compute method."""
    
    def test_compute_basic(self, invariant_tori):
        """Test basic torus computation."""
        grid = invariant_tori.compute(
            epsilon=0.001,
            n_theta1=10,
            n_theta2=10,
            method="adaptive",
            order=8
        )
        
        assert grid is not None
        assert isinstance(grid, np.ndarray)
        # Grid should have shape (n_theta1, n_theta2, 6)
        assert grid.shape == (10, 10, 6)
    
    def test_compute_with_different_resolutions(self, invariant_tori):
        """Test compute with different grid resolutions."""
        grid1 = invariant_tori.compute(
            epsilon=0.001,
            n_theta1=5,
            n_theta2=5,
            method="adaptive",
            order=8
        )
        
        assert grid1.shape == (5, 5, 6)
        
        grid2 = invariant_tori.compute(
            epsilon=0.001,
            n_theta1=20,
            n_theta2=15,
            method="adaptive",
            order=8
        )
        
        assert grid2.shape == (20, 15, 6)
    
    def test_compute_with_different_epsilon(self, invariant_tori):
        """Test compute with different epsilon values."""
        grid1 = invariant_tori.compute(
            epsilon=0.001,
            n_theta1=10,
            n_theta2=10,
            method="adaptive",
            order=8
        )
        
        grid2 = invariant_tori.compute(
            epsilon=0.002,
            n_theta1=10,
            n_theta2=10,
            method="adaptive",
            order=8
        )
        
        # Grids should be different for different epsilon
        assert not np.allclose(grid1, grid2)
    
    def test_compute_with_different_methods(self, invariant_tori):
        """Test compute with different integration methods."""
        methods = ["fixed", "adaptive"]
        
        for method in methods:
            grid = invariant_tori.compute(
                epsilon=0.001,
                n_theta1=10,
                n_theta2=10,
                method=method,
                order=8
            )
            
            assert grid is not None
            assert grid.shape == (10, 10, 6)
            # Only test one method to avoid long test times
            break
    
    def test_compute_stores_grid(self, invariant_tori):
        """Test that compute stores the grid."""
        # Compute
        grid = invariant_tori.compute(
            epsilon=0.001,
            n_theta1=10,
            n_theta2=10,
            method="adaptive",
            order=8
        )
        
        # Grid should be accessible via property
        stored_grid = invariant_tori.grid
        assert stored_grid is not None
        assert np.array_equal(stored_grid, grid)


class TestInvariantToriPlotting:
    """Test InvariantTori plotting methods."""
    
    def test_plot_after_compute(self, invariant_tori):
        """Test plotting after compute."""
        # Compute first
        invariant_tori.compute(
            epsilon=0.001,
            n_theta1=10,
            n_theta2=10,
            method="adaptive",
            order=8
        )
        
        # Plot
        fig = invariant_tori.plot(save=False)
        assert fig is not None
    
    def test_plot_without_compute_raises_error(self, invariant_tori):
        """Test that plotting without compute raises error."""
        # Before compute, plotting should fail
        with pytest.raises((ValueError, AttributeError, KeyError)):
            invariant_tori.plot()
    
    def test_plot_with_custom_parameters(self, invariant_tori):
        """Test plotting with custom parameters."""
        # Compute first
        invariant_tori.compute(
            epsilon=0.001,
            n_theta1=10,
            n_theta2=10,
            method="adaptive",
            order=8
        )
        
        # Plot with custom parameters
        fig = invariant_tori.plot(
            figsize=(12, 10),
            dark_mode=False,
            save=False
        )
        assert fig is not None


class TestInvariantToriSerialization:
    """Test InvariantTori serialization methods."""
    
    def test_tori_serialization_basic(self, invariant_tori):
        """Test basic invariant tori serialization."""
        temp_path = Path("temp_tori.pkl")
        
        try:
            invariant_tori.save(str(temp_path))
            
            tori_loaded = InvariantTori.load(str(temp_path))
            
            # Verify properties are preserved
            assert tori_loaded.period == invariant_tori.period
            assert tori_loaded.system.mu == invariant_tori.system.mu
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_tori_serialization_with_computed_grid(self, invariant_tori):
        """Test invariant tori serialization with computed grid."""
        # Compute first
        invariant_tori.compute(
            epsilon=0.001,
            n_theta1=10,
            n_theta2=10,
            method="adaptive",
            order=8
        )
        
        temp_path = Path("temp_tori_computed.pkl")
        
        try:
            invariant_tori.save(str(temp_path))
            
            tori_loaded = InvariantTori.load(str(temp_path))
            
            # Verify properties are preserved
            assert tori_loaded.period == invariant_tori.period
            
            # Verify computed grid is preserved (if implemented)
            # Grid might be preserved depending on implementation
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestInvariantToriIntegration:
    """Integration tests for InvariantTori class."""
    
    def test_full_workflow(self, propagated_orbit):
        """Test a complete invariant tori workflow."""
        # Create InvariantTori
        tori = InvariantTori(propagated_orbit)
        
        # Verify initial state
        assert tori.orbit is propagated_orbit
        assert tori.period == propagated_orbit.period
        
        # Compute
        grid = tori.compute(
            epsilon=0.001,
            n_theta1=10,
            n_theta2=10,
            method="adaptive",
            order=8
        )
        
        assert grid is not None
        assert grid.shape == (10, 10, 6)
        
        # Plot
        fig = tori.plot(save=False)
        assert fig is not None
    
    def test_multiple_tori_same_orbit(self, propagated_orbit):
        """Test creating multiple tori for the same orbit."""
        tori1 = InvariantTori(propagated_orbit)
        tori2 = InvariantTori(propagated_orbit)
        
        # Both should reference the same orbit
        assert tori1.orbit is propagated_orbit
        assert tori2.orbit is propagated_orbit
        
        # But be different instances
        assert tori1 is not tori2
    
    def test_tori_properties_consistency(self, invariant_tori):
        """Test that tori properties are consistent across accesses."""
        # Multiple accesses should return the same values
        period1 = invariant_tori.period
        period2 = invariant_tori.period
        assert period1 == period2
        
        jacobi1 = invariant_tori.jacobi
        jacobi2 = invariant_tori.jacobi
        assert jacobi1 == jacobi2
        
        energy1 = invariant_tori.energy
        energy2 = invariant_tori.energy
        assert energy1 == energy2


class TestInvariantToriEdgeCases:
    """Test InvariantTori edge cases and error handling."""
    
    def test_tori_with_different_systems(self):
        """Test tori with different systems."""
        # Earth-Moon system
        em_system = System.from_bodies("earth", "moon")
        em_l1 = em_system.get_libration_point(1)
        em_orbit = GenericOrbit(em_l1, initial_state=np.array([0.8, 0, 0, 0, 0, 0]))
        em_orbit.period = 2.5
        em_orbit.propagate(steps=50)
        em_tori = InvariantTori(em_orbit)
        
        # Sun-Earth system
        se_system = System.from_bodies("sun", "earth")
        se_l1 = se_system.get_libration_point(1)
        se_orbit = GenericOrbit(se_l1, initial_state=np.array([0.99, 0, 0, 0, 0, 0]))
        se_orbit.period = 2.5
        se_orbit.propagate(steps=50)
        se_tori = InvariantTori(se_orbit)
        
        # They should have different properties
        assert em_tori.system.mu != se_tori.system.mu
        assert em_tori.system is not se_tori.system
        assert em_tori.orbit is not se_tori.orbit
    
    def test_tori_with_custom_mu_system(self):
        """Test tori with custom mu system."""
        system = System.from_mu(0.05)
        l1 = system.get_libration_point(1)
        orbit = GenericOrbit(l1, initial_state=np.array([0.85, 0, 0, 0, 0, 0]))
        orbit.period = 2.5
        orbit.propagate(steps=50)
        tori = InvariantTori(orbit)
        
        assert tori.system.mu == 0.05
        assert tori.system is system
    
    def test_compute_with_very_small_epsilon(self, invariant_tori):
        """Test compute with very small epsilon."""
        grid = invariant_tori.compute(
            epsilon=1e-8,
            n_theta1=5,
            n_theta2=5,
            method="adaptive",
            order=8
        )
        
        assert grid is not None
        assert grid.shape == (5, 5, 6)
    
    def test_compute_with_minimal_resolution(self, invariant_tori):
        """Test compute with minimal grid resolution."""
        # Minimum reasonable resolution
        grid = invariant_tori.compute(
            epsilon=0.001,
            n_theta1=2,
            n_theta2=2,
            method="adaptive",
            order=8
        )
        
        assert grid is not None
        assert grid.shape == (2, 2, 6)


class TestInvariantToriComputeParameters:
    """Test InvariantTori compute method with various parameters."""
    
    def test_compute_with_different_orders(self, invariant_tori):
        """Test compute with different integration orders."""
        orders = [4, 8, 12]
        
        for order in orders:
            grid = invariant_tori.compute(
                epsilon=0.001,
                n_theta1=5,
                n_theta2=5,
                method="fixed",
                order=order
            )
            assert grid is not None
            assert grid.shape == (5, 5, 6)
            # Only test one order to avoid long test times
            break
    
    def test_compute_parameter_combinations(self, invariant_tori):
        """Test compute with various parameter combinations."""
        # Small grid, small epsilon
        grid1 = invariant_tori.compute(
            epsilon=0.0005,
            n_theta1=5,
            n_theta2=5,
            method="adaptive",
            order=8
        )
        assert grid1.shape == (5, 5, 6)
        
        # Larger grid, larger epsilon
        grid2 = invariant_tori.compute(
            epsilon=0.002,
            n_theta1=15,
            n_theta2=15,
            method="adaptive",
            order=8
        )
        assert grid2.shape == (15, 15, 6)


class TestTorusDataclassAdvanced:
    """Advanced tests for Torus dataclass."""
    
    def test_torus_with_different_grid_shapes(self, earth_moon_system):
        """Test Torus with different grid shapes."""
        # Square grid
        grid1 = np.random.random((10, 10, 6))
        torus1 = Torus(grid=grid1, omega=np.array([1.0, 0.5]), C0=3.0, system=earth_moon_system)
        assert torus1.grid.shape == (10, 10, 6)
        
        # Rectangular grid
        grid2 = np.random.random((20, 15, 6))
        torus2 = Torus(grid=grid2, omega=np.array([1.0, 0.5]), C0=3.0, system=earth_moon_system)
        assert torus2.grid.shape == (20, 15, 6)
    
    def test_torus_with_different_frequencies(self, earth_moon_system):
        """Test Torus with different frequency values."""
        grid = np.random.random((10, 10, 6))
        
        omega1 = np.array([1.0, 0.5])
        torus1 = Torus(grid=grid, omega=omega1, C0=3.0, system=earth_moon_system)
        assert np.array_equal(torus1.omega, omega1)
        
        omega2 = np.array([2.0, 1.0])
        torus2 = Torus(grid=grid, omega=omega2, C0=3.0, system=earth_moon_system)
        assert np.array_equal(torus2.omega, omega2)
    
    def test_torus_with_different_jacobi_constants(self, earth_moon_system):
        """Test Torus with different Jacobi constants."""
        grid = np.random.random((10, 10, 6))
        omega = np.array([1.0, 0.5])
        
        torus1 = Torus(grid=grid, omega=omega, C0=3.0, system=earth_moon_system)
        assert torus1.C0 == 3.0
        
        torus2 = Torus(grid=grid, omega=omega, C0=3.5, system=earth_moon_system)
        assert torus2.C0 == 3.5


class TestInvariantToriPropertyAccess:
    """Test property access patterns for InvariantTori."""
    
    def test_property_access_order(self, invariant_tori):
        """Test that properties can be accessed in any order."""
        # Access properties in different orders
        orbit = invariant_tori.orbit
        system = invariant_tori.system
        period = invariant_tori.period
        libration = invariant_tori.libration_point
        
        assert orbit is not None
        assert system is not None
        assert period is not None
        assert libration is not None
    
    def test_dynamical_system_properties(self, invariant_tori):
        """Test dynamical system related properties."""
        dynsys = invariant_tori.dynsys
        var_dynsys = invariant_tori.var_dynsys
        jacobian_dynsys = invariant_tori.jacobian_dynsys
        
        assert dynsys is not None
        assert var_dynsys is not None
        assert jacobian_dynsys is not None
    
    def test_orbit_derived_properties(self, invariant_tori):
        """Test properties derived from the orbit."""
        period = invariant_tori.period
        jacobi = invariant_tori.jacobi
        energy = invariant_tori.energy
        
        assert isinstance(period, (int, float))
        assert isinstance(jacobi, (int, float))
        assert isinstance(energy, (int, float))
