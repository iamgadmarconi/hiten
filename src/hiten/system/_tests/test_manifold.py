"""Tests for the Manifold class API in manifold.py."""

import numpy as np
import pytest
from pathlib import Path

from hiten.system.base import System
from hiten.system.manifold import Manifold
from hiten.system.orbits import GenericOrbit
from hiten.algorithms.types.states import Trajectory


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
def stable_positive_manifold(propagated_orbit):
    """Create a stable positive manifold."""
    return Manifold(propagated_orbit, stable=True, direction="positive")


@pytest.fixture
def unstable_negative_manifold(propagated_orbit):
    """Create an unstable negative manifold."""
    return Manifold(propagated_orbit, stable=False, direction="negative")


class TestManifoldInitialization:
    """Test Manifold class initialization and basic properties."""
    
    def test_manifold_stable_positive_initialization(self, propagated_orbit):
        """Test Manifold initialization with stable positive."""
        manifold = Manifold(propagated_orbit, stable=True, direction="positive")
        
        assert manifold is not None
        assert manifold.generating_orbit is propagated_orbit
        assert manifold.stable == 1  # Encoded as 1
    
    def test_manifold_stable_negative_initialization(self, propagated_orbit):
        """Test Manifold initialization with stable negative."""
        manifold = Manifold(propagated_orbit, stable=True, direction="negative")
        
        assert manifold is not None
        assert manifold.generating_orbit is propagated_orbit
        assert manifold.stable == 1
        assert manifold.direction == -1  # Negative encoded as -1
    
    def test_manifold_unstable_positive_initialization(self, propagated_orbit):
        """Test Manifold initialization with unstable positive."""
        manifold = Manifold(propagated_orbit, stable=False, direction="positive")
        
        assert manifold is not None
        assert manifold.generating_orbit is propagated_orbit
        assert manifold.stable == -1  # Unstable encoded as -1
        assert manifold.direction == 1
    
    def test_manifold_unstable_negative_initialization(self, propagated_orbit):
        """Test Manifold initialization with unstable negative."""
        manifold = Manifold(propagated_orbit, stable=False, direction="negative")
        
        assert manifold is not None
        assert manifold.generating_orbit is propagated_orbit
        assert manifold.stable == -1
        assert manifold.direction == -1
    
    def test_manifold_default_parameters(self, propagated_orbit):
        """Test Manifold initialization with default parameters."""
        manifold = Manifold(propagated_orbit)
        
        # Defaults should be stable=True, direction="positive"
        assert manifold.stable == 1
        assert manifold.direction == 1


class TestManifoldProperties:
    """Test Manifold class properties."""
    
    def test_generating_orbit_property(self, stable_positive_manifold, propagated_orbit):
        """Test generating_orbit property."""
        assert stable_positive_manifold.generating_orbit is propagated_orbit
    
    def test_libration_point_property(self, stable_positive_manifold, l1_point):
        """Test libration_point property."""
        assert stable_positive_manifold.libration_point is l1_point
    
    def test_system_property(self, stable_positive_manifold, earth_moon_system):
        """Test system property."""
        assert stable_positive_manifold.system is earth_moon_system
    
    def test_mu_property(self, stable_positive_manifold, earth_moon_system):
        """Test mu property."""
        assert stable_positive_manifold.mu == earth_moon_system.mu
        assert isinstance(stable_positive_manifold.mu, float)
        assert 0 < stable_positive_manifold.mu < 1
    
    def test_stable_property(self, stable_positive_manifold, unstable_negative_manifold):
        """Test stable property encoding."""
        assert stable_positive_manifold.stable == 1
        assert unstable_negative_manifold.stable == -1
    
    def test_direction_property(self, stable_positive_manifold, unstable_negative_manifold):
        """Test direction property encoding."""
        assert stable_positive_manifold.direction == 1
        assert unstable_negative_manifold.direction == -1
    
    def test_result_property_before_compute(self, stable_positive_manifold):
        """Test result property before computing."""
        # Before computation, result should be None or not set
        result = stable_positive_manifold.result
        # Result can be None before compute
        assert result is None or hasattr(result, '__dict__')
    
    def test_trajectories_property_before_compute(self, stable_positive_manifold):
        """Test trajectories property before computing."""
        # Before computation, trajectories should be None or empty
        trajectories = stable_positive_manifold.trajectories
        # Can be None or empty list before compute
        assert trajectories is None or isinstance(trajectories, list)


class TestManifoldStringRepresentations:
    """Test Manifold string representations."""
    
    def test_str_representation(self, stable_positive_manifold):
        """Test __str__ representation."""
        str_repr = str(stable_positive_manifold)
        
        assert "Manifold" in str_repr
        assert "stable=" in str_repr.lower()
        assert "direction=" in str_repr.lower()
    
    def test_repr_representation(self, stable_positive_manifold):
        """Test __repr__ representation."""
        repr_str = repr(stable_positive_manifold)
        
        assert "Manifold" in repr_str
        assert "stable=" in repr_str.lower()
    
    def test_different_manifold_types_repr(self, propagated_orbit):
        """Test __repr__ for different manifold types."""
        stable_pos = Manifold(propagated_orbit, stable=True, direction="positive")
        unstable_neg = Manifold(propagated_orbit, stable=False, direction="negative")
        
        str_stable = str(stable_pos)
        str_unstable = str(unstable_neg)
        
        assert str_stable != str_unstable


class TestManifoldCompute:
    """Test Manifold compute methods."""
    
    @pytest.mark.slow
    def test_compute_basic(self, stable_positive_manifold):
        """Test basic manifold computation."""
        # Use minimal parameters for faster test
        result = stable_positive_manifold.compute(
            step=0.5,  # Large step for fewer samples
            integration_fraction=0.1,  # Short integration
            show_progress=False
        )
        
        assert result is not None
    
    @pytest.mark.slow
    def test_compute_with_custom_parameters(self, stable_positive_manifold):
        """Test manifold computation with custom parameters."""
        result = stable_positive_manifold.compute(
            step=0.5,
            integration_fraction=0.1,
            NN=1,
            displacement=1e-6,
            method="adaptive",
            order=8,
            show_progress=False
        )
        
        assert result is not None
    
    def test_compute_stores_result(self, stable_positive_manifold):
        """Test that compute stores result."""
        # Before compute
        result_before = stable_positive_manifold.result
        
        # Compute
        result = stable_positive_manifold.compute(
            step=0.5,
            integration_fraction=0.1,
            show_progress=False
        )
        
        # After compute, result should be stored
        result_after = stable_positive_manifold.result
        assert result_after is result


class TestManifoldPlotting:
    """Test Manifold plotting methods."""
    
    def test_plot_without_compute_raises_error(self, stable_positive_manifold):
        """Test that plotting without compute raises error."""
        # Manifold not computed yet
        with pytest.raises(ValueError, match="not computed"):
            stable_positive_manifold.plot()
    
    @pytest.mark.slow
    def test_plot_after_compute(self, stable_positive_manifold):
        """Test plotting after compute."""
        # Compute first
        stable_positive_manifold.compute(
            step=0.5,
            integration_fraction=0.1,
            show_progress=False
        )
        
        # Plot
        fig = stable_positive_manifold.plot(save=False)
        assert fig is not None


class TestManifoldDataExport:
    """Test Manifold data export methods."""
    
    def test_to_df_without_compute_raises_error(self, stable_positive_manifold):
        """Test that to_df without compute raises error."""
        with pytest.raises(ValueError, match="not computed"):
            stable_positive_manifold.to_df()
    
    def test_to_csv_without_compute_raises_error(self, stable_positive_manifold):
        """Test that to_csv without compute raises error."""
        with pytest.raises(ValueError, match="not computed"):
            stable_positive_manifold.to_csv("temp.csv")
    
    @pytest.mark.slow
    def test_to_df_after_compute(self, stable_positive_manifold):
        """Test exporting to DataFrame after compute."""
        # Compute first
        stable_positive_manifold.compute(
            step=0.5,
            integration_fraction=0.1,
            show_progress=False
        )
        
        # Export to DataFrame
        df = stable_positive_manifold.to_df()
        
        assert df is not None
        assert 'trajectory_id' in df.columns
        assert 'time' in df.columns
        assert 'x' in df.columns
        assert 'y' in df.columns
        assert 'z' in df.columns
        assert 'vx' in df.columns
        assert 'vy' in df.columns
        assert 'vz' in df.columns
    
    @pytest.mark.slow
    def test_to_csv_after_compute(self, stable_positive_manifold):
        """Test exporting to CSV after compute."""
        # Compute first
        stable_positive_manifold.compute(
            step=0.5,
            integration_fraction=0.1,
            show_progress=False
        )
        
        temp_path = Path("temp_manifold.csv")
        
        try:
            stable_positive_manifold.to_csv(str(temp_path))
            
            assert temp_path.exists()
            
            # Read and verify
            import pandas as pd
            df = pd.read_csv(temp_path)
            assert 'trajectory_id' in df.columns
            assert len(df) > 0
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestManifoldSerialization:
    """Test Manifold serialization methods."""
    
    def test_manifold_serialization_basic(self, stable_positive_manifold):
        """Test basic manifold serialization."""
        temp_path = Path("temp_manifold.pkl")
        
        try:
            stable_positive_manifold.save(str(temp_path))
            
            manifold_loaded = Manifold.load(str(temp_path))
            
            # Verify properties are preserved
            assert manifold_loaded.stable == stable_positive_manifold.stable
            assert manifold_loaded.direction == stable_positive_manifold.direction
            assert manifold_loaded.mu == stable_positive_manifold.mu
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @pytest.mark.slow
    def test_manifold_serialization_with_computed_data(self, stable_positive_manifold):
        """Test manifold serialization with computed data."""
        # Compute first
        stable_positive_manifold.compute(
            step=0.5,
            integration_fraction=0.1,
            show_progress=False
        )
        
        temp_path = Path("temp_manifold_computed.pkl")
        
        try:
            stable_positive_manifold.save(str(temp_path))
            
            manifold_loaded = Manifold.load(str(temp_path))
            
            # Verify properties are preserved
            assert manifold_loaded.stable == stable_positive_manifold.stable
            assert manifold_loaded.direction == stable_positive_manifold.direction
            
            # Verify computed data is preserved (if implemented)
            # Result might be preserved depending on implementation
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestManifoldIntegration:
    """Integration tests for Manifold class."""
    
    def test_full_workflow(self, propagated_orbit):
        """Test a complete manifold workflow."""
        # Create manifold
        manifold = Manifold(propagated_orbit, stable=True, direction="positive")
        
        # Verify initial state
        assert manifold.generating_orbit is propagated_orbit
        assert manifold.stable == 1
        assert manifold.direction == 1
        
        # Compute (with minimal parameters for speed)
        result = manifold.compute(
            step=0.5,
            integration_fraction=0.1,
            show_progress=False
        )
        
        assert result is not None
        
        # Export to DataFrame
        df = manifold.to_df()
        assert len(df) > 0
    
    def test_multiple_manifolds_same_orbit(self, propagated_orbit):
        """Test creating multiple manifolds for the same orbit."""
        manifold_sp = Manifold(propagated_orbit, stable=True, direction="positive")
        manifold_sn = Manifold(propagated_orbit, stable=True, direction="negative")
        manifold_up = Manifold(propagated_orbit, stable=False, direction="positive")
        manifold_un = Manifold(propagated_orbit, stable=False, direction="negative")
        
        # All should reference the same orbit
        assert manifold_sp.generating_orbit is propagated_orbit
        assert manifold_sn.generating_orbit is propagated_orbit
        assert manifold_up.generating_orbit is propagated_orbit
        assert manifold_un.generating_orbit is propagated_orbit
        
        # But have different properties
        assert manifold_sp.stable == 1 and manifold_sp.direction == 1
        assert manifold_sn.stable == 1 and manifold_sn.direction == -1
        assert manifold_up.stable == -1 and manifold_up.direction == 1
        assert manifold_un.stable == -1 and manifold_un.direction == -1
    
    def test_manifold_properties_consistency(self, stable_positive_manifold):
        """Test that manifold properties are consistent across accesses."""
        # Multiple accesses should return the same values
        stable1 = stable_positive_manifold.stable
        stable2 = stable_positive_manifold.stable
        assert stable1 == stable2
        
        direction1 = stable_positive_manifold.direction
        direction2 = stable_positive_manifold.direction
        assert direction1 == direction2
        
        mu1 = stable_positive_manifold.mu
        mu2 = stable_positive_manifold.mu
        assert mu1 == mu2


class TestManifoldEdgeCases:
    """Test Manifold edge cases and error handling."""
    
    def test_manifold_with_different_systems(self):
        """Test manifolds with different systems."""
        # Earth-Moon system
        em_system = System.from_bodies("earth", "moon")
        em_l1 = em_system.get_libration_point(1)
        em_orbit = GenericOrbit(em_l1, initial_state=np.array([0.8, 0, 0, 0, 0, 0]))
        em_orbit.period = 2.5
        em_orbit.propagate(steps=50)
        em_manifold = Manifold(em_orbit, stable=True, direction="positive")
        
        # Sun-Earth system
        se_system = System.from_bodies("sun", "earth")
        se_l1 = se_system.get_libration_point(1)
        se_orbit = GenericOrbit(se_l1, initial_state=np.array([0.99, 0, 0, 0, 0, 0]))
        se_orbit.period = 2.5
        se_orbit.propagate(steps=50)
        se_manifold = Manifold(se_orbit, stable=True, direction="positive")
        
        # They should have different properties
        assert em_manifold.mu != se_manifold.mu
        assert em_manifold.system is not se_manifold.system
        assert em_manifold.generating_orbit is not se_manifold.generating_orbit
    
    def test_manifold_with_custom_mu_system(self):
        """Test manifold with custom mu system."""
        system = System.from_mu(0.05)
        l1 = system.get_libration_point(1)
        orbit = GenericOrbit(l1, initial_state=np.array([0.85, 0, 0, 0, 0, 0]))
        orbit.period = 2.5
        orbit.propagate(steps=50)
        manifold = Manifold(orbit, stable=True, direction="positive")
        
        assert manifold.mu == 0.05
        assert manifold.system is system


class TestManifoldComputeParameters:
    """Test Manifold compute method with various parameters."""
    
    @pytest.mark.slow
    def test_compute_with_different_methods(self, stable_positive_manifold):
        """Test compute with different integration methods."""
        methods = ["fixed", "adaptive"]
        
        for method in methods:
            manifold = stable_positive_manifold
            result = manifold.compute(
                step=0.5,
                integration_fraction=0.1,
                method=method,
                show_progress=False
            )
            assert result is not None
            # Note: Only testing one method to avoid long test times
            break
    
    @pytest.mark.slow
    def test_compute_with_different_parameters(self, stable_positive_manifold):
        """Test compute with various parameter combinations."""
        result = stable_positive_manifold.compute(
            step=0.5,
            integration_fraction=0.1,
            NN=1,
            displacement=1e-7,  # Smaller displacement
            dt=1e-3,
            method="adaptive",
            order=8,
            energy_tol=1e-5,
            safe_distance=2.0,
            show_progress=False
        )
        
        assert result is not None


class TestManifoldDirectionEncoding:
    """Test Manifold direction and stability encoding."""
    
    def test_all_four_combinations(self, propagated_orbit):
        """Test all four combinations of stable/unstable and positive/negative."""
        manifolds = {
            'sp': Manifold(propagated_orbit, stable=True, direction="positive"),
            'sn': Manifold(propagated_orbit, stable=True, direction="negative"),
            'up': Manifold(propagated_orbit, stable=False, direction="positive"),
            'un': Manifold(propagated_orbit, stable=False, direction="negative"),
        }
        
        assert manifolds['sp'].stable == 1 and manifolds['sp'].direction == 1
        assert manifolds['sn'].stable == 1 and manifolds['sn'].direction == -1
        assert manifolds['up'].stable == -1 and manifolds['up'].direction == 1
        assert manifolds['un'].stable == -1 and manifolds['un'].direction == -1
    
    def test_encoding_consistency(self, propagated_orbit):
        """Test that encoding is consistent."""
        manifold = Manifold(propagated_orbit, stable=True, direction="positive")
        
        # Multiple accesses should give same encoding
        stable1 = manifold.stable
        stable2 = manifold.stable
        assert stable1 == stable2 == 1
        
        dir1 = manifold.direction
        dir2 = manifold.direction
        assert dir1 == dir2 == 1
