"""Tests for the PeriodicOrbit classes API in orbits/."""

import numpy as np
import pytest
from pathlib import Path

from hiten.system.base import System
from hiten.system.orbits import (
    PeriodicOrbit,
    GenericOrbit,
    HaloOrbit,
    LyapunovOrbit,
    VerticalOrbit,
)
from hiten.algorithms.types.states import Trajectory
from hiten.algorithms.corrector.config import OrbitCorrectionConfig


@pytest.fixture
def earth_moon_system():
    """Create an Earth-Moon system for testing."""
    return System.from_bodies("earth", "moon")


@pytest.fixture
def l1_point(earth_moon_system):
    """Create L1 libration point."""
    return earth_moon_system.get_libration_point(1)


@pytest.fixture
def l2_point(earth_moon_system):
    """Create L2 libration point."""
    return earth_moon_system.get_libration_point(2)


@pytest.fixture
def sample_initial_state(l1_point):
    """Create a sample initial state near L1."""
    pos = l1_point.position
    return np.array([pos[0] + 0.01, 0.0, 0.0, 0.0, 0.0, 0.0])


class TestPeriodicOrbitInitialization:
    """Test PeriodicOrbit class initialization and basic properties."""
    
    def test_generic_orbit_initialization_with_state(self, l1_point, sample_initial_state):
        """Test GenericOrbit initialization with initial state."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        assert orbit is not None
        assert orbit.libration_point is l1_point
        assert np.allclose(orbit.initial_state, sample_initial_state)
        assert orbit.family == "generic"
    
    def test_generic_orbit_requires_initial_state(self, l1_point):
        """Test that GenericOrbit requires an initial state."""
        # GenericOrbit should raise ValueError without initial state
        with pytest.raises(ValueError, match="No initial state provided"):
            orbit = GenericOrbit(l1_point)
    
    def test_halo_orbit_initialization(self, l1_point):
        """Test HaloOrbit initialization."""
        orbit = HaloOrbit(l1_point, amplitude_z=0.01, zenith="northern")
        
        assert orbit is not None
        assert orbit.libration_point is l1_point
        assert orbit.family == "halo"
    
    def test_lyapunov_orbit_initialization(self, l1_point):
        """Test LyapunovOrbit initialization."""
        orbit = LyapunovOrbit(l1_point, amplitude_x=0.01)
        
        assert orbit is not None
        assert orbit.libration_point is l1_point
        assert orbit.family == "lyapunov"
    
    def test_vertical_orbit_initialization(self, l1_point, sample_initial_state):
        """Test VerticalOrbit initialization."""
        orbit = VerticalOrbit(l1_point, initial_state=sample_initial_state)
        
        assert orbit is not None
        assert orbit.libration_point is l1_point
        assert orbit.family == "vertical"


class TestPeriodicOrbitProperties:
    """Test PeriodicOrbit class properties."""
    
    @pytest.fixture
    def generic_orbit(self, l1_point, sample_initial_state):
        """Create a generic orbit for testing."""
        return GenericOrbit(l1_point, initial_state=sample_initial_state)
    
    def test_family_property(self, generic_orbit):
        """Test family property."""
        assert generic_orbit.family == "generic"
        assert isinstance(generic_orbit.family, str)
    
    def test_initial_state_property(self, generic_orbit, sample_initial_state):
        """Test initial_state property."""
        initial_state = generic_orbit.initial_state
        assert isinstance(initial_state, np.ndarray)
        assert initial_state.shape == (6,)
        assert np.allclose(initial_state, sample_initial_state)
    
    def test_libration_point_property(self, generic_orbit, l1_point):
        """Test libration_point property."""
        assert generic_orbit.libration_point is l1_point
    
    def test_system_property(self, generic_orbit, earth_moon_system):
        """Test system property."""
        assert generic_orbit.system is earth_moon_system
    
    def test_mu_property(self, generic_orbit, earth_moon_system):
        """Test mu property."""
        assert generic_orbit.mu == earth_moon_system.mu
        assert isinstance(generic_orbit.mu, float)
        assert 0 < generic_orbit.mu < 1
    
    def test_period_property(self, generic_orbit):
        """Test period property."""
        # Generic orbit should have default period
        assert generic_orbit.period is not None
        assert generic_orbit.period == np.pi
        
        # Test setting period
        generic_orbit.period = 2.5
        assert generic_orbit.period == 2.5
        
        # Test clearing period
        generic_orbit.period = None
        assert generic_orbit.period is None
    
    def test_energy_property(self, generic_orbit):
        """Test energy property."""
        energy = generic_orbit.energy
        assert isinstance(energy, float)
    
    def test_jacobi_property(self, generic_orbit):
        """Test jacobi constant property."""
        jacobi = generic_orbit.jacobi
        assert isinstance(jacobi, float)
        
        # CJ = -2E
        assert np.isclose(jacobi, -2 * generic_orbit.energy)
    
    def test_trajectory_property_before_propagation(self, generic_orbit):
        """Test trajectory property before propagation."""
        # Accessing trajectory before propagation raises ValueError
        with pytest.raises(ValueError, match="Trajectory not computed"):
            _ = generic_orbit.trajectory
    
    def test_amplitude_property_halo_orbit(self, l1_point):
        """Test amplitude property for HaloOrbit."""
        # HaloOrbit has amplitude property
        orbit = HaloOrbit(l1_point, amplitude_z=0.01, zenith="northern")
        
        # Should have an amplitude
        amplitude = orbit.amplitude
        
        # Test setting amplitude
        if amplitude is not None:
            orbit.amplitude = 0.05
            assert orbit.amplitude == 0.05


class TestPeriodicOrbitPropagation:
    """Test PeriodicOrbit propagation methods."""
    
    @pytest.fixture
    def generic_orbit_with_period(self, l1_point, sample_initial_state):
        """Create a generic orbit with a set period."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        orbit.period = 2.5
        return orbit
    
    def test_propagate_basic(self, generic_orbit_with_period):
        """Test basic propagation."""
        trajectory = generic_orbit_with_period.propagate(steps=100)
        
        assert trajectory is not None
        assert isinstance(trajectory, Trajectory)
        assert trajectory.n_samples == 100
        assert len(trajectory.times) == 100
        assert len(trajectory.states) == 100
    
    def test_propagate_with_different_steps(self, generic_orbit_with_period):
        """Test propagation with different step counts."""
        for steps in [50, 100, 200]:
            trajectory = generic_orbit_with_period.propagate(steps=steps)
            assert trajectory.n_samples == steps
    
    def test_propagate_with_different_methods(self, generic_orbit_with_period):
        """Test propagation with different integration methods."""
        methods = ["fixed", "adaptive"]
        
        for method in methods:
            trajectory = generic_orbit_with_period.propagate(steps=100, method=method)
            assert trajectory is not None
            assert trajectory.n_samples == 100
    
    def test_propagate_stores_trajectory(self, generic_orbit_with_period):
        """Test that propagation stores the trajectory."""
        # Before propagation, accessing trajectory raises ValueError
        with pytest.raises(ValueError, match="Trajectory not computed"):
            _ = generic_orbit_with_period.trajectory
        
        generic_orbit_with_period.propagate(steps=100)
        
        # After propagation, trajectory should be accessible
        assert generic_orbit_with_period.trajectory is not None
        assert isinstance(generic_orbit_with_period.trajectory, Trajectory)
    
    def test_monodromy_property(self, generic_orbit_with_period):
        """Test monodromy matrix property."""
        # Propagate first
        generic_orbit_with_period.propagate(steps=100)
        
        # Get monodromy matrix
        monodromy = generic_orbit_with_period.monodromy
        
        assert isinstance(monodromy, np.ndarray)
        assert monodromy.shape == (6, 6)


class TestHaloOrbitSpecific:
    """Test HaloOrbit specific features."""
    
    def test_halo_orbit_northern_hemisphere(self, l1_point):
        """Test HaloOrbit with northern hemisphere."""
        orbit = HaloOrbit(l1_point, amplitude_z=0.01, zenith="northern")
        
        assert orbit.zenith == "northern"
        assert orbit.family == "halo"
    
    def test_halo_orbit_southern_hemisphere(self, l1_point):
        """Test HaloOrbit with southern hemisphere."""
        orbit = HaloOrbit(l1_point, amplitude_z=0.01, zenith="southern")
        
        assert orbit.zenith == "southern"
        assert orbit.family == "halo"
    
    def test_halo_orbit_with_initial_state(self, l1_point, sample_initial_state):
        """Test HaloOrbit with user-provided initial state."""
        orbit = HaloOrbit(l1_point, initial_state=sample_initial_state)
        
        assert orbit is not None
        assert np.allclose(orbit.initial_state, sample_initial_state)


class TestLyapunovOrbitSpecific:
    """Test LyapunovOrbit specific features."""
    
    def test_lyapunov_orbit_with_amplitude(self, l1_point):
        """Test LyapunovOrbit with amplitude."""
        orbit = LyapunovOrbit(l1_point, amplitude_x=0.01)
        
        assert orbit is not None
        assert orbit.family == "lyapunov"
    
    def test_lyapunov_orbit_with_initial_state(self, l1_point, sample_initial_state):
        """Test LyapunovOrbit with user-provided initial state."""
        orbit = LyapunovOrbit(l1_point, initial_state=sample_initial_state)
        
        assert orbit is not None
        assert np.allclose(orbit.initial_state, sample_initial_state)


class TestVerticalOrbitSpecific:
    """Test VerticalOrbit specific features."""
    
    def test_vertical_orbit_initialization(self, l1_point, sample_initial_state):
        """Test VerticalOrbit initialization."""
        orbit = VerticalOrbit(l1_point, initial_state=sample_initial_state)
        
        assert orbit is not None
        assert orbit.family == "vertical"


class TestPeriodicOrbitStringRepresentations:
    """Test PeriodicOrbit string representations."""
    
    def test_str_representation(self, l1_point, sample_initial_state):
        """Test __str__ representation."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        str_repr = str(orbit)
        
        assert "generic" in str_repr.lower()
        assert "orbit" in str_repr.lower()
    
    def test_repr_representation(self, l1_point, sample_initial_state):
        """Test __repr__ representation."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        repr_str = repr(orbit)
        
        assert "GenericOrbit" in repr_str
        assert "family=" in repr_str
    
    def test_different_orbit_families_repr(self, l1_point):
        """Test __repr__ for different orbit families."""
        halo = HaloOrbit(l1_point, amplitude_z=0.01, zenith="northern")
        lyap = LyapunovOrbit(l1_point, amplitude_x=0.01)
        
        assert "HaloOrbit" in repr(halo)
        assert "LyapunovOrbit" in repr(lyap)


class TestPeriodicOrbitCorrection:
    """Test PeriodicOrbit correction methods."""
    
    def test_correction_config_not_set_raises_error(self, l1_point, sample_initial_state):
        """Test that accessing correction_config on GenericOrbit raises error when not set."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        # GenericOrbit doesn't have a default correction_config
        with pytest.raises(NotImplementedError, match="Differential correction is not defined"):
            _ = orbit.correction_config


class TestPeriodicOrbitManifold:
    """Test PeriodicOrbit manifold generation."""
    
    @pytest.fixture
    def propagated_orbit(self, l1_point, sample_initial_state):
        """Create and propagate an orbit."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        orbit.period = 2.5
        orbit.propagate(steps=100)
        return orbit
    
    def test_manifold_stable_positive(self, propagated_orbit):
        """Test creating stable positive manifold."""
        manifold = propagated_orbit.manifold(stable=True, direction="positive")
        
        assert manifold is not None
        from hiten.system.manifold import Manifold
        assert isinstance(manifold, Manifold)
    
    def test_manifold_stable_negative(self, propagated_orbit):
        """Test creating stable negative manifold."""
        manifold = propagated_orbit.manifold(stable=True, direction="negative")
        
        assert manifold is not None
    
    def test_manifold_unstable_positive(self, propagated_orbit):
        """Test creating unstable positive manifold."""
        manifold = propagated_orbit.manifold(stable=False, direction="positive")
        
        assert manifold is not None
    
    def test_manifold_unstable_negative(self, propagated_orbit):
        """Test creating unstable negative manifold."""
        manifold = propagated_orbit.manifold(stable=False, direction="negative")
        
        assert manifold is not None


class TestPeriodicOrbitPlotting:
    """Test PeriodicOrbit plotting methods."""
    
    @pytest.fixture
    def propagated_orbit(self, l1_point, sample_initial_state):
        """Create and propagate an orbit."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        orbit.period = 2.5
        orbit.propagate(steps=100)
        return orbit
    
    def test_plot_without_trajectory_raises_error(self, l1_point, sample_initial_state):
        """Test that plotting without trajectory raises error."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        with pytest.raises(RuntimeError, match="No trajectory to plot"):
            orbit.plot()
    
    def test_plot_rotating_frame(self, propagated_orbit):
        """Test plotting in rotating frame."""
        fig = propagated_orbit.plot(frame="rotating", save=False)
        assert fig is not None
    
    def test_plot_inertial_frame(self, propagated_orbit):
        """Test plotting in inertial frame."""
        fig = propagated_orbit.plot(frame="inertial", save=False)
        assert fig is not None
    
    def test_plot_invalid_frame_raises_error(self, propagated_orbit):
        """Test that invalid frame raises error."""
        with pytest.raises(ValueError, match="Invalid frame"):
            propagated_orbit.plot(frame="invalid")
    
    def test_animate_without_trajectory(self, l1_point, sample_initial_state):
        """Test animation without trajectory returns None."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        result = orbit.animate()
        assert result == (None, None)


class TestPeriodicOrbitDataExport:
    """Test PeriodicOrbit data export methods."""
    
    @pytest.fixture
    def propagated_orbit(self, l1_point, sample_initial_state):
        """Create and propagate an orbit."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        orbit.period = 2.5
        orbit.propagate(steps=100)
        return orbit
    
    def test_to_df(self, propagated_orbit):
        """Test exporting to DataFrame."""
        df = propagated_orbit.to_df()
        
        assert df is not None
        assert len(df) == 100
        assert "time" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns
        assert "vx" in df.columns
        assert "vy" in df.columns
        assert "vz" in df.columns
    
    def test_to_df_without_trajectory_raises_error(self, l1_point, sample_initial_state):
        """Test that to_df without trajectory raises error."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        with pytest.raises(ValueError, match="Trajectory not computed"):
            orbit.to_df()
    
    def test_to_csv(self, propagated_orbit):
        """Test exporting to CSV."""
        temp_path = Path("temp_orbit.csv")
        
        try:
            propagated_orbit.to_csv(str(temp_path))
            
            assert temp_path.exists()
            
            # Read and verify
            import pandas as pd
            df = pd.read_csv(temp_path)
            assert len(df) == 100
            assert "time" in df.columns
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_to_csv_without_trajectory_raises_error(self, l1_point, sample_initial_state):
        """Test that to_csv without trajectory raises error."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        with pytest.raises(ValueError, match="Trajectory not computed"):
            orbit.to_csv("temp.csv")


class TestPeriodicOrbitSerialization:
    """Test PeriodicOrbit serialization methods."""
    
    def test_orbit_serialization_roundtrip(self, l1_point, sample_initial_state):
        """Test orbit serialization and deserialization."""
        orbit_original = GenericOrbit(l1_point, initial_state=sample_initial_state)
        orbit_original.period = 2.5
        
        temp_path = Path("temp_orbit.pkl")
        
        try:
            orbit_original.save(str(temp_path))
            
            orbit_loaded = GenericOrbit.load(str(temp_path))
            
            # Verify properties are preserved
            assert np.allclose(orbit_loaded.initial_state, orbit_original.initial_state)
            assert orbit_loaded.period == orbit_original.period
            assert orbit_loaded.mu == orbit_original.mu
            assert orbit_loaded.family == orbit_original.family
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_halo_orbit_serialization(self, l1_point):
        """Test HaloOrbit serialization."""
        orbit_original = HaloOrbit(l1_point, amplitude_z=0.01, zenith="northern")
        
        temp_path = Path("temp_halo.pkl")
        
        try:
            orbit_original.save(str(temp_path))
            
            orbit_loaded = HaloOrbit.load(str(temp_path))
            
            assert orbit_loaded.family == "halo"
            assert orbit_loaded.zenith == orbit_original.zenith
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestPeriodicOrbitStability:
    """Test PeriodicOrbit stability analysis."""
    
    @pytest.fixture
    def propagated_orbit(self, l1_point, sample_initial_state):
        """Create and propagate an orbit."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        orbit.period = 2.5
        orbit.propagate(steps=100)
        return orbit
    
    def test_stability_indices_property(self, propagated_orbit):
        """Test stability_indices property."""
        stability_indices = propagated_orbit.stability_indices
        
        # May be None or a numpy array depending on computation
        if stability_indices is not None:
            assert isinstance(stability_indices, (tuple, np.ndarray))
    
    def test_eigenvalues_property(self, propagated_orbit):
        """Test eigenvalues property."""
        eigenvalues = propagated_orbit.eigenvalues
        
        # May be None or a tuple/array depending on computation
        if eigenvalues is not None:
            assert isinstance(eigenvalues, (tuple, np.ndarray))
    
    def test_eigenvectors_property(self, propagated_orbit):
        """Test eigenvectors property."""
        eigenvectors = propagated_orbit.eigenvectors
        
        # May be None or a tuple/array depending on computation
        if eigenvectors is not None:
            assert isinstance(eigenvectors, (tuple, np.ndarray))


class TestPeriodicOrbitIntegration:
    """Integration tests for PeriodicOrbit class."""
    
    def test_full_workflow_generic_orbit(self, l1_point, sample_initial_state):
        """Test a complete workflow with GenericOrbit."""
        # Create orbit
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        # Set period
        orbit.period = 2.5
        
        # Propagate
        trajectory = orbit.propagate(steps=100)
        
        # Verify
        assert trajectory is not None
        assert orbit.trajectory is not None
        assert len(trajectory.times) == 100
        
        # Export to DataFrame
        df = orbit.to_df()
        assert len(df) == 100
        
        # Create manifold
        manifold = orbit.manifold(stable=True, direction="positive")
        assert manifold is not None
    
    def test_multiple_orbits_same_libration_point(self, l1_point):
        """Test creating multiple orbits for the same libration point."""
        orbit1 = GenericOrbit(l1_point, initial_state=np.array([0.8, 0, 0, 0, 0, 0]))
        orbit2 = GenericOrbit(l1_point, initial_state=np.array([0.85, 0, 0, 0, 0, 0]))
        
        assert orbit1.libration_point is l1_point
        assert orbit2.libration_point is l1_point
        assert orbit1 is not orbit2
        assert not np.allclose(orbit1.initial_state, orbit2.initial_state)
    
    def test_different_orbit_types_same_point(self, l1_point, sample_initial_state):
        """Test creating different orbit types for the same libration point."""
        generic = GenericOrbit(l1_point, initial_state=sample_initial_state)
        halo = HaloOrbit(l1_point, amplitude_z=0.01, zenith="northern")
        lyap = LyapunovOrbit(l1_point, amplitude_x=0.01)
        vertical = VerticalOrbit(l1_point, initial_state=sample_initial_state)
        
        assert generic.family == "generic"
        assert halo.family == "halo"
        assert lyap.family == "lyapunov"
        assert vertical.family == "vertical"
        
        # All should reference the same libration point
        assert generic.libration_point is l1_point
        assert halo.libration_point is l1_point
        assert lyap.libration_point is l1_point
        assert vertical.libration_point is l1_point


class TestPeriodicOrbitEdgeCases:
    """Test PeriodicOrbit edge cases and error handling."""
    
    def test_orbit_with_zero_period_raises_error(self, l1_point, sample_initial_state):
        """Test orbit with zero period raises error."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        # Period must be positive
        with pytest.raises(ValueError, match="period must be a positive number"):
            orbit.period = 0.0
    
    def test_orbit_with_negative_period_raises_error(self, l1_point, sample_initial_state):
        """Test orbit with negative period raises error."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        # Period must be positive
        with pytest.raises(ValueError, match="period must be a positive number"):
            orbit.period = -2.5
    
    def test_orbit_with_very_large_period(self, l1_point, sample_initial_state):
        """Test orbit with very large period."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        orbit.period = 1e6
        
        assert orbit.period == 1e6
    
    def test_propagate_with_few_steps(self, l1_point, sample_initial_state):
        """Test propagation with very few steps."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        orbit.period = 2.5
        
        trajectory = orbit.propagate(steps=2)
        assert trajectory.n_samples == 2
    
    def test_orbit_properties_consistency(self, l1_point, sample_initial_state):
        """Test that orbit properties are consistent across accesses."""
        orbit = GenericOrbit(l1_point, initial_state=sample_initial_state)
        
        # Multiple accesses should return the same values
        state1 = orbit.initial_state
        state2 = orbit.initial_state
        assert np.allclose(state1, state2)
        
        mu1 = orbit.mu
        mu2 = orbit.mu
        assert mu1 == mu2
        
        energy1 = orbit.energy
        energy2 = orbit.energy
        assert energy1 == energy2


class TestPeriodicOrbitWithDifferentSystems:
    """Test PeriodicOrbit with different systems."""
    
    def test_orbit_earth_moon_vs_sun_earth(self):
        """Test orbits with different systems have different properties."""
        # Earth-Moon system
        em_system = System.from_bodies("earth", "moon")
        em_l1 = em_system.get_libration_point(1)
        em_orbit = GenericOrbit(em_l1, initial_state=np.array([0.8, 0, 0, 0, 0, 0]))
        
        # Sun-Earth system
        se_system = System.from_bodies("sun", "earth")
        se_l1 = se_system.get_libration_point(1)
        se_orbit = GenericOrbit(se_l1, initial_state=np.array([0.99, 0, 0, 0, 0, 0]))
        
        # Properties should be different
        assert em_orbit.mu != se_orbit.mu
        assert em_orbit.system is not se_orbit.system
        assert em_orbit.libration_point is not se_orbit.libration_point
    
    def test_orbit_with_custom_mu_system(self):
        """Test orbit with custom mu system."""
        system = System.from_mu(0.05)
        l1 = system.get_libration_point(1)
        orbit = GenericOrbit(l1, initial_state=np.array([0.85, 0, 0, 0, 0, 0]))
        
        assert orbit.mu == 0.05
        assert orbit.system is system
