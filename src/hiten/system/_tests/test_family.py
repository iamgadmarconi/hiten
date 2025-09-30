"""Tests for the OrbitFamily class API in family.py."""

import numpy as np
import pytest
from pathlib import Path

from hiten.system.base import System
from hiten.system.family import OrbitFamily
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
def sample_orbits(l1_point):
    """Create a list of sample orbits."""
    orbits = []
    for i in range(3):
        initial_state = np.array([l1_point.position[0] + 0.01 * (i + 1), 0, 0, 0, 0, 0])
        orbit = GenericOrbit(l1_point, initial_state=initial_state)
        orbit.period = 2.5 + 0.1 * i
        orbits.append(orbit)
    return orbits


@pytest.fixture
def orbit_family(sample_orbits):
    """Create an OrbitFamily for testing."""
    parameter_values = np.array([0.1, 0.2, 0.3])
    return OrbitFamily(sample_orbits, parameter_name="amplitude", parameter_values=parameter_values)


@pytest.fixture
def empty_orbit_family():
    """Create an empty OrbitFamily."""
    return OrbitFamily()


class TestOrbitFamilyInitialization:
    """Test OrbitFamily class initialization."""
    
    def test_family_initialization_with_orbits(self, sample_orbits):
        """Test OrbitFamily initialization with orbits."""
        parameter_values = np.array([0.1, 0.2, 0.3])
        family = OrbitFamily(sample_orbits, parameter_name="amplitude", parameter_values=parameter_values)
        
        assert family is not None
        assert len(family) == 3
        assert family.parameter_name == "amplitude"
        assert np.array_equal(family.parameter_values, parameter_values)
    
    def test_family_initialization_empty(self):
        """Test OrbitFamily initialization with no orbits."""
        family = OrbitFamily()
        
        assert family is not None
        assert len(family) == 0
        assert family.parameter_name == "param"
        assert len(family.parameter_values) == 0
    
    def test_family_initialization_without_parameter_values(self, sample_orbits):
        """Test OrbitFamily initialization without parameter values."""
        family = OrbitFamily(sample_orbits)
        
        assert len(family) == 3
        # Should create NaN values for parameters
        assert len(family.parameter_values) == 3
        assert np.all(np.isnan(family.parameter_values))
    
    def test_family_initialization_mismatched_parameter_values(self, sample_orbits):
        """Test that mismatched parameter values raises error."""
        parameter_values = np.array([0.1, 0.2])  # Only 2 values for 3 orbits
        
        with pytest.raises(ValueError, match="Length of parameter_values must match"):
            OrbitFamily(sample_orbits, parameter_values=parameter_values)
    
    def test_family_initialization_custom_parameter_name(self, sample_orbits):
        """Test OrbitFamily initialization with custom parameter name."""
        family = OrbitFamily(sample_orbits, parameter_name="energy")
        
        assert family.parameter_name == "energy"


class TestOrbitFamilyProperties:
    """Test OrbitFamily class properties."""
    
    def test_periods_property(self, orbit_family):
        """Test periods property."""
        periods = orbit_family.periods
        
        assert isinstance(periods, np.ndarray)
        assert len(periods) == len(orbit_family)
        assert periods[0] == 2.5
        assert periods[1] == 2.6
        assert periods[2] == 2.7
    
    def test_periods_property_with_default_period(self, l1_point):
        """Test periods property when some orbits have default period."""
        orbit1 = GenericOrbit(l1_point, initial_state=np.array([0.8, 0, 0, 0, 0, 0]))
        orbit1.period = 2.5
        
        orbit2 = GenericOrbit(l1_point, initial_state=np.array([0.8, 0, 0, 0, 0, 0]))
        # orbit2 should have default period of np.pi
        
        family = OrbitFamily([orbit1, orbit2])
        periods = family.periods
        
        assert periods[0] == 2.5
        assert np.isclose(periods[1], np.pi)
    
    def test_jacobis_property(self, orbit_family):
        """Test jacobis property."""
        jacobis = orbit_family.jacobis
        
        assert isinstance(jacobis, np.ndarray)
        assert len(jacobis) == len(orbit_family)
        # All should be finite numbers
        assert np.all(np.isfinite(jacobis))
    
    def test_parameter_name_property(self, orbit_family):
        """Test parameter_name property."""
        assert orbit_family.parameter_name == "amplitude"
    
    def test_parameter_values_property(self, orbit_family):
        """Test parameter_values property."""
        assert np.array_equal(orbit_family.parameter_values, np.array([0.1, 0.2, 0.3]))


class TestOrbitFamilyDunderMethods:
    """Test OrbitFamily dunder methods."""
    
    def test_len(self, orbit_family, empty_orbit_family):
        """Test __len__ method."""
        assert len(orbit_family) == 3
        assert len(empty_orbit_family) == 0
    
    def test_iter(self, orbit_family, sample_orbits):
        """Test __iter__ method."""
        orbits_from_iter = list(orbit_family)
        
        assert len(orbits_from_iter) == 3
        assert orbits_from_iter[0] is sample_orbits[0]
        assert orbits_from_iter[1] is sample_orbits[1]
        assert orbits_from_iter[2] is sample_orbits[2]
    
    def test_getitem_single_index(self, orbit_family, sample_orbits):
        """Test __getitem__ with single index."""
        assert orbit_family[0] is sample_orbits[0]
        assert orbit_family[1] is sample_orbits[1]
        assert orbit_family[2] is sample_orbits[2]
        assert orbit_family[-1] is sample_orbits[2]
    
    def test_getitem_slice(self, orbit_family, sample_orbits):
        """Test __getitem__ with slice."""
        sliced = orbit_family[0:2]
        
        assert len(sliced) == 2
        assert sliced[0] is sample_orbits[0]
        assert sliced[1] is sample_orbits[1]
    
    def test_str_representation(self, orbit_family):
        """Test __str__ representation."""
        str_repr = str(orbit_family)
        
        assert "OrbitFamily" in str_repr
        assert "n_orbits=3" in str_repr
        assert "amplitude" in str_repr
    
    def test_repr_representation(self, orbit_family):
        """Test __repr__ representation."""
        repr_str = repr(orbit_family)
        
        assert "OrbitFamily" in repr_str
        assert "n_orbits=3" in repr_str
        assert "amplitude" in repr_str
    
    def test_empty_family_str(self, empty_orbit_family):
        """Test __str__ for empty family."""
        str_repr = str(empty_orbit_family)
        
        assert "OrbitFamily" in str_repr
        assert "n_orbits=0" in str_repr


class TestOrbitFamilyPropagate:
    """Test OrbitFamily propagate method."""
    
    def test_propagate_all_orbits(self, orbit_family):
        """Test propagating all orbits in the family."""
        # Before propagation, accessing trajectory should raise ValueError
        for orbit in orbit_family:
            with pytest.raises(ValueError, match="Trajectory not computed"):
                _ = orbit.trajectory
        
        # Propagate
        orbit_family.propagate(steps=50)
        
        # After propagation, all should have trajectories
        for orbit in orbit_family:
            assert orbit.trajectory is not None
            assert orbit.trajectory.n_samples == 50
    
    def test_propagate_with_kwargs(self, orbit_family):
        """Test propagate with keyword arguments."""
        orbit_family.propagate(steps=100, method="adaptive")
        
        for orbit in orbit_family:
            assert orbit.trajectory is not None
            assert orbit.trajectory.n_samples > 0


class TestOrbitFamilyDataExport:
    """Test OrbitFamily data export methods."""
    
    def test_to_df_after_propagation(self, orbit_family):
        """Test to_df after propagation."""
        orbit_family.propagate(steps=50)
        
        df = orbit_family.to_df()
        
        assert df is not None
        assert "orbit_id" in df.columns
        assert "amplitude" in df.columns  # parameter_name
        assert "time" in df.columns
        assert "x" in df.columns
        assert "y" in df.columns
        assert "z" in df.columns
        assert "vx" in df.columns
        assert "vy" in df.columns
        assert "vz" in df.columns
        
        # Should have data for all 3 orbits
        assert len(df["orbit_id"].unique()) == 3
    
    def test_to_df_without_propagation(self, orbit_family):
        """Test to_df without prior propagation (should auto-propagate)."""
        # Don't propagate manually
        df = orbit_family.to_df(steps=50)
        
        assert df is not None
        assert len(df) > 0
        
        # All orbits should now have trajectories
        for orbit in orbit_family:
            assert orbit.trajectory is not None
    
    def test_to_csv(self, orbit_family):
        """Test to_csv export."""
        orbit_family.propagate(steps=50)
        
        temp_path = Path("temp_family.csv")
        
        try:
            orbit_family.to_csv(str(temp_path))
            
            assert temp_path.exists()
            
            # Read and verify
            import pandas as pd
            df = pd.read_csv(temp_path)
            assert "orbit_id" in df.columns
            assert "amplitude" in df.columns
            assert len(df) > 0
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_to_csv_creates_directory(self, orbit_family):
        """Test that to_csv creates parent directories if needed."""
        orbit_family.propagate(steps=50)
        
        temp_dir = Path("temp_test_dir")
        temp_path = temp_dir / "family.csv"
        
        try:
            orbit_family.to_csv(str(temp_path))
            
            assert temp_path.exists()
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()


class TestOrbitFamilyPlotting:
    """Test OrbitFamily plotting methods."""
    
    def test_plot_after_propagation(self, orbit_family):
        """Test plotting after propagation."""
        orbit_family.propagate(steps=50)
        
        fig = orbit_family.plot(save=False)
        assert fig is not None
    
    def test_plot_without_propagation_raises_error(self, orbit_family):
        """Test that plotting without propagation raises error."""
        # Don't propagate
        with pytest.raises(ValueError, match="no trajectory data"):
            orbit_family.plot()
    
    def test_plot_with_custom_parameters(self, orbit_family):
        """Test plotting with custom parameters."""
        orbit_family.propagate(steps=50)
        
        fig = orbit_family.plot(
            dark_mode=False,
            save=False
        )
        assert fig is not None


class TestOrbitFamilySerialization:
    """Test OrbitFamily serialization methods."""
    
    def test_family_serialization_basic(self, orbit_family):
        """Test basic family serialization."""
        temp_path = Path("temp_family.pkl")
        
        try:
            orbit_family.save(str(temp_path))
            
            family_loaded = OrbitFamily.load(str(temp_path))
            
            # Verify properties are preserved
            assert len(family_loaded) == len(orbit_family)
            assert family_loaded.parameter_name == orbit_family.parameter_name
            assert np.array_equal(family_loaded.parameter_values, orbit_family.parameter_values)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_family_serialization_with_propagated_orbits(self, orbit_family):
        """Test family serialization with propagated orbits."""
        orbit_family.propagate(steps=50)
        
        temp_path = Path("temp_family_propagated.pkl")
        
        try:
            orbit_family.save(str(temp_path))
            
            family_loaded = OrbitFamily.load(str(temp_path))
            
            # Verify properties are preserved
            assert len(family_loaded) == len(orbit_family)
            assert family_loaded.parameter_name == orbit_family.parameter_name
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_family_serialization_with_compression(self, orbit_family):
        """Test family serialization with custom compression."""
        temp_path = Path("temp_family_compressed.pkl")
        
        try:
            orbit_family.save(str(temp_path), compression="gzip", level=9)
            
            family_loaded = OrbitFamily.load(str(temp_path))
            
            assert len(family_loaded) == len(orbit_family)
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestOrbitFamilyFromResult:
    """Test OrbitFamily.from_result class method."""
    
    def test_from_result_basic(self, sample_orbits):
        """Test from_result with a mock result."""
        # Create a mock continuation result
        class MockResult:
            def __init__(self, orbits, param_vals):
                self.family = orbits
                self.parameter_values = param_vals
        
        param_vals = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
        result = MockResult(sample_orbits, param_vals)
        
        family = OrbitFamily.from_result(result, parameter_name="amplitude")
        
        assert len(family) == 3
        assert family.parameter_name == "amplitude"
        assert np.allclose(family.parameter_values, [0.1, 0.2, 0.3])
    
    def test_from_result_default_parameter_name(self, sample_orbits):
        """Test from_result with default parameter name."""
        class MockResult:
            def __init__(self, orbits, param_vals):
                self.family = orbits
                self.parameter_values = param_vals
        
        param_vals = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
        result = MockResult(sample_orbits, param_vals)
        
        family = OrbitFamily.from_result(result)
        
        assert family.parameter_name == "param"
    
    def test_from_result_with_scalar_parameters(self, sample_orbits):
        """Test from_result with scalar parameter values."""
        class MockResult:
            def __init__(self, orbits, param_vals):
                self.family = orbits
                self.parameter_values = param_vals
        
        # Scalar values
        param_vals = [0.1, 0.2, 0.3]
        result = MockResult(sample_orbits, param_vals)
        
        family = OrbitFamily.from_result(result, parameter_name="energy")
        
        assert np.allclose(family.parameter_values, [0.1, 0.2, 0.3])
    
    def test_from_result_with_multidimensional_parameters(self, sample_orbits):
        """Test from_result with multi-dimensional parameter values."""
        class MockResult:
            def __init__(self, orbits, param_vals):
                self.family = orbits
                self.parameter_values = param_vals
        
        # Multi-dimensional parameters (should use norm)
        param_vals = [
            np.array([0.1, 0.2]),
            np.array([0.3, 0.4]),
            np.array([0.5, 0.6])
        ]
        result = MockResult(sample_orbits, param_vals)
        
        family = OrbitFamily.from_result(result)
        
        # Should use Euclidean norm
        expected = [np.linalg.norm([0.1, 0.2]), np.linalg.norm([0.3, 0.4]), np.linalg.norm([0.5, 0.6])]
        assert np.allclose(family.parameter_values, expected)


class TestOrbitFamilyIntegration:
    """Integration tests for OrbitFamily class."""
    
    def test_full_workflow(self, sample_orbits):
        """Test a complete orbit family workflow."""
        # Create family
        parameter_values = np.array([0.1, 0.2, 0.3])
        family = OrbitFamily(sample_orbits, parameter_name="amplitude", parameter_values=parameter_values)
        
        # Verify initial state
        assert len(family) == 3
        assert family.parameter_name == "amplitude"
        
        # Propagate
        family.propagate(steps=50)
        
        # Export to DataFrame
        df = family.to_df()
        assert len(df) > 0
        
        # Plot
        fig = family.plot(save=False)
        assert fig is not None
    
    def test_iteration_and_access(self, orbit_family):
        """Test various ways to iterate and access orbits."""
        # Via iteration
        count = 0
        for orbit in orbit_family:
            assert orbit is not None
            count += 1
        assert count == 3
        
        # Via indexing
        first = orbit_family[0]
        assert first is not None
        
        # Via slicing
        subset = orbit_family[0:2]
        assert len(subset) == 2


class TestOrbitFamilyEdgeCases:
    """Test OrbitFamily edge cases and error handling."""
    
    def test_empty_family_operations(self, empty_orbit_family):
        """Test operations on empty family."""
        assert len(empty_orbit_family) == 0
        assert len(list(empty_orbit_family)) == 0
        assert len(empty_orbit_family.periods) == 0
        assert len(empty_orbit_family.jacobis) == 0
    
    def test_single_orbit_family(self, l1_point):
        """Test family with single orbit."""
        orbit = GenericOrbit(l1_point, initial_state=np.array([0.8, 0, 0, 0, 0, 0]))
        orbit.period = 2.5
        
        family = OrbitFamily([orbit], parameter_values=np.array([0.1]))
        
        assert len(family) == 1
        assert family[0] is orbit
    
    def test_family_with_different_orbit_types(self, l1_point):
        """Test family with different types of orbits."""
        orbit1 = GenericOrbit(l1_point, initial_state=np.array([0.8, 0, 0, 0, 0, 0]))
        orbit1.period = 2.5
        
        orbit2 = GenericOrbit(l1_point, initial_state=np.array([0.85, 0, 0, 0, 0, 0]))
        orbit2.period = 2.6
        
        family = OrbitFamily([orbit1, orbit2])
        
        assert len(family) == 2
    
    def test_parameter_values_array_conversion(self, sample_orbits):
        """Test that parameter values are converted to array."""
        # Pass list instead of array
        param_vals = [0.1, 0.2, 0.3]
        family = OrbitFamily(sample_orbits, parameter_values=param_vals)
        
        assert isinstance(family.parameter_values, np.ndarray)
        assert np.array_equal(family.parameter_values, [0.1, 0.2, 0.3])
    
    def test_family_properties_consistency(self, orbit_family):
        """Test that family properties are consistent across accesses."""
        # Multiple accesses should return the same values
        len1 = len(orbit_family)
        len2 = len(orbit_family)
        assert len1 == len2
        
        periods1 = orbit_family.periods
        periods2 = orbit_family.periods
        assert np.array_equal(periods1, periods2)
        
        jacobis1 = orbit_family.jacobis
        jacobis2 = orbit_family.jacobis
        assert np.array_equal(jacobis1, jacobis2)


class TestOrbitFamilyAccessPatterns:
    """Test various access patterns for OrbitFamily."""
    
    def test_negative_indexing(self, orbit_family):
        """Test negative indexing."""
        assert orbit_family[-1] is orbit_family[2]
        assert orbit_family[-2] is orbit_family[1]
        assert orbit_family[-3] is orbit_family[0]
    
    def test_slice_patterns(self, orbit_family):
        """Test various slice patterns."""
        # Forward slice
        subset1 = orbit_family[0:2]
        assert len(subset1) == 2
        
        # Step slice
        subset2 = orbit_family[::2]
        assert len(subset2) == 2
        
        # Reverse slice
        subset3 = orbit_family[::-1]
        assert len(subset3) == 3
        assert subset3[0] is orbit_family[2]
    
    def test_out_of_bounds_indexing(self, orbit_family):
        """Test out of bounds indexing raises error."""
        with pytest.raises(IndexError):
            _ = orbit_family[10]


class TestOrbitFamilyModification:
    """Test modifying OrbitFamily after creation."""
    
    def test_modify_orbits_list(self, orbit_family, l1_point):
        """Test modifying the orbits list after creation."""
        original_len = len(orbit_family)
        
        # Add a new orbit
        new_orbit = GenericOrbit(l1_point, initial_state=np.array([0.9, 0, 0, 0, 0, 0]))
        new_orbit.period = 3.0
        orbit_family.orbits.append(new_orbit)
        
        # Length should increase
        assert len(orbit_family) == original_len + 1
    
    def test_modify_parameter_name(self, orbit_family):
        """Test modifying parameter name after creation."""
        original_name = orbit_family.parameter_name
        
        orbit_family.parameter_name = "energy"
        
        assert orbit_family.parameter_name == "energy"
        assert orbit_family.parameter_name != original_name
    
    def test_modify_parameter_values(self, orbit_family):
        """Test modifying parameter values after creation."""
        new_values = np.array([1.0, 2.0, 3.0])
        orbit_family.parameter_values = new_values
        
        assert np.array_equal(orbit_family.parameter_values, new_values)
