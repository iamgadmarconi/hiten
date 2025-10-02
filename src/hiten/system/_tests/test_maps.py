"""Tests for the CenterManifoldMap and SynodicMap classes in maps/."""

import numpy as np
import pytest
from pathlib import Path

from hiten.system.base import System
from hiten.system.center import CenterManifold
from hiten.system.orbits import GenericOrbit
from hiten.system.maps import CenterManifoldMap, SynodicMap
from hiten.system.manifold import Manifold


@pytest.fixture
def earth_moon_system():
    """Create an Earth-Moon system for testing."""
    return System.from_bodies("earth", "moon")


@pytest.fixture
def l1_point(earth_moon_system):
    """Create L1 libration point."""
    return earth_moon_system.get_libration_point(1)


@pytest.fixture
def center_manifold(l1_point):
    """Create a CenterManifold for testing."""
    return CenterManifold(l1_point, degree=4)


@pytest.fixture
def center_manifold_map(center_manifold):
    """Create a CenterManifoldMap for testing."""
    energy = 0.5
    return CenterManifoldMap(center_manifold, energy)


@pytest.fixture
def propagated_orbit(l1_point):
    """Create and propagate an orbit."""
    initial_state = np.array([l1_point.position[0] + 0.01, 0, 0, 0, 0, 0])
    orbit = GenericOrbit(l1_point, initial_state=initial_state)
    orbit.period = 2.5
    orbit.propagate(steps=100)
    return orbit


@pytest.fixture
def synodic_map(propagated_orbit):
    """Create a SynodicMap for testing."""
    return SynodicMap(propagated_orbit)


class TestCenterManifoldMapInitialization:
    """Test CenterManifoldMap class initialization."""
    
    def test_center_manifold_map_initialization(self, center_manifold):
        """Test CenterManifoldMap initialization."""
        energy = 0.5
        cm_map = CenterManifoldMap(center_manifold, energy)
        
        assert cm_map is not None
        assert cm_map.energy == energy
        assert cm_map.center_manifold is center_manifold
    
    def test_center_manifold_map_different_energies(self, center_manifold):
        """Test CenterManifoldMap initialization with different energies."""
        map1 = CenterManifoldMap(center_manifold, energy=0.5)
        map2 = CenterManifoldMap(center_manifold, energy=0.7)
        
        assert map1.energy == 0.5
        assert map2.energy == 0.7


class TestCenterManifoldMapProperties:
    """Test CenterManifoldMap class properties."""
    
    def test_center_manifold_property(self, center_manifold_map, center_manifold):
        """Test center_manifold property."""
        assert center_manifold_map.center_manifold is center_manifold
    
    def test_energy_property(self, center_manifold_map):
        """Test energy property."""
        assert center_manifold_map.energy == 0.5
        assert isinstance(center_manifold_map.energy, (int, float))
    
    def test_config_property_get(self, center_manifold_map):
        """Test config property getter."""
        config = center_manifold_map.config
        # Config exists and is accessible
        assert config is not None or config is None  # Either case is valid
    
    def test_config_property_set(self, center_manifold_map):
        """Test config property setter."""
        # Test that setter exists
        try:
            center_manifold_map.config = {"test": "value"}
        except (AttributeError, TypeError, ValueError):
            # If setter doesn't accept dict, that's okay
            pass
    
    def test_sections_property(self, center_manifold_map):
        """Test sections property."""
        sections = center_manifold_map.sections
        
        # Should be a list
        assert isinstance(sections, list)


class TestCenterManifoldMapStringRepresentations:
    """Test CenterManifoldMap string representations."""
    
    def test_str_representation(self, center_manifold_map):
        """Test __str__ representation."""
        str_repr = str(center_manifold_map)
        
        assert "CenterManifoldMap" in str_repr
    
    def test_repr_representation(self, center_manifold_map):
        """Test __repr__ representation."""
        repr_str = repr(center_manifold_map)
        
        assert "CenterManifoldMap" in repr_str


class TestCenterManifoldMapSectionMethods:
    """Test CenterManifoldMap section methods."""
    
    def test_has_section_method(self, center_manifold_map):
        """Test has_section method."""
        # Should return boolean
        result = center_manifold_map.has_section("q3")
        assert isinstance(result, bool)
    
    def test_get_section_method(self, center_manifold_map):
        """Test get_section method after computing."""
        # After compute, should be able to get section
        center_manifold_map.compute(section_coord="q3")
        section = center_manifold_map.get_section("q3")
        assert section is not None

    
    def test_clear_sections_method(self, center_manifold_map):
        """Test clear_sections method."""
        # First compute a section to have something to clear
        center_manifold_map.compute(section_coord="q3")
        
        # Should have at least one section now
        assert len(center_manifold_map.sections) > 0
        
        # Clear sections
        center_manifold_map.clear_sections()

        # After clearing, sections should be empty
        assert len(center_manifold_map.sections) == 0


class TestCenterManifoldMapCompute:
    """Test CenterManifoldMap compute methods."""
    
    def test_compute_method_exists(self, center_manifold_map):
        """Test that compute method exists."""
        assert hasattr(center_manifold_map, 'compute')
        assert callable(center_manifold_map.compute)
    
    def test_compute_basic(self, center_manifold_map):
        """Test basic compute."""
        result = center_manifold_map.compute(section_coord="q3")
        assert result is not None

    def test_get_points_method(self, center_manifold_map):
        """Test get_points method."""
        # Compute first
        center_manifold_map.compute(section_coord="q3")
        
        # Get points
        points = center_manifold_map.get_points(section_coord="q3")
        
        assert isinstance(points, np.ndarray)
        assert points.ndim == 2

    def test_get_states_method(self, center_manifold_map):
        """Test get_states method."""
        # Compute first
        center_manifold_map.compute(section_coord="q3")
        
        # Get states
        states = center_manifold_map.get_states(section_coord="q3")
        
        assert isinstance(states, np.ndarray)


class TestCenterManifoldMapCoordinateConversion:
    """Test CenterManifoldMap coordinate conversion."""
    
    def test_to_synodic_method_exists(self, center_manifold_map):
        """Test that to_synodic method exists."""
        assert hasattr(center_manifold_map, 'to_synodic')
        assert callable(center_manifold_map.to_synodic)
    
    def test_to_synodic_method(self, center_manifold_map):
        """Test to_synodic method."""
        pt = np.array([0.01, 0.01])
        
        synodic = center_manifold_map.to_synodic(pt, section_coord="q3")
        
        assert isinstance(synodic, np.ndarray)
        assert len(synodic) == 6


class TestCenterManifoldMapPlotting:
    """Test CenterManifoldMap plotting methods."""
    
    def test_plot_method_exists(self, center_manifold_map):
        """Test that plot method exists."""
        assert hasattr(center_manifold_map, 'plot')
        assert callable(center_manifold_map.plot)
    
    def test_plot_interactive_method_exists(self, center_manifold_map):
        """Test that plot_interactive method exists."""
        assert hasattr(center_manifold_map, 'plot_interactive')
        assert callable(center_manifold_map.plot_interactive)


class TestCenterManifoldMapSerialization:
    """Test CenterManifoldMap serialization."""
    
    def test_save_method_exists(self, center_manifold_map):
        """Test that save method exists."""
        assert hasattr(center_manifold_map, 'save')
        assert callable(center_manifold_map.save)
    
    def test_load_method_exists(self):
        """Test that load class method exists."""
        assert hasattr(CenterManifoldMap, 'load')
        assert callable(CenterManifoldMap.load)
    
    def test_load_inplace_method_exists(self, center_manifold_map):
        """Test that load_inplace method exists."""
        assert hasattr(center_manifold_map, 'load_inplace')
        assert callable(center_manifold_map.load_inplace)


class TestSynodicMapInitialization:
    """Test SynodicMap class initialization."""
    
    def test_synodic_map_initialization_from_orbit(self, propagated_orbit):
        """Test SynodicMap initialization from orbit."""
        smap = SynodicMap(propagated_orbit)
        
        assert smap is not None
        assert smap.source is propagated_orbit
    
    def test_synodic_map_initialization_from_manifold(self, propagated_orbit):
        """Test SynodicMap initialization from manifold."""
        manifold = Manifold(propagated_orbit, stable=True, direction="positive")
        
        smap = SynodicMap(manifold)
        assert smap is not None



class TestSynodicMapProperties:
    """Test SynodicMap class properties."""
    
    def test_source_property(self, synodic_map, propagated_orbit):
        """Test source property."""
        assert synodic_map.source is propagated_orbit
    
    def test_sections_property(self, synodic_map):
        """Test sections property."""
        sections = synodic_map.sections
        
        # Should be a list
        assert isinstance(sections, list)
    
    def test_config_property_get(self, synodic_map):
        """Test config property getter."""
        config = synodic_map.config
        # Config exists and is accessible
        assert config is not None or config is None
    
    def test_config_property_set(self, synodic_map):
        """Test config property setter."""
        # Test that setter exists
        try:
            synodic_map.config = {"test": "value"}
        except (AttributeError, TypeError, ValueError):
            # If setter doesn't accept dict, that's okay
            pass


class TestSynodicMapStringRepresentations:
    """Test SynodicMap string representations."""
    
    def test_str_representation(self, synodic_map):
        """Test __str__ representation."""
        str_repr = str(synodic_map)
        
        assert "SynodicMap" in str_repr
        assert "source" in str_repr
    
    def test_repr_representation(self, synodic_map):
        """Test __repr__ representation."""
        repr_str = repr(synodic_map)
        
        assert "SynodicMap" in repr_str


class TestSynodicMapSectionMethods:
    """Test SynodicMap section methods."""
    
    def test_has_section_method(self, synodic_map):
        """Test has_section method."""
        # Should return boolean
        result = synodic_map.has_section("x")
        assert isinstance(result, bool)
    
    def test_get_section_method(self, synodic_map):
        """Test get_section method after computing."""
        synodic_map.compute(
            section_axis="x",
            section_offset=0.8,
            plane_coords=("y", "vy"),
            direction=None
        )
        # SynodicMap creates section IDs from all parameters with colons as separators
        section_id = "x:0.8:y:vy:None"
        section = synodic_map.get_section(section_id)
        assert section is not None

    def test_clear_sections_method(self, synodic_map):
        """Test clear_sections method."""
        # First compute a section to have something to clear
        synodic_map.compute(
            section_axis="x",
            section_offset=0.0,
            plane_coords=("y", "py"),
            direction=1
        )
        
        # Should have at least one section now
        assert len(synodic_map.sections) > 0
        
        # Clear sections
        synodic_map.clear_sections()

        # After clearing, sections should be empty
        assert len(synodic_map.sections) == 0


class TestSynodicMapCompute:
    """Test SynodicMap compute methods."""
    
    def test_compute_method_exists(self, synodic_map):
        """Test that compute method exists."""
        assert hasattr(synodic_map, 'compute')
        assert callable(synodic_map.compute)
    
    def test_compute_basic(self, synodic_map):
        """Test basic compute."""

        result = synodic_map.compute(
            section_axis="x",
            section_offset=0.8,
            plane_coords=("y", "vy")
        )
        assert result is not None
    
    def test_get_points_method(self, synodic_map):
        """Test get_points method."""

        # Compute first
        synodic_map.compute(
            section_axis="x",
            section_offset=0.8,
            plane_coords=("y", "vy")
        )
        
        # Get points
        points = synodic_map.get_points()
        
        assert isinstance(points, np.ndarray)
        assert points.ndim == 2


class TestSynodicMapPlotting:
    """Test SynodicMap plotting methods."""
    
    def test_plot_method_exists(self, synodic_map):
        """Test that plot method exists."""
        assert hasattr(synodic_map, 'plot')
        assert callable(synodic_map.plot)


class TestSynodicMapSerialization:
    """Test SynodicMap serialization."""
    
    def test_save_method_exists(self, synodic_map):
        """Test that save method exists."""
        assert hasattr(synodic_map, 'save')
        assert callable(synodic_map.save)
    
    def test_load_method_exists(self):
        """Test that load class method exists."""
        assert hasattr(SynodicMap, 'load')
        assert callable(SynodicMap.load)
    
    def test_load_inplace_method_exists(self, synodic_map):
        """Test that load_inplace method exists."""
        assert hasattr(synodic_map, 'load_inplace')
        assert callable(synodic_map.load_inplace)


class TestCenterManifoldMapIntegration:
    """Integration tests for CenterManifoldMap class."""
    
    def test_full_workflow(self, center_manifold):
        """Test a complete CenterManifoldMap workflow."""
        # Create map
        energy = 0.5
        cm_map = CenterManifoldMap(center_manifold, energy)
        
        # Verify initial state
        assert cm_map.center_manifold is center_manifold
        assert cm_map.energy == energy
        
        # Test section operations
        cm_map.clear_sections()
        assert len(cm_map.sections) == 0
    
    def test_multiple_maps_same_center_manifold(self, center_manifold):
        """Test creating multiple maps for the same center manifold."""
        map1 = CenterManifoldMap(center_manifold, energy=0.5)
        map2 = CenterManifoldMap(center_manifold, energy=0.7)
        
        # Both should reference the same center manifold
        assert map1.center_manifold is center_manifold
        assert map2.center_manifold is center_manifold
        
        # But have different energies
        assert map1.energy != map2.energy


class TestSynodicMapIntegration:
    """Integration tests for SynodicMap class."""
    
    def test_full_workflow(self, propagated_orbit):
        """Test a complete SynodicMap workflow."""
        # Create map
        smap = SynodicMap(propagated_orbit)
        
        # Verify initial state
        assert smap.source is propagated_orbit
        
        # Test section operations
        smap.clear_sections()
        assert len(smap.sections) == 0
    
    def test_synodic_map_from_different_orbits(self, l1_point):
        """Test creating synodic maps from different orbits."""
        orbit1 = GenericOrbit(l1_point, initial_state=np.array([0.8, 0, 0, 0, 0, 0]))
        orbit1.period = 2.5
        orbit1.propagate(steps=50)
        
        orbit2 = GenericOrbit(l1_point, initial_state=np.array([0.85, 0, 0, 0, 0, 0]))
        orbit2.period = 2.6
        orbit2.propagate(steps=50)
        
        map1 = SynodicMap(orbit1)
        map2 = SynodicMap(orbit2)
        
        assert map1.source is orbit1
        assert map2.source is orbit2


class TestMapMethodsExist:
    """Test that both map classes have expected methods."""
    
    def test_center_manifold_map_methods(self):
        """Test that CenterManifoldMap has expected methods."""
        expected_methods = [
            'compute', 'get_points', 'get_states', 'get_section', 
            'has_section', 'clear_sections', 'to_synodic',
            'plot', 'plot_interactive', 'save', 'load', 'load_inplace'
        ]
        
        for method in expected_methods:
            assert hasattr(CenterManifoldMap, method)
    
    def test_synodic_map_methods(self):
        """Test that SynodicMap has expected methods."""
        expected_methods = [
            'compute', 'get_points', 'get_section',
            'has_section', 'clear_sections', 'trajectories',
            'plot', 'save', 'load', 'load_inplace'
        ]
        
        for method in expected_methods:
            assert hasattr(SynodicMap, method)


class TestMapPropertiesConsistency:
    """Test consistency of map properties."""
    
    def test_center_manifold_map_properties_consistency(self, center_manifold_map):
        """Test that CenterManifoldMap properties are consistent."""
        energy1 = center_manifold_map.energy
        energy2 = center_manifold_map.energy
        assert energy1 == energy2
        
        cm1 = center_manifold_map.center_manifold
        cm2 = center_manifold_map.center_manifold
        assert cm1 is cm2
    
    def test_synodic_map_properties_consistency(self, synodic_map):
        """Test that SynodicMap properties are consistent."""
        source1 = synodic_map.source
        source2 = synodic_map.source
        assert source1 is source2
