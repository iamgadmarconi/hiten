"""Tests for the Body class API in body.py."""

import pytest
from pathlib import Path

from hiten.system.body import Body
from hiten.utils.constants import Constants


class TestBodyInitialization:
    """Test Body class initialization and basic properties."""
    
    def test_body_initialization_basic(self):
        """Test basic Body initialization."""
        body = Body("Earth", 5.972e24, 6.371e6, color="#0000FF")
        
        assert body.name == "Earth"
        assert body.mass == 5.972e24
        assert body.radius == 6.371e6
        assert body.color == "#0000FF"
        # Without a parent, the body is its own parent
        assert body.parent is body
    
    def test_body_initialization_with_parent(self):
        """Test Body initialization with a parent."""
        sun = Body("Sun", 1.989e30, 6.96e8, color="#FFFF00")
        earth = Body("Earth", 5.972e24, 6.371e6, color="#0000FF", parent=sun)
        
        assert earth.name == "Earth"
        assert earth.mass == 5.972e24
        assert earth.radius == 6.371e6
        assert earth.color == "#0000FF"
        assert earth.parent is sun
        assert earth.parent is not earth
    
    def test_body_initialization_default_color(self):
        """Test Body initialization with default color."""
        body = Body("Moon", 7.342e22, 1.737e6)
        
        assert body.name == "Moon"
        assert body.mass == 7.342e22
        assert body.radius == 1.737e6
        assert body.color == "#000000"  # Default color
    
    def test_body_initialization_hierarchical(self):
        """Test hierarchical Body initialization."""
        sun = Body("Sun", 1.989e30, 6.96e8)
        earth = Body("Earth", 5.972e24, 6.371e6, parent=sun)
        moon = Body("Moon", 7.342e22, 1.737e6, parent=earth)
        
        assert moon.parent is earth
        assert earth.parent is sun
        assert sun.parent is sun  # Sun is its own parent


class TestBodyProperties:
    """Test Body class properties."""
    
    @pytest.fixture
    def earth(self):
        """Create an Earth Body instance."""
        return Body("Earth", 5.972e24, 6.371e6, color="#0000FF")
    
    @pytest.fixture
    def moon_with_parent(self, earth):
        """Create a Moon Body instance with Earth as parent."""
        return Body("Moon", 7.342e22, 1.737e6, color="#808080", parent=earth)
    
    def test_name_property(self, earth):
        """Test name property access."""
        assert earth.name == "Earth"
        assert isinstance(earth.name, str)
    
    def test_mass_property(self, earth):
        """Test mass property access."""
        assert earth.mass == 5.972e24
        assert isinstance(earth.mass, float)
        assert earth.mass > 0
    
    def test_radius_property(self, earth):
        """Test radius property access."""
        assert earth.radius == 6.371e6
        assert isinstance(earth.radius, float)
        assert earth.radius > 0
    
    def test_color_property(self, earth):
        """Test color property access."""
        assert earth.color == "#0000FF"
        assert isinstance(earth.color, str)
    
    def test_parent_property_primary(self, earth):
        """Test parent property for a primary body (self-referencing)."""
        assert earth.parent is earth
    
    def test_parent_property_secondary(self, moon_with_parent, earth):
        """Test parent property for a secondary body."""
        assert moon_with_parent.parent is earth
        assert moon_with_parent.parent is not moon_with_parent


class TestBodyStringRepresentations:
    """Test Body string representations."""
    
    def test_str_primary_body(self):
        """Test __str__ for a primary body."""
        sun = Body("Sun", 1.989e30, 6.96e8)
        str_repr = str(sun)
        
        assert "Sun" in str_repr
        assert "(Primary)" in str_repr
    
    def test_str_secondary_body(self):
        """Test __str__ for a secondary body."""
        earth = Body("Earth", 5.972e24, 6.371e6)
        moon = Body("Moon", 7.342e22, 1.737e6, parent=earth)
        str_repr = str(moon)
        
        assert "Moon" in str_repr
        assert "orbiting Earth" in str_repr
    
    def test_repr_primary_body(self):
        """Test __repr__ for a primary body."""
        sun = Body("Sun", 1.989e30, 6.96e8, color="#FFFF00")
        repr_str = repr(sun)
        
        assert "Body(" in repr_str
        assert "name='Sun'" in repr_str
        assert "mass=" in repr_str
        assert "radius=" in repr_str
        assert "color='#FFFF00'" in repr_str
        # Primary body should not have parent in repr
        assert "parent=" not in repr_str
    
    def test_repr_secondary_body(self):
        """Test __repr__ for a secondary body."""
        earth = Body("Earth", 5.972e24, 6.371e6)
        moon = Body("Moon", 7.342e22, 1.737e6, color="#808080", parent=earth)
        repr_str = repr(moon)
        
        assert "Body(" in repr_str
        assert "name='Moon'" in repr_str
        assert "mass=" in repr_str
        assert "radius=" in repr_str
        assert "color='#808080'" in repr_str
        assert "parent=Body(name='Earth', ...)" in repr_str


class TestBodyWithConstants:
    """Test Body creation using Constants."""
    
    def test_body_from_constants_earth(self):
        """Test creating Earth Body using Constants."""
        earth_mass = Constants.get_mass("earth")
        earth_radius = Constants.get_radius("earth")
        
        earth = Body("Earth", earth_mass, earth_radius)
        
        assert earth.name == "Earth"
        assert earth.mass == earth_mass
        assert earth.radius == earth_radius
        assert earth.mass > 0
        assert earth.radius > 0
    
    def test_body_from_constants_moon(self):
        """Test creating Moon Body using Constants."""
        moon_mass = Constants.get_mass("moon")
        moon_radius = Constants.get_radius("moon")
        
        moon = Body("Moon", moon_mass, moon_radius)
        
        assert moon.name == "Moon"
        assert moon.mass == moon_mass
        assert moon.radius == moon_radius
        assert moon.mass > 0
        assert moon.radius > 0
    
    def test_earth_moon_system_from_constants(self):
        """Test creating Earth-Moon system using Constants."""
        earth_mass = Constants.get_mass("earth")
        earth_radius = Constants.get_radius("earth")
        moon_mass = Constants.get_mass("moon")
        moon_radius = Constants.get_radius("moon")
        
        earth = Body("Earth", earth_mass, earth_radius, color="blue")
        moon = Body("Moon", moon_mass, moon_radius, color="gray", parent=earth)
        
        assert earth.parent is earth
        assert moon.parent is earth
        assert moon.mass < earth.mass
        assert moon.radius < earth.radius


class TestBodySerialization:
    """Test Body serialization methods."""
    
    def test_body_serialization_roundtrip_primary(self):
        """Test Body serialization and deserialization for primary body."""
        earth_original = Body("Earth", 5.972e24, 6.371e6, color="#0000FF")
        
        # Save to temporary file
        temp_path = Path("temp_body.pkl")
        try:
            earth_original.save(str(temp_path))
            
            # Load from file
            earth_loaded = Body.load(str(temp_path))
            
            # Verify properties are preserved
            assert earth_loaded.name == earth_original.name
            assert earth_loaded.mass == earth_original.mass
            assert earth_loaded.radius == earth_original.radius
            assert earth_loaded.color == earth_original.color
            assert earth_loaded.parent is earth_loaded  # Self-referencing preserved
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_body_serialization_roundtrip_with_parent(self):
        """Test Body serialization and deserialization for body with parent."""
        earth = Body("Earth", 5.972e24, 6.371e6, color="#0000FF")
        moon_original = Body("Moon", 7.342e22, 1.737e6, color="#808080", parent=earth)
        
        # Save to temporary file
        temp_path = Path("temp_moon.pkl")
        try:
            moon_original.save(str(temp_path))
            
            # Load from file
            moon_loaded = Body.load(str(temp_path))
            
            # Verify properties are preserved
            assert moon_loaded.name == moon_original.name
            assert moon_loaded.mass == moon_original.mass
            assert moon_loaded.radius == moon_original.radius
            assert moon_loaded.color == moon_original.color
            
            # Parent should be preserved
            assert moon_loaded.parent.name == earth.name
            assert moon_loaded.parent.mass == earth.mass
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()


class TestBodyEdgeCases:
    """Test Body error handling and edge cases."""
    
    def test_zero_mass_body(self):
        """Test Body with zero mass."""
        body = Body("Massless", 0.0, 1.0)
        assert body.mass == 0.0
        assert body.name == "Massless"
    
    def test_zero_radius_body(self):
        """Test Body with zero radius."""
        body = Body("Point", 1.0, 0.0)
        assert body.radius == 0.0
        assert body.name == "Point"
    
    def test_negative_mass_body(self):
        """Test Body with negative mass (unusual but allowed)."""
        body = Body("Antimatter", -1.0, 1.0)
        assert body.mass == -1.0
        assert body.name == "Antimatter"
    
    def test_negative_radius_body(self):
        """Test Body with negative radius (unusual but allowed)."""
        body = Body("Inverted", 1.0, -1.0)
        assert body.radius == -1.0
        assert body.name == "Inverted"
    
    def test_very_large_values(self):
        """Test Body with very large mass and radius values."""
        body = Body("Supermassive", 1e50, 1e20)
        assert body.mass == 1e50
        assert body.radius == 1e20
        assert body.name == "Supermassive"
    
    def test_very_small_values(self):
        """Test Body with very small mass and radius values."""
        body = Body("Tiny", 1e-50, 1e-20)
        assert body.mass == 1e-50
        assert body.radius == 1e-20
        assert body.name == "Tiny"
    
    def test_special_characters_in_name(self):
        """Test Body with special characters in name."""
        body = Body("Test-Body_123", 1.0, 1.0)
        assert body.name == "Test-Body_123"
    
    def test_unicode_in_name(self):
        """Test Body with unicode characters in name."""
        body = Body("Étoile", 1.0, 1.0)
        assert body.name == "Étoile"
    
    def test_empty_name(self):
        """Test Body with empty name."""
        body = Body("", 1.0, 1.0)
        assert body.name == ""


class TestBodyEquality:
    """Test Body equality and comparison operations."""
    
    def test_body_identity(self):
        """Test that two Body instances with same parameters are different objects."""
        body1 = Body("Earth", 5.972e24, 6.371e6)
        body2 = Body("Earth", 5.972e24, 6.371e6)
        
        # They should be different objects
        assert body1 is not body2
    
    def test_body_parent_reference(self):
        """Test that parent references are maintained correctly."""
        earth = Body("Earth", 5.972e24, 6.371e6)
        moon1 = Body("Moon", 7.342e22, 1.737e6, parent=earth)
        moon2 = Body("Moon", 7.342e22, 1.737e6, parent=earth)
        
        # Both moons should reference the same Earth
        assert moon1.parent is earth
        assert moon2.parent is earth
        assert moon1.parent is moon2.parent


class TestBodyIntegration:
    """Integration tests for Body class."""
    
    def test_solar_system_hierarchy(self):
        """Test creating a simple solar system hierarchy."""
        sun = Body("Sun", 1.989e30, 6.96e8, color="#FFFF00")
        earth = Body("Earth", 5.972e24, 6.371e6, color="#0000FF", parent=sun)
        moon = Body("Moon", 7.342e22, 1.737e6, color="#808080", parent=earth)
        mars = Body("Mars", 6.39e23, 3.389e6, color="#FF0000", parent=sun)
        
        # Verify hierarchy
        assert sun.parent is sun
        assert earth.parent is sun
        assert moon.parent is earth
        assert mars.parent is sun
        
        # Verify properties are accessible
        assert sun.mass > earth.mass > moon.mass
        assert earth.mass > mars.mass > moon.mass
    
    def test_multiple_bodies_serialization(self):
        """Test serializing multiple bodies."""
        earth = Body("Earth", 5.972e24, 6.371e6, color="#0000FF")
        moon = Body("Moon", 7.342e22, 1.737e6, color="#808080", parent=earth)
        
        temp_earth = Path("temp_earth.pkl")
        temp_moon = Path("temp_moon.pkl")
        
        try:
            # Save both bodies
            earth.save(str(temp_earth))
            moon.save(str(temp_moon))
            
            # Load both bodies
            earth_loaded = Body.load(str(temp_earth))
            moon_loaded = Body.load(str(temp_moon))
            
            # Verify
            assert earth_loaded.name == "Earth"
            assert moon_loaded.name == "Moon"
            assert earth_loaded.mass == earth.mass
            assert moon_loaded.mass == moon.mass
            
        finally:
            # Clean up
            if temp_earth.exists():
                temp_earth.unlink()
            if temp_moon.exists():
                temp_moon.unlink()
    
    def test_body_properties_are_consistent(self):
        """Test that Body properties are consistent across accesses."""
        body = Body("Test", 1.0e24, 1.0e6, color="#123456")
        
        # Multiple accesses should return the same values
        assert body.name == body.name
        assert body.mass == body.mass
        assert body.radius == body.radius
        assert body.color == body.color
        assert body.parent is body.parent
