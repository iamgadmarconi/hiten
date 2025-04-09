"""
Physical constants for astrodynamics simulations.

This module contains fundamental physical constants and system-specific values
for use in astrodynamics simulations. All constants are defined in SI units
and stored as numpy float64 data types for precision and consistency in 
numerical computations.

The module includes:
1. Universal physical constants (gravitational constant)
2. Planetary and lunar masses
3. Characteristic distances for common systems
4. Body radii

These constants provide the foundation for various dynamical calculations
including the Circular Restricted Three-Body Problem (CR3BP) and orbital
mechanics problems.

References
----------
Values are based on standard IAU (International Astronomical Union) and
NASA/JPL data. For detailed sources, see:
- IAU 2015 Resolution B3 (https://www.iau.org/static/resolutions/IAU2015_English.pdf)
- NASA JPL Solar System Dynamics (https://ssd.jpl.nasa.gov/)
"""

import numpy as np
from typing import Dict, Any


class Constants:
    """Class containing physical constants for astrodynamics simulations."""
    
    # Universal physical constants
    G: float = np.float64(6.67430e-11)  # Universal gravitational constant (m^3 kg^-1 s^-2)
    
    # Celestial body properties organized by body
    bodies: Dict[str, Dict[str, Any]] = {
        "sun": {
            "mass": np.float64(1.989e30),  # kg
            "radius": np.float64(696340e3),  # m
        },
        "mercury": {
            "mass": np.float64(3.302e23),  # kg
            "radius": np.float64(2439.7e3),  # m
        },
        "venus": {
            "mass": np.float64(4.867e24),  # kg
            "radius": np.float64(6051.8e3),  # m
        },
        "earth": {
            "mass": np.float64(5.972e24),  # kg
            "radius": np.float64(6378.137e3),  # m
        },
        "moon": {
            "mass": np.float64(7.348e22),  # kg
            "radius": np.float64(1737.4e3),  # m
        },
        "mars": {
            "mass": np.float64(6.417e23),  # kg
            "radius": np.float64(3396.2e3),  # m
        },
        "phobos": {
            "mass": np.float64(1.072e16),  # kg
            "radius": np.float64(11.269e3),  # m
        },
        "deimos": {
            "mass": np.float64(1.476e15),  # kg
            "radius": np.float64(6.2e3),  # m
        },
        "jupiter": {
            "mass": np.float64(1.898e27),  # kg
            "radius": np.float64(69911e3),  # m
        },
        "io": {
            "mass": np.float64(8.932e22),  # kg
            "radius": np.float64(1821.6e3),  # m
        },
        "europa": {
            "mass": np.float64(4.8e22),  # kg
            "radius": np.float64(1560.8e3),  # m
        },
        "ganymede": {
            "mass": np.float64(1.482e23),  # kg
            "radius": np.float64(2631.2e3),  # m
        },
        "callisto": {
            "mass": np.float64(1.076e23),  # kg
            "radius": np.float64(2410.3e3),  # m
        },
        "saturn": {
            "mass": np.float64(5.683e26),  # kg
            "radius": np.float64(58232e3),  # m
        },
        "titan": {
            "mass": np.float64(1.345e23),  # kg
            "radius": np.float64(2574.7e3),  # m
        },
        "uranus": {
            "mass": np.float64(8.681e25),  # kg
            "radius": np.float64(25362e3),  # m
        },
        "neptune": {
            "mass": np.float64(1.024e26),  # kg
            "radius": np.float64(24622e3),  # m
        },
        "triton": {
            "mass": np.float64(2.14e22),  # kg
            "radius": np.float64(1737.4e3),  # m
        },
        "pluto": {
            "mass": np.float64(1.0887e22),  # kg
            "radius": np.float64(1188.3e3),  # m
        }
    }
    
    # Orbital distances organized by system
    orbital_distances: Dict[str, Dict[str, float]] = {
        "sun": {
            "mercury": np.float64(57.91e9),  # m
            "venus": np.float64(108.2e9),  # m
            "earth": np.float64(149.6e9),  # m
            "mars": np.float64(227.9e9),  # m
            "jupiter": np.float64(778.5e9),  # m
            "saturn": np.float64(1426.7e9),  # m
            "uranus": np.float64(2870.97e9),  # m
            "neptune": np.float64(4498.25e9),  # m
            "pluto": np.float64(5906.38e9),  # m
        },
        "earth": {
            "moon": np.float64(384400e3),  # m
        },
        "mars": {
            "phobos": np.float64(9248e3),  # m
            "deimos": np.float64(23460e3),  # m
        },
        "jupiter": {
            "io": np.float64(421700e3),  # m
            "europa": np.float64(671100e3),  # m
            "ganymede": np.float64(1070400e3),  # m
            "callisto": np.float64(1882700e3),  # m
        },
        "saturn": {
            "titan": np.float64(1221870e3),  # m
        },
        "neptune": {
            "triton": np.float64(354759e3),  # m
        }
    }
    
    @classmethod
    def get_mass(cls, body: str) -> float:
        """Get the mass of a celestial body.
        
        Parameters
        ----------
        body : str
            Name of the celestial body
            
        Returns
        -------
        float
            Mass in kg
        """
        return cls.bodies[body.lower()]["mass"]
    
    @classmethod
    def get_radius(cls, body: str) -> float:
        """Get the radius of a celestial body.
        
        Parameters
        ----------
        body : str
            Name of the celestial body
            
        Returns
        -------
        float
            Radius in m
        """
        return cls.bodies[body.lower()]["radius"]
    
    @classmethod
    def get_orbital_distance(cls, primary: str, secondary: str) -> float:
        """Get the orbital distance between a primary and secondary body.
        
        Parameters
        ----------
        primary : str
            Name of the primary body (e.g., "sun", "earth")
        secondary : str
            Name of the secondary body (e.g., "earth", "moon")
            
        Returns
        -------
        float
            Orbital distance (semi-major axis) in m
        """
        return cls.orbital_distances[primary.lower()][secondary.lower()]


# For backward compatibility, define the old variable names
# Users can migrate to the new class-based approach at their own pace

# Universal physical constants
G = Constants.G

# Masses
M_sun = Constants.get_mass("sun")
M_mercury = Constants.get_mass("mercury")
M_venus = Constants.get_mass("venus")
M_earth = Constants.get_mass("earth")
M_moon = Constants.get_mass("moon")
M_mars = Constants.get_mass("mars")
M_phobos = Constants.get_mass("phobos")
M_deimos = Constants.get_mass("deimos")
M_jupiter = Constants.get_mass("jupiter")
M_io = Constants.get_mass("io")
M_europa = Constants.get_mass("europa")
M_ganymede = Constants.get_mass("ganymede")
M_callisto = Constants.get_mass("callisto")
M_saturn = Constants.get_mass("saturn")
M_titan = Constants.get_mass("titan")
M_uranus = Constants.get_mass("uranus")
M_neptune = Constants.get_mass("neptune")
M_triton = Constants.get_mass("triton")
M_pluto = Constants.get_mass("pluto")

# Characteristic distances
R_sun_mercury = Constants.get_orbital_distance("sun", "mercury")
R_sun_venus = Constants.get_orbital_distance("sun", "venus")
R_earth_sun = Constants.get_orbital_distance("sun", "earth")
R_earth_moon = Constants.get_orbital_distance("earth", "moon")
R_sun_mars = Constants.get_orbital_distance("sun", "mars")
R_mars_phobos = Constants.get_orbital_distance("mars", "phobos")
R_mars_deimos = Constants.get_orbital_distance("mars", "deimos")
R_sun_jupiter = Constants.get_orbital_distance("sun", "jupiter")
R_jupiter_io = Constants.get_orbital_distance("jupiter", "io")
R_jupiter_europa = Constants.get_orbital_distance("jupiter", "europa")
R_jupiter_ganymede = Constants.get_orbital_distance("jupiter", "ganymede")
R_jupiter_callisto = Constants.get_orbital_distance("jupiter", "callisto")
R_sun_saturn = Constants.get_orbital_distance("sun", "saturn")
R_saturn_titan = Constants.get_orbital_distance("saturn", "titan")
R_sun_uranus = Constants.get_orbital_distance("sun", "uranus")
R_sun_neptune = Constants.get_orbital_distance("sun", "neptune")
R_neptune_triton = Constants.get_orbital_distance("neptune", "triton")
R_sun_pluto = Constants.get_orbital_distance("sun", "pluto")

# Body radii
R_sun = Constants.get_radius("sun")
R_mercury = Constants.get_radius("mercury")
R_venus = Constants.get_radius("venus")
R_earth = Constants.get_radius("earth")
R_moon = Constants.get_radius("moon")
R_mars = Constants.get_radius("mars")
R_phobos = Constants.get_radius("phobos")
R_deimos = Constants.get_radius("deimos")
R_jupiter = Constants.get_radius("jupiter")
R_io = Constants.get_radius("io")
R_europa = Constants.get_radius("europa")
R_ganymede = Constants.get_radius("ganymede")
R_callisto = Constants.get_radius("callisto")
R_saturn = Constants.get_radius("saturn")
R_titan = Constants.get_radius("titan")
R_uranus = Constants.get_radius("uranus")
R_neptune = Constants.get_radius("neptune")
R_triton = Constants.get_radius("triton")
R_pluto = Constants.get_radius("pluto")







