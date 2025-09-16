Constants Module
================

The constants module provides physical constants and celestial body data for astrodynamics simulations.

This module contains fundamental physical constants and system-specific values for use in astrodynamics simulations. All constants are defined in SI units and stored as numpy float64 data types for precision and consistency in numerical computations.

The module includes:
- Universal physical constants (gravitational constant)
- Planetary and lunar masses
- Characteristic distances for common systems
- Body radii

These constants provide the foundation for various dynamical calculations including the Circular Restricted Three-Body Problem (CR3BP) and orbital mechanics problems.

.. currentmodule:: hiten.utils.constants

Constants
---------

Class containing physical constants for astrodynamics simulations.

.. autoclass:: Constants
   :members:
   :undoc-members:
   :exclude-members: __init__
