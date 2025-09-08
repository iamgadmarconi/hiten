"""Fourier analysis algorithms for periodic solutions in the CR3BP.

This module provides comprehensive tools for Fourier analysis of periodic
solutions in the Circular Restricted Three-Body Problem (CR3BP). It includes
algorithms for computing Fourier coefficients, evaluating Fourier series,
and performing spectral analysis of periodic orbits and invariant manifolds.

The module supports both symbolic and numerical Fourier analysis with
efficient implementations using numba acceleration for performance-critical
computations.

Examples
--------
Basic Fourier analysis of a periodic orbit:

>>> from hiten.algorithms.fourier import fourier_analyze
>>> from hiten.system import HaloOrbit
>>> 
>>> # Create a periodic orbit
>>> orbit = HaloOrbit(constants)
>>> 
>>> # Perform Fourier analysis
>>> coeffs = fourier_analyze(orbit.trajectory, orbit.period)

See Also
--------
:mod:`~hiten.algorithms.polynomial`
    Polynomial operations used in Fourier analysis
:mod:`~hiten.system.orbits`
    Periodic orbit classes that can be analyzed
"""
