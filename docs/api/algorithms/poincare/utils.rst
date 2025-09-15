Poincare Map Utilities
========================

The utils module provides utility functions for interpolation and numerical methods used in Poincare map computation. These functions are optimized for high-performance numerical operations and are used throughout the Poincare module for trajectory interpolation and section crossing refinement.

.. currentmodule:: hiten.algorithms.poincare.utils

.. autofunction:: _interp_linear()

Linear interpolation function for trajectory segments. Efficiently interpolates between two states at given times using linear approximation.

.. autofunction:: _hermite_scalar()

Scalar Hermite interpolation function. Computes high-order interpolation using function values and derivatives for improved accuracy in section crossing detection.

.. autofunction:: _hermite_der()

Hermite interpolation derivative function. Computes the derivative of Hermite interpolated values, useful for velocity and acceleration calculations in trajectory refinement.
