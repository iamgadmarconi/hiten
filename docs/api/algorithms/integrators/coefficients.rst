Integrators Coefficients
=========================

The coefficients module provides Butcher tableaux for various Runge-Kutta methods.

.. currentmodule:: hiten.algorithms.integrators.coefficients

Coefficient Arrays
------------------

The module provides coefficient arrays for different Runge-Kutta methods:

- RK4: Classical 4th-order Runge-Kutta method
- RK6: 6th-order Runge-Kutta method  
- RK8: 8th-order Runge-Kutta method
- RK45: Dormand-Prince 5(4) adaptive method
- DOP853: Dormand-Prince 8(5,3) adaptive method

These arrays are used internally by the Runge-Kutta integrator classes.
