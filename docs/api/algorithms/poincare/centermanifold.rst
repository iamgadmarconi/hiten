Center Manifold Backend
=======================

The backend module provides Numba-compiled kernels for computing center manifold trajectories in the CR3BP.

.. currentmodule:: hiten.algorithms.poincare.centermanifold.backend

.. autoclass:: _CenterManifoldBackend()
   :members:
   :exclude-members: __init__

Backend for center manifold computations in the CR3BP. Uses Numba-compiled kernels for efficient Hamiltonian integration and Poincare map evaluation.

.. autofunction:: _detect_crossing()

Detect if trajectory crossed the Poincare section using Hermite interpolation.

.. autofunction:: _solve_bracketed()

Pure-Python bracketed bisection solver for general callables.

.. autofunction:: _get_rk_coefficients()

Return Runge-Kutta coefficients for specified order.

.. autofunction:: _integrate_rk_ham()

Integrate Hamiltonian system using Runge-Kutta method.

.. autofunction:: _integrate_map()

Integrate Hamiltonian system using specified method (Runge-Kutta or symplectic).

.. autofunction:: _poincare_step()

Perform one Poincare map step for center manifold integration.

.. autofunction:: _poincare_map()

Compute Poincare map for multiple center manifold seeds in parallel.
