Algorithms Module
=================

The algorithms module contains the core computational algorithms for dynamical 
systems analysis in the CR3BP.

Dynamics
--------

The dynamics module provides a comprehensive framework for defining, analyzing, and
integrating dynamical systems with emphasis on applications in astrodynamics.

.. currentmodule:: hiten.algorithms.dynamics

Core Framework
~~~~~~~~~~~~~~

The core framework provides the base classes for dynamical systems and their interfaces. 
An abstract base class for dynamical systems is provided, as well as a protocol for dynamical systems.

Abstract Base Class
~~~~~~~~~~~~~~~~~~~

The :class:`_DynamicalSystem` class provides the foundation for implementing
concrete dynamical systems that can be integrated numerically. It defines the
minimal interface required for compatibility with the integrator framework
and includes utilities for state validation and dimension checking.

.. autoclass:: _DynamicalSystem()
   :members:
   :undoc-members:
   :exclude-members: __init__

The :class:`_DirectedSystem` class provides a directional wrapper for forward/backward time integration.

.. autoclass:: _DirectedSystem()
   :members:
   :undoc-members:
   :exclude-members: __init__

The :class:`_DynamicalSystemProtocol` class provides the protocol for the minimal interface for dynamical systems.

.. autoclass:: _DynamicalSystemProtocol()
   :members:
   :undoc-members:
   :exclude-members: __init__

The :func:`_propagate_dynsys` function provides a generic propagation function for dynamical systems.

.. autofunction:: _propagate_dynsys()

The :func:`_validate_initial_state` function provides a function to validate the initial state of a dynamical system.

.. autofunction:: _validate_initial_state()

Integrators
-----------

The integrators module provides numerical integration methods for dynamical systems.

.. currentmodule:: hiten.algorithms.integrators

Continuation
------------

The continuation module provides methods for numerical continuation of solutions.

.. currentmodule:: hiten.algorithms.continuation

Corrector
---------

The corrector module provides methods for correcting approximate solutions.

.. currentmodule:: hiten.algorithms.corrector

Bifurcation
-----------

The bifurcation module provides methods for detecting and analyzing bifurcations.

.. currentmodule:: hiten.algorithms.bifurcation

Connections
-----------

The connections module provides functionality for discovering ballistic and impulsive 
transfers between manifolds in the CR3BP.

.. currentmodule:: hiten.algorithms.connections

Fourier
-------

The Fourier module provides methods for Fourier analysis of periodic solutions.

.. currentmodule:: hiten.algorithms.fourier

Hamiltonian
-----------

The Hamiltonian module provides methods for Hamiltonian systems analysis.

.. currentmodule:: hiten.algorithms.hamiltonian

Poincare Maps
-------------

The Poincare module provides methods for Poincare maps and center manifolds.

.. currentmodule:: hiten.algorithms.poincare

Polynomial
----------

The polynomial module provides methods for polynomial operations in Hamiltonian systems.

.. currentmodule:: hiten.algorithms.polynomial

Tori
----

The tori module provides methods for invariant tori analysis.

.. currentmodule:: hiten.algorithms.tori

Utilities
---------

The utils module provides utility functions for algorithms.

.. currentmodule:: hiten.algorithms.utils
