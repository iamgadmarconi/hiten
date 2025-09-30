Input/Output Module
===================

The io module provides comprehensive serialization and file I/O utilities for the hiten package.

This module provides a complete set of utilities for serializing and deserializing various hiten objects to/from HDF5 files. It includes specialized I/O functions for center manifolds, Hamiltonians, manifolds, Poincare maps, and periodic orbits. All data is stored in HDF5 format with version tracking and supports compression for efficient storage.

.. currentmodule:: hiten.utils.io

Common I/O Utilities
--------------------

The common module provides shared utilities for file and directory operations.

.. currentmodule:: hiten.utils.io.common

.. autofunction:: _ensure_dir

.. autofunction:: _write_dataset

Center Manifold I/O
-------------------

The center module provides I/O utilities for center manifold data.

.. currentmodule:: hiten.utils.io.center

.. autofunction:: save_center_manifold

.. autofunction:: load_center_manifold

.. autofunction:: load_center_manifold_inplace

Hamiltonian I/O
---------------

The hamiltonian module provides I/O utilities for Hamiltonian data.

.. currentmodule:: hiten.utils.io.hamiltonian

.. autofunction:: save_hamiltonian

.. autofunction:: load_hamiltonian

.. autofunction:: save_lie_generating_function

.. autofunction:: load_lie_generating_function

.. autofunction:: load_hamiltonian_inplace

.. autofunction:: load_lie_generating_function_inplace

Manifold I/O
------------

The manifold module provides I/O utilities for manifold data.

.. currentmodule:: hiten.utils.io.manifold

.. autofunction:: save_manifold

.. autofunction:: load_manifold

Poincare Map I/O
----------------

The map module provides I/O utilities for Poincare map data.

.. currentmodule:: hiten.utils.io.map

.. autofunction:: save_poincare_map

.. autofunction:: load_poincare_map_inplace

.. autofunction:: load_poincare_map

Periodic Orbit I/O
------------------

The orbits module provides I/O utilities for periodic orbit data.

.. currentmodule:: hiten.utils.io.orbits

.. autofunction:: save_periodic_orbit

.. autofunction:: load_periodic_orbit_inplace

.. autofunction:: load_periodic_orbit

.. autofunction:: register_orbit_class

Body I/O
--------

The body module provides I/O utilities for body data.

.. currentmodule:: hiten.utils.io.body

.. autofunction:: save_body

.. autofunction:: load_body

.. autofunction:: load_body_inplace

Family I/O
----------

The family module provides I/O utilities for orbit family data.

.. currentmodule:: hiten.utils.io.family

.. autofunction:: save_family

.. autofunction:: load_family

.. autofunction:: load_family_inplace

Libration Point I/O
--------------------

The libration module provides I/O utilities for libration point data.

.. currentmodule:: hiten.utils.io.libration

.. autofunction:: save_libration_point

.. autofunction:: load_libration_point

.. autofunction:: load_libration_point_inplace

System I/O
----------

The system module provides I/O utilities for system data.

.. currentmodule:: hiten.utils.io.system

.. autofunction:: save_system

.. autofunction:: load_system

.. autofunction:: load_system_inplace

Torus I/O
---------

The torus module provides I/O utilities for invariant torus data.

.. currentmodule:: hiten.utils.io.torus

.. autofunction:: save_torus

.. autofunction:: load_torus

.. autofunction:: load_torus_inplace

