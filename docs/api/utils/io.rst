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

Hamiltonian I/O
---------------

The hamiltonian module provides I/O utilities for Hamiltonian data.

.. currentmodule:: hiten.utils.io.hamiltonian

.. autofunction:: save_hamiltonian

.. autofunction:: load_hamiltonian

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

