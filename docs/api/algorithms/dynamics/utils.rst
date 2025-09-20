Dynamics Utilities
==================

The utils module provides utility functions for dynamical systems analysis.

.. currentmodule:: hiten.algorithms.dynamics.utils

Energy Functions
^^^^^^^^^^^^^^^^

The energy module provides energy and potential functions for the CR3BP.

.. currentmodule:: hiten.algorithms.dynamics.utils.energy

_max_rel_energy_error()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_max_rel_energy_error` function computes maximum relative deviation of Jacobi constant along trajectory.

.. autofunction:: _max_rel_energy_error()

crtbp_energy()
^^^^^^^^^^^^^^

The :func:`crtbp_energy` function computes Hamiltonian energy of a state in the CR3BP.

.. autofunction:: crtbp_energy()

effective_potential()
^^^^^^^^^^^^^^^^^^^^^

The :func:`effective_potential` function computes effective potential in the CR3BP rotating frame.

.. autofunction:: effective_potential()

kinetic_energy()
^^^^^^^^^^^^^^^^

The :func:`kinetic_energy` function computes kinetic energy of a state.

.. autofunction:: kinetic_energy()

gravitational_potential()
^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`gravitational_potential` function computes gravitational potential energy of test particle.

.. autofunction:: gravitational_potential()

hill_region()
^^^^^^^^^^^^^

The :func:`hill_region` function computes Hill region for zero-velocity surface analysis.

.. autofunction:: hill_region()

energy_to_jacobi()
^^^^^^^^^^^^^^^^^^

The :func:`energy_to_jacobi` function converts Hamiltonian energy to Jacobi constant.

.. autofunction:: energy_to_jacobi()

jacobi_to_energy()
^^^^^^^^^^^^^^^^^^

The :func:`jacobi_to_energy` function converts Jacobi constant to Hamiltonian energy.

.. autofunction:: jacobi_to_energy()

primary_distance()
^^^^^^^^^^^^^^^^^^

The :func:`primary_distance` function computes distance from test particle to primary body.

.. autofunction:: primary_distance()

secondary_distance()
^^^^^^^^^^^^^^^^^^^^^

The :func:`secondary_distance` function computes distance from test particle to secondary body.

.. autofunction:: secondary_distance()

pseudo_potential_at_point()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`pseudo_potential_at_point` function evaluates pseudo-potential Omega at a planar point.

.. autofunction:: pseudo_potential_at_point()

Linear Algebra Functions
^^^^^^^^^^^^^^^^^^^^^^^^

The linalg module provides linear algebra utilities for dynamical systems analysis.

.. currentmodule:: hiten.algorithms.dynamics.utils.linalg

eigenvalue_decomposition()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`eigenvalue_decomposition` function classifies eigenvalue-eigenvector pairs into stable, unstable, and center subspaces.

.. autofunction:: eigenvalue_decomposition()

_stability_indices()
^^^^^^^^^^^^^^^^^^^^

The :func:`_stability_indices` function computes Floquet stability indices for periodic orbit analysis.

.. autofunction:: _stability_indices()

Other Utility Functions
^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: hiten.algorithms.dynamics.utils.linalg

_zero_small_imag_part()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_zero_small_imag_part` function removes small imaginary parts from complex numbers.

.. autofunction:: _zero_small_imag_part()

_remove_infinitesimals_in_place()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_remove_infinitesimals_in_place` function removes numerical noise from complex vector components in-place.

.. autofunction:: _remove_infinitesimals_in_place()

_remove_infinitesimals_array()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_remove_infinitesimals_array` function creates cleaned copy of vector with numerical noise removed.

.. autofunction:: _remove_infinitesimals_array()

_totime()
^^^^^^^^^

The :func:`_totime` function finds indices of closest time values in array.

.. autofunction:: _totime()
