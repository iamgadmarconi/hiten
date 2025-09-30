Algorithms Utilities Module
===========================

The utils module provides utility functions for algorithms in the circular restricted three-body problem.

.. currentmodule:: hiten.algorithms.utils

Configuration
~~~~~~~~~~~~~

The config module provides configuration utilities.

.. currentmodule:: hiten.algorithms.utils.config

Configuration constants and settings for the algorithms module.

Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

The coordinates module provides comprehensive coordinate transformation functions for the circular restricted three-body problem.

.. currentmodule:: hiten.algorithms.utils.coordinates

_rotating_to_inertial()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_rotating_to_inertial` function converts state from rotating to inertial frame.

.. autofunction:: _rotating_to_inertial()

_inertial_to_rotating()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_inertial_to_rotating` function converts state from inertial to rotating frame.

.. autofunction:: _inertial_to_rotating()

_get_mass_parameter()
^^^^^^^^^^^^^^^^^^^^^

The :func:`_get_mass_parameter` function calculates the mass parameter mu for the CR3BP.

.. autofunction:: _get_mass_parameter()

_get_angular_velocity()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_get_angular_velocity` function calculates the mean motion (angular velocity) of the CR3BP.

.. autofunction:: _get_angular_velocity()

_to_crtbp_units()
^^^^^^^^^^^^^^^^^

The :func:`_to_crtbp_units` function converts an SI-state vector into the dimensionless state used by crtbp_accel.

.. autofunction:: _to_crtbp_units()

_to_si_units()
^^^^^^^^^^^^^^

The :func:`_to_si_units` function converts a dimensionless state vector into the SI-state vector.

.. autofunction:: _to_si_units()

_dimless_time()
^^^^^^^^^^^^^^^

The :func:`_dimless_time` function converts time from SI units (seconds) to dimensionless CR3BP time units.

.. autofunction:: _dimless_time()

_si_time()
^^^^^^^^^^

The :func:`_si_time` function converts time from dimensionless CR3BP time units to SI units (seconds).

.. autofunction:: _si_time()

_velocity_scale_si_per_canonical()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_velocity_scale_si_per_canonical` function returns the scale factor to convert canonical CRTBP velocities to SI (m/s).

.. autofunction:: _velocity_scale_si_per_canonical()

_get_distance()
^^^^^^^^^^^^^^^

The :func:`_get_distance` function calculates physical distance between two bodies in meters.

.. autofunction:: _get_distance()

