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

High-Precision Arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~

The precision module provides comprehensive high-precision arithmetic capabilities for numerical computations.

.. currentmodule:: hiten.algorithms.utils.precision

_Number()
^^^^^^^^^

The :class:`_Number` class provides a number class that supports high-precision arithmetic operations.

.. autoclass:: _Number()
   :members:
   :undoc-members:
   :exclude-members: __init__

hp()
^^^^

The :func:`hp` function creates a high-precision number instance.

.. autofunction:: hp()

with_precision()
^^^^^^^^^^^^^^^^

The :func:`with_precision` function provides a context manager for setting mpmath precision.

.. autofunction:: with_precision()

divide()
^^^^^^^^

The :func:`divide` function performs high precision division if enabled, otherwise standard division.

.. autofunction:: divide()

sqrt()
^^^^^^

The :func:`sqrt` function computes square root with high precision if enabled.

.. autofunction:: sqrt()

power()
^^^^^^^

The :func:`power` function computes power with high precision if enabled.

.. autofunction:: power()

find_root()
^^^^^^^^^^^

The :func:`find_root` function finds root with high precision using mpmath.

.. autofunction:: find_root()

Type Definitions
~~~~~~~~~~~~~~~~

The types module provides comprehensive type definitions and state vector containers for the CR3BP.

.. currentmodule:: hiten.algorithms.utils.states

SynodicState()
^^^^^^^^^^^^^^

The :class:`SynodicState` class provides enumeration for synodic frame coordinates.

.. autoclass:: SynodicState()
   :members:
   :undoc-members:
   :exclude-members: __init__

CenterManifoldState()
^^^^^^^^^^^^^^^^^^^^^

The :class:`CenterManifoldState` class provides enumeration for center manifold coordinates.

.. autoclass:: CenterManifoldState()
   :members:
   :undoc-members:
   :exclude-members: __init__

RestrictedCenterManifoldState()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`RestrictedCenterManifoldState` class provides enumeration for restricted center manifold coordinates.

.. autoclass:: RestrictedCenterManifoldState()
   :members:
   :undoc-members:
   :exclude-members: __init__

_BaseStateContainer()
^^^^^^^^^^^^^^^^^^^^^

The :class:`_BaseStateContainer` class provides minimal mutable container for a single state vector, indexed by an IntEnum.

.. autoclass:: _BaseStateContainer()
   :members:
   :undoc-members:
   :exclude-members: __init__

SynodicStateVector()
^^^^^^^^^^^^^^^^^^^^

The :class:`SynodicStateVector` class provides container for synodic frame state vectors.

.. autoclass:: SynodicStateVector()
   :members:
   :undoc-members:
   :exclude-members: __init__

CenterManifoldStateVector()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`CenterManifoldStateVector` class provides container for center manifold state vectors.

.. autoclass:: CenterManifoldStateVector()
   :members:
   :undoc-members:
   :exclude-members: __init__

RestrictedCenterManifoldStateVector()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`RestrictedCenterManifoldStateVector` class provides container for restricted center manifold state vectors.

.. autoclass:: RestrictedCenterManifoldStateVector()
   :members:
   :undoc-members:
   :exclude-members: __init__

Trajectory()
^^^^^^^^^^^^

The :class:`Trajectory` class provides lightweight container for trajectory data: a time array and matching state vectors.

.. autoclass:: Trajectory()
   :members:
   :undoc-members:
   :exclude-members: __init__

