Orbits Module
=============

The orbits module provides high-level utilities for representing and analyzing periodic orbits in the circular restricted three-body problem.

This module provides the fundamental classes for representing different types of periodic orbits, including halo orbits, Lyapunov orbits, and vertical orbits.

.. currentmodule:: hiten.system.orbits

Base Classes
~~~~~~~~~~~~

The base module provides the core periodic orbit framework.

.. currentmodule:: hiten.system.orbits.base

PeriodicOrbit()
^^^^^^^^^^^^^^^

Base class for periodic orbits.

.. autoclass:: PeriodicOrbit()
   :members:
   :undoc-members:
   :exclude-members: __init__

Halo Orbits
~~~~~~~~~~~

The halo module provides halo orbit classes.

.. currentmodule:: hiten.system.orbits.halo

HaloOrbit()
^^^^^^^^^^^

Halo orbit representation.

.. autoclass:: HaloOrbit()
   :members:
   :undoc-members:
   :exclude-members: __init__

Lyapunov Orbits
~~~~~~~~~~~~~~~

The lyapunov module provides Lyapunov orbit classes.

.. currentmodule:: hiten.system.orbits.lyapunov

LyapunovOrbit()
^^^^^^^^^^^^^^^

Lyapunov orbit representation.

.. autoclass:: LyapunovOrbit()
   :members:
   :undoc-members:
   :exclude-members: __init__

Vertical Orbits
~~~~~~~~~~~~~~~

The vertical module provides vertical orbit classes.

.. currentmodule:: hiten.system.orbits.vertical

VerticalOrbit()
^^^^^^^^^^^^^^^

Vertical orbit representation.

.. autoclass:: VerticalOrbit()
   :members:
   :undoc-members:
   :exclude-members: __init__
