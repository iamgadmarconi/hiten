Body Module
===========

The body module provides light-weight representation of a celestial body participating in a circular restricted three body problem (CR3BP) or standalone dynamical simulation.

The module defines the :class:`~hiten.system.body.Body` class, a minimal container that stores basic physical quantities and plotting attributes while preserving the hierarchical relation to a central body through the :attr:`~hiten.system.body.Body.parent` attribute.

.. currentmodule:: hiten.system.body

Body
----

Celestial body container for CR3BP systems.

.. autoclass:: Body
   :members:
   :undoc-members:
   :exclude-members: __init__