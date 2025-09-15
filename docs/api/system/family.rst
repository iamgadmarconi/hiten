Family Module
=============

The family module provides light-weight container that groups a family of periodic orbits obtained via a continuation engine.

It offers convenience helpers for iteration, random access, conversion to a pandas.DataFrame, and basic serialisation to an HDF5 file leveraging the existing utilities in :mod:`~hiten.utils.io`.

.. currentmodule:: hiten.system.family

OrbitFamily
-----------

Container for an ordered family of periodic orbits.

.. autoclass:: OrbitFamily
   :members:
   :undoc-members:
   :exclude-members: __init__