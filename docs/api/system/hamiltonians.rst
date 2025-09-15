Hamiltonians Module
===================

The hamiltonians module provides base classes for Hamiltonian representations in the CR3BP and a transformation pipeline for different Hamiltonian representations.

This module provides the fundamental classes for representing and manipulating Hamiltonian functions in the circular restricted three-body problem. It includes the base Hamiltonian class and Lie generating function class for canonical transformations.

.. currentmodule:: hiten.system.hamiltonians

Base Classes
~~~~~~~~~~~~

The base module provides the core Hamiltonian framework.

.. currentmodule:: hiten.system.hamiltonians.base

Hamiltonian()
^^^^^^^^^^^^^

Abstract container for a specific polynomial Hamiltonian representation.

.. autoclass:: Hamiltonian()
   :members:
   :undoc-members:
   :exclude-members: __init__

LieGeneratingFunction()
^^^^^^^^^^^^^^^^^^^^^^^

Class for Lie generating functions in canonical transformations.

.. autoclass:: LieGeneratingFunction()
   :members:
   :undoc-members:
   :exclude-members: __init__

Pipeline Classes
~~~~~~~~~~~~~~~~

The pipeline module provides Hamiltonian transformation pipeline classes.

.. currentmodule:: hiten.system.hamiltonians.pipeline

HamiltonianPipeline()
^^^^^^^^^^^^^^^^^^^^^

Manages the transformation pipeline for Hamiltonian representations.

.. autoclass:: HamiltonianPipeline()
   :members:
   :undoc-members:
   :exclude-members: __init__
