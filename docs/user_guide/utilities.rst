Utilities Guide
===============

This guide explains the utility functions and classes available in HITEN.

Constants
---------

The :class:`hiten.utils.constants.Constants` class defines the physical 
parameters of the Circular Restricted Three-Body Problem (CR3BP).

Creating Constants
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import Constants
   
   # Earth-Moon system
   constants = Constants(mu=0.012150585609624)
   
   # Sun-Earth system
   constants = Constants(mu=3.0404e-6)

Mass Parameter
~~~~~~~~~~~~~~

The mass parameter :math:`\mu` is defined as:

.. math::

   \mu = \frac{m_2}{m_1 + m_2}

where :math:`m_1` is the mass of the primary body and :math:`m_2` is the mass 
of the secondary body.

Common Values
~~~~~~~~~~~~~

- Earth-Moon: :math:`\mu = 0.012150585609624`
- Sun-Earth: :math:`\mu = 3.0404 \times 10^{-6}`
- Sun-Jupiter: :math:`\mu = 9.537 \times 10^{-4}`

Lagrange Points
~~~~~~~~~~~~~~~

The Lagrange points are automatically computed when creating a Constants object:

.. code-block:: python

   constants = Constants(mu=0.012150585609624)
   
   print(f"L1 position: {constants.l1_position}")
   print(f"L2 position: {constants.l2_position}")
   print(f"L3 position: {constants.l3_position}")
   print(f"L4 position: {constants.l4_position}")
   print(f"L5 position: {constants.l5_position}")

API Reference
~~~~~~~~~~~~~

.. automodule:: hiten.utils.constants
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
--------

Visualization tools for dynamical systems analysis.

.. automodule:: hiten.utils.plots
   :members:
   :undoc-members:
   :show-inheritance:

Printing
--------

Formatted output utilities.

.. automodule:: hiten.utils.printing
   :members:
   :undoc-members:
   :show-inheritance:

Testing
-------

Testing utilities and helpers.

.. automodule:: hiten.utils.tests
   :members:
   :undoc-members:
   :show-inheritance:

Input/Output
------------

File I/O utilities for saving and loading data.

.. automodule:: hiten.utils.io
   :members:
   :undoc-members:
   :show-inheritance:

Logging
-------

Logging configuration and utilities.

.. automodule:: hiten.utils.log_config
   :members:
   :undoc-members:
   :show-inheritance:
