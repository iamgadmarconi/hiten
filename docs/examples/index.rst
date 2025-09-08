Examples
========

This section contains detailed examples demonstrating various features of HITEN.

Basic Examples
--------------

.. toctree::
   :maxdepth: 1

   periodic_orbits
   center_manifold
   invariant_tori
   orbit_family

Advanced Examples
-----------------

.. toctree::
   :maxdepth: 1

   heteroclinic_connection
   orbit_manifold
   poincare_map
   synodic_map

Running Examples
----------------

All examples can be run from the command line:

.. code-block:: bash

   # Run a specific example
   python examples/periodic_orbits.py
   
   # Run all examples
   python -m examples

Note: Some examples may require additional data files or take significant 
computation time to complete.

Example Data
------------

Some examples use pre-computed data files that are included in the repository.
These files are located in the `results/` directory and contain:

- Periodic orbit families
- Invariant manifold data
- Poincare map results
- Bifurcation diagrams

If you need to regenerate this data, see the individual example files for 
instructions on how to compute the required data.
