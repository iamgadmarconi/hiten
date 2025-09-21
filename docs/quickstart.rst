Quick Start Guide
==================

This guide will help you get started with HITEN by walking through some basic examples.

Basic Setup
-----------

First, import the necessary modules:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from hiten import Constants, CenterManifold, HaloOrbit

Setting Up a System
-------------------

HITEN works with the Circular Restricted Three-Body Problem (CR3BP). Let's set up the Earth-Moon system:

.. code-block:: python

   from hiten import System
   system = System.from_bodies("earth", "moon")

   print(f"Mass parameter: {system.mu}")
   print(f"L1 position: {system.l1_position}")
   print(f"L2 position: {system.l2_position}")

Working with Periodic Orbits
----------------------------

Let's compute a simple periodic orbit near the L1 Lagrange point:

.. code-block:: python

   l1 = system.get_libration_point(1)
   orbit = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   orbit.correct(max_attempts=25)
   orbit.propagate(steps=1000)

   print(f"Period: {orbit.period}")
   print(f"Initial state: {orbit.initial_state}")
   orbit.plot()

Next Steps
----------

Now that you have a basic understanding of HITEN, you can:

1. Explore the :doc:`examples/index` for more detailed examples
2. Read the :doc:`user_guide/guide_01_systems`
3. Check the :doc:`api/system` for detailed API documentation

For more examples, see the `examples/` directory in the source code repository.

Troubleshooting
---------------

If you encounter issues:

1. Check that all dependencies are installed correctly
2. Verify your Python version (3.9+ required)
3. Check the `GitHub Issues <https://github.com/iamgadmarconi/hiten/issues>`_ for known problems
4. Create a new issue if you find a bug

For more help, see the :doc:`installation` guide or the full :doc:`api/system` documentation.
