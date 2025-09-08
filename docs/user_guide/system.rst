System Configuration
====================

This guide explains how to configure and work with dynamical systems in HITEN.

Body Objects
------------

The :class:`hiten.system.body.Body` class represents celestial bodies in the system.

Creating Bodies
~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import Body
   
   # Primary body (e.g., Earth)
   primary = Body(
       name="Earth",
       mass=5.972e24,  # kg
       position=[-mu, 0, 0],  # nondimensional
       velocity=[0, 0, 0]  # nondimensional
   )
   
   # Secondary body (e.g., Moon)
   secondary = Body(
       name="Moon",
       mass=7.342e22,  # kg
       position=[1-mu, 0, 0],  # nondimensional
       velocity=[0, 0, 0]  # nondimensional
   )

Center Objects
--------------

The :class:`hiten.system.center.CenterManifold` class represents center manifolds 
and their associated dynamics.

Working with Centers
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import CenterManifold
   
   # Create a center manifold
   center_manifold = CenterManifold(constants)
   
   # Set up for a specific libration point
   center_manifold.setup_libration_point(1)
   
   # Get the position
   position = center_manifold.position

Family Objects
--------------

The :class:`hiten.system.family.OrbitFamily` class represents families of periodic 
orbits or other parameterized solutions.

Creating Families
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import OrbitFamily
   
   # Create a family of periodic orbits
   family = OrbitFamily(constants, orbit_type="halo")
   
   # Add orbits to the family
   family.add_orbit(orbit1)
   family.add_orbit(orbit2)
   
   # Get all orbits
   orbits = family.orbits

Manifold Objects
----------------

The :class:`hiten.system.manifold.Manifold` class represents invariant manifolds 
of periodic orbits.

Creating Manifolds
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import Manifold
   
   # Create a manifold
   manifold = Manifold(constants)
   
   # Set up for a specific orbit
   manifold.setup_orbit(periodic_orbit)
   
   # Compute the manifold
   result = manifold.compute()

Orbit Objects
-------------

The :class:`hiten.system.orbits.PeriodicOrbit` class represents individual periodic orbits.

Creating Orbits
~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten import HaloOrbit
   
   # Create a halo orbit
   orbit = HaloOrbit(constants)
   
   # Set initial conditions
   orbit.set_initial_state([x0, y0, z0, vx0, vy0, vz0])
   
   # Compute the orbit
   result = orbit.compute()

Best Practices
--------------

1. **Use appropriate mass parameters** for your specific system
2. **Check Lagrange point positions** to ensure they're reasonable
3. **Validate initial conditions** before computing orbits
4. **Use consistent units** throughout your analysis
5. **Save intermediate results** for long computations

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Invalid mass parameter**
   - Ensure :math:`0 < \mu < 1`
   - Check that the parameter is physically reasonable

**Lagrange point computation fails**
   - Verify the mass parameter is valid
   - Check for numerical precision issues

**Orbit computation fails**
   - Verify initial conditions are reasonable
   - Check that the orbit is not too close to the primary bodies
   - Ensure the period guess is reasonable

For more help, see the :doc:`api/system` documentation or check the 
`GitHub Issues <https://github.com/iamgadmarconi/hiten/issues>`_.
