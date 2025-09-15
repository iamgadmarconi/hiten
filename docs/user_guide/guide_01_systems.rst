System Creation and Configuration
==================================

This guide covers the fundamental concepts of creating and configuring dynamical systems in HITEN, starting with the most basic operations and building up to more complex configurations.

Basic System Creation
---------------------

HITEN provides several ways to create CR3BP systems, each suited for different use cases.

From Predefined Bodies
~~~~~~~~~~~~~~~~~~~~~~

The simplest way to create a system is using predefined celestial bodies:

.. code-block:: python

   from hiten import System
   
   # Create Earth-Moon system
   system = System.from_bodies("earth", "moon")
   
   # Create Sun-Earth system  
   system = System.from_bodies("sun", "earth")
   
   # Create Sun-Jupiter system
   system = System.from_bodies("sun", "jupiter")

The system automatically computes the mass parameter mu and sets up the appropriate physical constants.

From Mass Parameter
~~~~~~~~~~~~~~~~~~~

For custom systems or when you only know the mass parameter:

.. code-block:: python

   from hiten import System
   
   # Create system directly from mu
   system = System.from_mu(0.012150585609624)  # Earth-Moon mu
   
   # Custom mass parameter
   system = System.from_mu(0.000953875)  # Sun-Earth mu

Custom Body Creation
~~~~~~~~~~~~~~~~~~~~

For maximum control, create custom bodies:

.. code-block:: python

   from hiten import Body, System
   
   # Create custom primary body
   primary = Body(
       name="Custom Primary",
       mass=5.972e24,  # kg
       radius=6.371e6,  # m
       color="#4A90E2"
   )
   
   # Create secondary body orbiting the primary
   secondary = Body(
       name="Custom Secondary", 
       mass=7.342e22,  # kg
       radius=1.737e6,  # m
       color="#F5A623",
       _parent_input=primary
   )
   
   # Create system with custom bodies
   system = System(primary, secondary, distance=384400e3)  # km

System Properties
-----------------

Once created, systems provide access to key properties:

.. code-block:: python

   # Basic properties
   print(f"Mass parameter: {system.mu}")
   print(f"Primary: {system.primary.name}")
   print(f"Secondary: {system.secondary.name}")
   print(f"Distance: {system.distance} km")
   
   # Libration points
   l1 = system.get_libration_point(1)
   l2 = system.get_libration_point(2)
   l3 = system.get_libration_point(3)
   l4 = system.get_libration_point(4)
   l5 = system.get_libration_point(5)
   
   print(f"L1 position: {l1.position}")
   print(f"L2 position: {l2.position}")

Body Configuration
------------------

Bodies can be configured with various physical and visual properties:

.. code-block:: python

   from hiten import Body
   
   # Earth-like body
   earth = Body(
       name="Earth",
       mass=5.972e24,
       radius=6.371e6,
       color="#6B93D6"  # Blue color for plotting
   )
   
   # Moon-like body
   moon = Body(
       name="Moon",
       mass=7.342e22,
       radius=1.737e6,
       color="#C0C0C0",  # Silver color
       _parent_input=earth
   )

Body Properties
~~~~~~~~~~~~~~~

Bodies provide access to their physical properties:

.. code-block:: python

   print(f"Name: {earth.name}")
   print(f"Mass: {earth.mass} kg")
   print(f"Radius: {earth.radius} m")
   print(f"Color: {earth.color}")
   print(f"Parent: {earth.parent}")

System Validation
-----------------

HITEN performs basic validation on system parameters:

.. code-block:: python

   # Valid mass parameter range
   try:
       system = System.from_mu(0.51)  # Invalid: mu must be <= 0.5
   except ValueError as e:
       print(f"Error: {e}")
   
   # Valid range
   system = System.from_mu(0.01215)  # Earth-Moon system

Common System Configurations
----------------------------

Here are some commonly used system configurations:

Earth-Moon System
~~~~~~~~~~~~~~~~~

.. code-block:: python

   system = System.from_bodies("earth", "moon")
   # mu ≈ 0.01215
   # Distance ≈ 384,400 km

Sun-Earth System
~~~~~~~~~~~~~~~~

.. code-block:: python

   system = System.from_bodies("sun", "earth")
   # mu ≈ 0.000953875
   # Distance ≈ 149.6 million km

Sun-Jupiter System
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   system = System.from_bodies("sun", "jupiter")
   # mu ≈ 0.000953875
   # Distance ≈ 778.5 million km

Custom Binary System
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example: Binary star system
   primary_star = Body("Primary Star", 2.0e30, 1.0e9)
   secondary_star = Body("Secondary Star", 1.5e30, 8.0e8, _parent_input=primary_star)
   
   system = System(primary_star, secondary_star, distance=1.0e12)  # 1 AU

Next Steps
----------

Once you have a system configured, you can:

- Analyze libration points (see :doc:`guide_02_libration`)
- Propagate orbits (see :doc:`guide_03_propagation`)
- Create periodic orbits (see :doc:`guide_04_orbits`)

For more advanced system configurations, see :doc:`guide_17_dynamical_systems`.
