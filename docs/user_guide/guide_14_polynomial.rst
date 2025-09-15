Polynomial Methods and Algebra
==============================

This guide covers HITEN's polynomial manipulation capabilities, which form the foundation for Hamiltonian methods, normal form computations, and center manifold analysis.

Understanding Polynomial Methods
-------------------------------------

Polynomial methods are essential for:
- Hamiltonian system representation
- Normal form transformations
- Center manifold computations
- Lie series methods
- Coordinate transformations

HITEN's polynomial system is designed for high-performance manipulation of multivariate polynomials in the 6D phase space (q1, q2, q3, p1, p2, p3) of the circular restricted three-body problem. The implementation uses Numba JIT compilation and optimized data structures for efficient computation.

Polynomial Representation
~~~~~~~~~~~~~~~~~~~~~~~~~

HITEN represents polynomials as lists of coefficient arrays, where each array contains coefficients for a specific degree:

.. code-block:: python

   import numpy as np
   from hiten.algorithms.polynomial.base import _init_index_tables, _make_poly
   from hiten.algorithms.polynomial.operations import _polynomial_evaluate, _polynomial_variable

   # Initialize polynomial tables for degree 6
   psi, clmo = _init_index_tables(6)
   
   # Create a zero polynomial of degree 2
   poly_degree_2 = _make_poly(2, psi)
   print(f"Polynomial degree 2 has {len(poly_degree_2)} coefficients")

   # Create a polynomial representing the variable x1 (position q1)
   x1_poly = _polynomial_variable(0, 6, psi, clmo, _ENCODE_DICT_GLOBAL)
   print(f"x1 polynomial has {len(x1_poly)} homogeneous parts")

Polynomial Evaluation
--------------------

Evaluate polynomials at specific points in the 6D phase space:

.. code-block:: python

   # Evaluate a polynomial at a point in 6D phase space
   point = np.array([0.1, 0.2, 0.0, 0.05, 0.1, 0.0])  # [q1, q2, q3, p1, p2, p3]
   
   # Evaluate the x1 polynomial at this point
   value = _polynomial_evaluate(x1_poly, point, clmo)
   print(f"x1 polynomial value at point: {value}")

Coordinate Transformations
--------------------------

Polynomial methods enable coordinate transformations for normal form analysis:

Physical to Modal Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.polynomial.coordinates import physical_to_real_modal

   # Transform from physical to modal coordinates
   # This is used in center manifold computations
   physical_poly = _polynomial_variable(0, 6, psi, clmo, _ENCODE_DICT_GLOBAL)  # x1 variable

   # Transform to modal coordinates
   modal_poly = physical_to_real_modal(physical_poly)
   print(f"Modal polynomial created")

Modal to Complex Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.polynomial.coordinates import real_modal_to_complex

   # Transform from modal to complex coordinates
   complex_poly = real_modal_to_complex(modal_poly)
   print(f"Complex polynomial created")

Hamiltonian Polynomial Construction
----------------------------------------

Build polynomial representations of Hamiltonian systems:

CR3BP Hamiltonian
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.hamiltonian.hamiltonian import build_crtbp_hamiltonian
   from hiten import System

   system = System.from_bodies("earth", "moon")
   l1 = system.get_libration_point(1)

   # Build polynomial Hamiltonian around L1
   hamiltonian_poly = build_crtbp_hamiltonian(
       libration_point=l1,
       max_degree=6  # Truncate at 6th order
   )

   print(f"Hamiltonian polynomial degree: {hamiltonian_poly.degree}")
   print(f"Number of terms: {len(hamiltonian_poly.coefficients)}")

Lie Series Transformations
--------------------------------

Use Lie series for normal form computations:

Lie Series Application
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.hamiltonian.lie import lie_series_transform

   # Apply Lie series transformation
   # This is used in normal form computations
   transformed_poly = lie_series_transform(
       hamiltonian_poly,
       generating_function=generating_function,  # Some generating function
       order=4  # Transform up to 4th order
   )

   print(f"Transformed polynomial created")

Polynomial Algebra
------------------------

Advanced polynomial operations:

Polynomial Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.polynomial.operations import _polynomial_differentiate

   # Compute partial derivatives
   dx_poly, max_deg = _polynomial_differentiate(
       x1_poly, 
       var_idx=0,  # d/dx1
       max_deg=6,
       psi_table=psi,
       clmo_table=clmo,
       derivative_psi_table=psi,
       derivative_clmo_table=clmo,
       encode_dict_list=_ENCODE_DICT_GLOBAL
   )

   print(f"Derivative polynomial created with max degree {max_deg}")

Polynomial Multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.polynomial.operations import _polynomial_multiply

   # Multiply two polynomials
   x1_poly = _polynomial_variable(0, 6, psi, clmo, _ENCODE_DICT_GLOBAL)
   x2_poly = _polynomial_variable(1, 6, psi, clmo, _ENCODE_DICT_GLOBAL)
   
   product = _polynomial_multiply(x1_poly, x2_poly, 6, psi, clmo, _ENCODE_DICT_GLOBAL)
   print(f"Product polynomial created")

Poisson Brackets
~~~~~~~~~~~~~~~~

.. code-block:: python

   from hiten.algorithms.polynomial.operations import _polynomial_poisson_bracket

   # Compute Poisson bracket of two polynomials
   poisson_bracket = _polynomial_poisson_bracket(
       x1_poly, x2_poly, 6, psi, clmo, _ENCODE_DICT_GLOBAL
   )
   print(f"Poisson bracket computed")

Working with Polynomial Arrays
------------------------------------

Handle arrays of polynomials efficiently:

.. code-block:: python

   # Create array of polynomials
   poly_array = [x1_poly, x2_poly, product]
   
   # Evaluate all polynomials at once
   point = np.array([0.1, 0.2, 0.0, 0.05, 0.1, 0.0])
   values = [_polynomial_evaluate(p, point, clmo) for p in poly_array]
   
   print(f"Values: {values}")

Next Steps
----------

Once you understand polynomial methods, you can:

- Learn about Hamiltonian methods (see :doc:`guide_07_center_manifold`)
- Explore connection analysis (see :doc:`guide_16_connections`)
- Study advanced integration techniques (see :doc:`guide_10_integrators`)

For more advanced polynomial techniques, see the HITEN source code in :mod:`hiten.algorithms.polynomial`.
