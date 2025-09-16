Polynomial Algebra Functions
============================

The algebra module provides Numba accelerated helpers for algebraic manipulation of multivariate polynomial coefficient arrays.

.. currentmodule:: hiten.algorithms.polynomial.algebra

_poly_add()
^^^^^^^^^^^

The :func:`_poly_add` function adds two polynomial coefficient arrays element-wise.

.. autofunction:: _poly_add()

_poly_scale()
^^^^^^^^^^^^^

The :func:`_poly_scale` function scales a polynomial coefficient array by a constant factor.

.. autofunction:: _poly_scale()

_poly_mul()
^^^^^^^^^^^

The :func:`_poly_mul` function multiplies two polynomials using their coefficient arrays.

.. autofunction:: _poly_mul()

_poly_diff()
^^^^^^^^^^^^

The :func:`_poly_diff` function computes the partial derivative of a polynomial with respect to a variable.

.. autofunction:: _poly_diff()

_poly_poisson()
^^^^^^^^^^^^^^^

The :func:`_poly_poisson` function computes the Poisson bracket of two polynomials.

.. autofunction:: _poly_poisson()

_get_degree()
^^^^^^^^^^^^^

The :func:`_get_degree` function determines the degree of a polynomial from its coefficient array length.

.. autofunction:: _get_degree()

_poly_clean_inplace()
^^^^^^^^^^^^^^^^^^^^^

The :func:`_poly_clean_inplace` function sets coefficients with absolute value below tolerance to zero (in-place).

.. autofunction:: _poly_clean_inplace()

_poly_clean()
^^^^^^^^^^^^^

The :func:`_poly_clean` function sets coefficients with absolute value below tolerance to zero (out-of-place).

.. autofunction:: _poly_clean()

_poly_evaluate()
^^^^^^^^^^^^^^^^

The :func:`_poly_evaluate` function evaluates a polynomial at a specific point.

.. autofunction:: _poly_evaluate()

_evaluate_reduced_monomial()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_evaluate_reduced_monomial` function evaluates a monomial with modified exponent at specified coordinates.

.. autofunction:: _evaluate_reduced_monomial()

_poly_integrate()
^^^^^^^^^^^^^^^^^

The :func:`_poly_integrate` function integrates a polynomial with respect to one variable.

.. autofunction:: _poly_integrate()
