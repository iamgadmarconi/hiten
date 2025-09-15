Polynomial Operations
=====================

The operations module provides high-level utilities for manipulating multivariate polynomials.

.. currentmodule:: hiten.algorithms.polynomial.operations

_polynomial_zero_list()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_zero_list` function creates a list of zero polynomial coefficient arrays up to a maximum degree.

.. autofunction:: _polynomial_zero_list()

_polynomial_variable()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_variable` function creates a polynomial representing a single variable.

.. autofunction:: _polynomial_variable()

_polynomial_variables_list()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_variables_list` function creates a list of polynomials for each variable in the system.

.. autofunction:: _polynomial_variables_list()

_polynomial_add_inplace()
^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_add_inplace` function adds or subtracts one polynomial to/from another in-place.

.. autofunction:: _polynomial_add_inplace()

_polynomial_multiply()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_multiply` function multiplies two polynomials.

.. autofunction:: _polynomial_multiply()

_polynomial_power()
^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_power` function raises a polynomial to a power using binary exponentiation.

.. autofunction:: _polynomial_power()

_polynomial_poisson_bracket()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_poisson_bracket` function computes the Poisson bracket of two polynomials.

.. autofunction:: _polynomial_poisson_bracket()

_polynomial_clean()
^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_clean` function creates a new polynomial with small coefficients set to zero.

.. autofunction:: _polynomial_clean()

_polynomial_degree()
^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_degree` function gets the degree of a polynomial represented as a list of homogeneous parts.

.. autofunction:: _polynomial_degree()

_polynomial_total_degree()
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_total_degree` function gets the total degree of a polynomial using the _get_degree kernel function.

.. autofunction:: _polynomial_total_degree()

_polynomial_differentiate()
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_differentiate` function computes the partial derivative of a polynomial with respect to a variable.

.. autofunction:: _polynomial_differentiate()

_polynomial_jacobian()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_jacobian` function computes the Jacobian matrix of a polynomial.

.. autofunction:: _polynomial_jacobian()

_polynomial_evaluate()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_evaluate` function evaluates a polynomial at a specific point.

.. autofunction:: _polynomial_evaluate()

_polynomial_integrate()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_polynomial_integrate` function integrates a polynomial with respect to one variable.

.. autofunction:: _polynomial_integrate()

_linear_variable_polys()
^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_linear_variable_polys` function creates polynomials for new variables after a linear transformation.

.. autofunction:: _linear_variable_polys()

_substitute_linear()
^^^^^^^^^^^^^^^^^^^^

The :func:`_substitute_linear` function performs variable substitution in a polynomial using a linear transformation.

.. autofunction:: _substitute_linear()

_linear_affine_variable_polys()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_linear_affine_variable_polys` function builds polynomials for variables after an affine change of variables.

.. autofunction:: _linear_affine_variable_polys()

_substitute_affine()
^^^^^^^^^^^^^^^^^^^^

The :func:`_substitute_affine` function substitutes an affine change of variables into a polynomial.

.. autofunction:: _substitute_affine()
