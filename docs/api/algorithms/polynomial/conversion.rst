Polynomial Conversion Functions
===============================

The conversion module provides helpers that convert between the internal coefficient-array representation of multivariate polynomials and symbolic SymPy expression objects.

.. currentmodule:: hiten.algorithms.polynomial.conversion

poly2sympy()
^^^^^^^^^^^^

The :func:`poly2sympy` function converts a polynomial represented as a list of coefficient arrays to a SymPy expression.

.. autofunction:: poly2sympy()

sympy2poly()
^^^^^^^^^^^^

The :func:`sympy2poly` function converts a SymPy expression to a polynomial represented as a list of coefficient arrays.

.. autofunction:: sympy2poly()

hpoly2sympy()
^^^^^^^^^^^^^

The :func:`hpoly2sympy` function converts a homogeneous polynomial coefficient array to a SymPy expression.

.. autofunction:: hpoly2sympy()
