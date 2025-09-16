Fourier Module
==============

The Fourier module provides low-level helpers for working with Fourier-Taylor coefficient arrays in action-angle variables.

.. currentmodule:: hiten.algorithms.fourier

Base Functions
~~~~~~~~~~~~~~

The base module provides core Fourier analysis framework.

.. currentmodule:: hiten.algorithms.fourier.base

_pack_fourier_index()
^^^^^^^^^^^^^^^^^^^^^

The :func:`_pack_fourier_index` function packs exponents into a 64-bit key for constant-time lookup.

.. autofunction:: _pack_fourier_index()

_decode_fourier_index()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_decode_fourier_index` function is the inverse of :func:`_pack_fourier_index`.

.. autofunction:: _decode_fourier_index()

_init_fourier_tables()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_init_fourier_tables` function builds psiF and clmoF lookup tables for Fourier polynomials.

.. autofunction:: _init_fourier_tables()

_create_encode_dict_fourier()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_create_encode_dict_fourier` function creates a list of dictionaries mapping packed index to position for each degree.

.. autofunction:: _create_encode_dict_fourier()

_encode_fourier_index()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_encode_fourier_index` function encodes a Fourier index tuple to find its position in the coefficient array.

.. autofunction:: _encode_fourier_index()

Algebra Functions
~~~~~~~~~~~~~~~~~

The algebra module provides Fourier algebra operations.

.. currentmodule:: hiten.algorithms.fourier.algebra

_fpoly_add()
^^^^^^^^^^^^

The :func:`_fpoly_add` function adds two Fourier polynomial coefficient arrays element-wise.

.. autofunction:: _fpoly_add()

_fpoly_scale()
^^^^^^^^^^^^^^

The :func:`_fpoly_scale` function scales a Fourier polynomial coefficient array by a constant factor.

.. autofunction:: _fpoly_scale()

_fpoly_mul()
^^^^^^^^^^^^

The :func:`_fpoly_mul` function multiplies two Fourier polynomials using their coefficient arrays.

.. autofunction:: _fpoly_mul()

_fpoly_diff_action()
^^^^^^^^^^^^^^^^^^^^

The :func:`_fpoly_diff_action` function computes the partial derivative of a Fourier polynomial with respect to an action variable.

.. autofunction:: _fpoly_diff_action()

_fpoly_diff_angle()
^^^^^^^^^^^^^^^^^^^

The :func:`_fpoly_diff_angle` function computes the partial derivative of a Fourier polynomial with respect to an angle variable.

.. autofunction:: _fpoly_diff_angle()

_fpoly_poisson()
^^^^^^^^^^^^^^^^

The :func:`_fpoly_poisson` function computes the Poisson bracket of two Fourier polynomials.

.. autofunction:: _fpoly_poisson()

_fpoly_block_evaluate()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_fpoly_block_evaluate` function evaluates a Fourier polynomial block at specific action and angle values.

.. autofunction:: _fpoly_block_evaluate()

_fpoly_block_gradient()
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_fpoly_block_gradient` function computes the gradient of a Fourier polynomial block.

.. autofunction:: _fpoly_block_gradient()

_fpoly_block_hessian()
^^^^^^^^^^^^^^^^^^^^^^

The :func:`_fpoly_block_hessian` function computes the Hessian matrix of a Fourier polynomial block.

.. autofunction:: _fpoly_block_hessian()

Operations Functions
~~~~~~~~~~~~~~~~~~~~

The operations module provides high-level Fourier operations.

.. currentmodule:: hiten.algorithms.fourier.operations

_make_fourier_poly()
^^^^^^^^^^^^^^^^^^^^

The :func:`_make_fourier_poly` function creates a new Fourier polynomial coefficient array of specified degree.

.. autofunction:: _make_fourier_poly()

_fourier_evaluate()
^^^^^^^^^^^^^^^^^^^

The :func:`_fourier_evaluate` function evaluates a Fourier polynomial at specific action and angle values.

.. autofunction:: _fourier_evaluate()

_fourier_evaluate_with_grad()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_fourier_evaluate_with_grad` function evaluates a Fourier polynomial with gradient computation.

.. autofunction:: _fourier_evaluate_with_grad()

_fourier_hessian()
^^^^^^^^^^^^^^^^^^

The :func:`_fourier_hessian` function computes the Hessian matrix of a Fourier polynomial.

.. autofunction:: _fourier_hessian()

