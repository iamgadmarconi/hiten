Polynomial Base Functions
=========================

The base module provides low-level helpers for manipulating multivariate polynomial coefficient arrays.

.. currentmodule:: hiten.algorithms.polynomial.base

_factorial()
^^^^^^^^^^^^

The :func:`_factorial` function calculates the factorial of a non-negative integer.

.. autofunction:: _factorial()

_combinations()
^^^^^^^^^^^^^^^

The :func:`_combinations` function calculates the binomial coefficient C(n,k).

.. autofunction:: _combinations()

_init_index_tables()
^^^^^^^^^^^^^^^^^^^^

The :func:`_init_index_tables` function initializes lookup tables for polynomial multi-index encoding and decoding.

.. autofunction:: _init_index_tables()

_pack_multiindex()
^^^^^^^^^^^^^^^^^^

The :func:`_pack_multiindex` function packs the exponents k_1 through k_5 into a 32-bit integer.

.. autofunction:: _pack_multiindex()

_decode_multiindex()
^^^^^^^^^^^^^^^^^^^^

The :func:`_decode_multiindex` function decodes a packed multi-index from its position in the lookup table.

.. autofunction:: _decode_multiindex()

_fill_exponents()
^^^^^^^^^^^^^^^^^

The :func:`_fill_exponents` function fills an output array with decoded exponents.

.. autofunction:: _fill_exponents()

_encode_multiindex()
^^^^^^^^^^^^^^^^^^^^

The :func:`_encode_multiindex` function encodes a multi-index to find its position in the coefficient array.

.. autofunction:: _encode_multiindex()

_make_poly()
^^^^^^^^^^^^

The :func:`_make_poly` function creates a new polynomial coefficient array of specified degree.

.. autofunction:: _make_poly()

_create_encode_dict_from_clmo()
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :func:`_create_encode_dict_from_clmo` function creates a list of dictionaries mapping packed multi-indices to their positions.

.. autofunction:: _create_encode_dict_from_clmo()
