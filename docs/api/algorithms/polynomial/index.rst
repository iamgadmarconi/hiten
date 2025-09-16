Polynomial Module
=================

The polynomial module provides comprehensive tools for manipulating multivariate polynomials
in the 6D phase space of the circular restricted three-body problem.

Polynomial Representation
-------------------------

The polynomial representation uses a compressed monomial ordering scheme that stores multivariate polynomials as homogeneous components:

.. math::

    P(q_1,q_2,q_3,p_1,p_2,p_3)=p^{(0)}+p^{(1)}+p^{(2)}+\cdots +p^{(n)}

where each component :math:`p^{(i)}` is represented by a dense coefficient array containing terms of degree :math:`i`. Polynomials are initialised for a given degree and are stored in lists containing the homogeneous terms up to the desired degree. For degree :math:`i`, the coefficient array has a size determined by the number of ways to choose from :math:`i` items with replacement from a set of 6 variables:

.. math::

    \text{size}=\ ^{i+6-1}C_{i} = \binom{i+5}{5} = \frac{(i+5)!}{5!\cdot i!}

For example, a degree 2 polynomial has :math:`\binom{2+5}{5} = 21` terms.

The representation uses a bit-packing scheme where each multi-index, representing the exponents of a monomial, is encoded into a compact 32-bit integer. This encoding allocates six bits for each of five variables (:math:`q_2, q_3, p_1, p_2, p_3`), with the sixth variable's exponent (:math:`q_1`) determined implicitly by the total degree constraint. This approach provides a memory-efficient representation while maintaining fast access patterns for polynomial operations.

The architecture of the polynomial system is built around three core components: the index table generation, the encoding and decoding mechanisms, and the algebraic operation kernels. The encoding system employs a dictionary-based lookup mechanism that provides constant-time access to coefficient positions, while the decoding system uses bitwise operations to extract individual exponents from the packed representation.

The index tables are precomputed combinatorial structures that map between the packed multi-index representation and the coefficient array positions. These tables consist of a combination count matrix that tracks the number of monomials for each degree and variable count, and a packed multi-index list that stores all valid exponent combinations for each degree. The tables are precomputed at initialisation.

The combinatorial (psi) table is a 2D array containing the monomial count per degree:

.. math::

    \psi[i,d]=\ ^{i+d-1}C_{i-1} = \binom{i+d-1}{i-1}

Where :math:`d` is the degree of the monomial and :math:`i` is the number of variables. The number of monomials increases significantly for higher degrees for the case :math:`i=6`.

The indices (clmo) table contains the packed multi-indices, where clmo[D] is the array of packed multi-indices for degree D, while an encode dictionary maps the packed indices to their array position.

The algebraic operations are implemented as Numba-accelerated kernels that operate directly on the coefficient arrays. The multiplication algorithm employs a parallel convolution-style approach where each thread accumulates partial results in private scratch arrays before a final reduction step combines the results. This design eliminates race conditions while maximizing parallel efficiency. The differentiation and integration operations leverage the packed multi-index representation to efficiently compute coefficient updates, while the Poisson bracket computation combines differentiation and multiplication operations in a specialized kernel that maintains the canonical structure of Hamiltonian systems.

The module is organized into several submodules:

.. toctree::
   :maxdepth: 2

   base
   algebra
   operations
   conversion
   coordinates

.. currentmodule:: hiten.algorithms.polynomial
