Installation
============

HITEN requires Python 3.9 or higher. The recommended way to install HITEN is using pip.

Prerequisites
-------------

HITEN depends on several scientific Python packages:

- `numpy <https://numpy.org/>`_ (>= 1.23)
- `scipy <https://scipy.org/>`_ (>= 1.15)
- `numba <https://numba.pydata.org/>`_ (>= 0.61)
- `mpmath <https://mpmath.org/>`_ (>= 1.3)
- `sympy <https://www.sympy.org/>`_ (>= 1.14)
- `h5py <https://www.h5py.org/>`_ (>= 3.13)
- `matplotlib <https://matplotlib.org/>`_ (>= 3.10)
- `tqdm <https://tqdm.github.io/>`_ (>= 4.67)
- `pandas <https://pandas.pydata.org/>`_ (>= 2.3.0)

Installation Methods
--------------------

Stable Release
~~~~~~~~~~~~~~

To install the latest stable release from PyPI:

.. code-block:: bash

   pip install hiten

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

To install HITEN in development mode from source:

.. code-block:: bash

   git clone https://github.com/iamgadmarconi/hiten.git
   cd hiten
   pip install -e .

This will install HITEN in "editable" mode, meaning changes to the source code
will be immediately reflected in your Python environment.

With Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install HITEN with development tools (pytest, black, ruff):

.. code-block:: bash

   git clone https://github.com/iamgadmarconi/hiten.git
   cd hiten
   pip install -e ".[dev]"

Verification
------------

To verify that HITEN is installed correctly, run the following in a Python interpreter:

.. code-block:: python

   import hiten
   print(f"HITEN version: {hiten.__version__}")
   
   # Test basic functionality
   from hiten import Constants
   constants = Constants(mu=0.012150585609624)
   print(f"Earth-Moon system mu: {constants.mu}")

If this runs without errors, HITEN is properly installed.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'hiten'**
   - Ensure HITEN is installed: ``pip list | grep hiten``
   - Check your Python path: ``python -c "import sys; print(sys.path)"``

**Numba compilation errors**
   - Update numba: ``pip install --upgrade numba``
   - Clear numba cache: ``numba --clear-cache``

**Memory issues with large computations**
   - Reduce problem size or use more efficient algorithms
   - Consider using ``numba`` JIT compilation for better performance

**Performance issues**
   - Ensure you're using optimized BLAS/LAPACK libraries
   - Consider using Intel MKL or OpenBLAS

Getting Help
------------

If you encounter issues during installation:

1. Check the `GitHub Issues <https://github.com/iamgadmarconi/hiten/issues>`_
2. Create a new issue with:

   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the issue

Uninstallation
--------------

To uninstall HITEN:

.. code-block:: bash

   pip uninstall hiten

This will remove HITEN and its dependencies (unless they're used by other packages).
