HITEN Documentation
===================

.. image:: https://img.shields.io/badge/Python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python version

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/badge/Code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

HITEN is a computational toolkit for the Circular Restricted Three-Body Problem (CR3BP). 
It provides algorithms for computing periodic orbits, invariant manifolds, bifurcation analysis, 
and various numerical methods essential for dynamical systems analysis in astrodynamics.

Key Features
------------

- **Periodic Orbit Computation**: Continuation methods for families of periodic orbits
- **Invariant Manifolds**: Stable and unstable manifold computation and analysis
- **Bifurcation Analysis**: Detection and analysis of bifurcations in parameter families
- **Hamiltonian Methods**: Normal form theory and center manifold reduction
- **Poincare Maps**: Various mapping techniques for dynamical analysis
- **Fourier Analysis**: Spectral methods for periodic solutions
- **Polynomial Methods**: Algebraic approaches to dynamical systems
- **Integration Methods**: Symplectic and Runge-Kutta integrators

Quick Start
-----------

.. code-block:: python

   from hiten import Constants, CenterManifold, HaloOrbit
   
   # Set up the Earth-Moon system
   constants = Constants(mu=0.012150585609624)
   
   # Create a center manifold
   manifold = CenterManifold(constants)
   
   # Compute a halo orbit
   orbit = HaloOrbit(constants)

Installation
------------

.. code-block:: bash

   pip install hiten

For development installation:

.. code-block:: bash

   git clone https://github.com/iamgadmarconi/hiten.git
   cd hiten
   pip install -e .

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/system
   user_guide/algorithms
   user_guide/utilities

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/system
   api/algorithms
   api/utils

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/contributing
   developer/changelog

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   references
   glossary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/iamgadmarconi/hiten
.. _Issues: https://github.com/iamgadmarconi/hiten/issues
.. _Discussions: https://github.com/iamgadmarconi/hiten/discussions
