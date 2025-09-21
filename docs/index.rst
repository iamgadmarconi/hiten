HITEN Documentation
===================

.. image:: https://img.shields.io/badge/Python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python version

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

.. image:: https://img.shields.io/pypi/v/hiten.svg
   :target: https://pypi.org/project/hiten/
   :alt: PyPI version

HITEN is a computational toolkit for the Circular Restricted Three-Body Problem (CR3BP). 
It provides algorithms for computing periodic orbits, invariant manifolds, bifurcation analysis, 
and various numerical methods essential for dynamical systems analysis in astrodynamics.

Quick Start
-----------

.. code-block:: python

   from hiten import System

   system = System.from_bodies("earth", "moon")

   l1 = system.get_libration_point(1)

   orbit = l1.create_orbit("halo", amplitude_z=0.2, zenith="southern")
   orbit.correct(max_attempts=25)
   orbit.propagate(steps=1000)
   
   manifold = orbit.manifold(stable=True, direction="positive")
   manifold.compute()
   manifold.plot()

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

   user_guide/guide_01_systems
   user_guide/guide_02_libration
   user_guide/guide_03_propagation
   user_guide/guide_04_orbits
   user_guide/guide_05_manifolds
   user_guide/guide_06_poincare
   user_guide/guide_07_center_manifold
   user_guide/guide_14_polynomial
   user_guide/guide_13_fourier
   user_guide/guide_15_bifurcation
   user_guide/guide_10_integrators
   user_guide/guide_11_correction
   user_guide/guide_12_continuation
   user_guide/guide_16_connections
   user_guide/guide_17_dynamical_systems

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
   readme

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
