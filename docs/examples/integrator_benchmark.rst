Integrator Benchmark Example
============================


Test Hiten integrators against SciPy on various ODE problems and provides

.. literalinclude:: ../../examples/integrators/integrator_benchmark.py
   :language: python
   :linenos:
   :caption: Testing Hiten integrators against SciPy on various ODE problems


Event Integrator Benchmark Example
==================================

Tests Hiten's event-capable integrator (DOP853) against SciPy solvers on simple problems with known event times. Reports detection accuracy and speed, and saves a comparison plot.

.. literalinclude:: ../../examples/integrators/event_integrator_benchmark.py
   :language: python
   :linenos:
   :caption: Testing Hiten's event-capable integrator (DOP853) against SciPy solvers on simple problems with known event times

Symplectic Event Integrator Benchmark Example
=============================================

Tests Hiten's extended symplectic integrators with event detection enabled against SciPy solvers on simple problems with known event times. Reports detection accuracy and speed, and saves a comparison plot.

.. literalinclude:: ../../examples/integrators/symplectic_event_benchmark.py
   :language: python
   :linenos:
   :caption: Testing Hiten's extended symplectic integrators with event detection enabled against SciPy solvers on simple problems with known event times