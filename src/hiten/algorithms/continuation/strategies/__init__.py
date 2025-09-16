"""Provide the core algorithmic components for numerical continuation.

The :mod:`~hiten.algorithms.continuation.strategies` package provides the core
algorithmic components for numerical continuation in dynamical systems.
This package implements various continuation strategies, stepping methods,
and protocol definitions that form the foundation of the continuation
framework.

Examples
-------------
The strategies are typically used through higher-level interfaces in the
:mod:`~hiten.algorithms.continuation` package, but can be combined directly
for custom continuation scenarios:

>>> from hiten.algorithms.continuation.strategies._algorithms import _NaturalParameter
>>> from hiten.algorithms.continuation.strategies._stepping import _NaturalParameterStep
>>> from hiten.algorithms.continuation.strategies._step_interface import _ContinuationStep
>>>
>>> # Create a custom continuation algorithm
>>> class CustomOrbitContinuation(_NaturalParameter, _PeriodicOrbitContinuationInterface):
>>>     def __init__(self, predictor_fn, **kwargs):
>>>         stepper = _NaturalParameterStep(predictor_fn)
>>>         super().__init__(stepper, **kwargs)

See Also
--------
:mod:`~hiten.algorithms.continuation.base`
    Base continuation engine that coordinates with strategies.
:mod:`~hiten.algorithms.continuation.strategies._step_interface`
    Domain-specific interfaces for different problem types.
:mod:`~hiten.algorithms.continuation.predictors`
    High-level predictor classes for common continuation tasks.
"""
