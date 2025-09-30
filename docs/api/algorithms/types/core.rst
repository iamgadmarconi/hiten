Core Types
==========

The core module defines the fundamental architectural patterns and base classes used throughout the Hiten framework.

.. currentmodule:: hiten.algorithms.types.core

Base Classes
------------

_HitenBase()
^^^^^^^^^^^^

The abstract base class for all public Hiten classes. Provides common functionality for serialization, service management, and computed property handling.

.. autoclass:: _HitenBase
   :members:
   :undoc-members:
   :exclude-members: __init__

_HitenBaseFacade()
^^^^^^^^^^^^^^^^^^

Abstract base class for user-facing facades in the Hiten framework. Provides a common pattern for building facades that orchestrate the entire pipeline: facade → engine → interface → backend.

.. autoclass:: _HitenBaseFacade
   :members:
   :undoc-members:
   :exclude-members: __init__

_HitenBaseEngine()
^^^^^^^^^^^^^^^^^^

Template providing the canonical engine flow. Handles orchestration between interfaces and backends.

.. autoclass:: _HitenBaseEngine
   :members:
   :undoc-members:
   :exclude-members: __init__

_HitenBaseInterface()
^^^^^^^^^^^^^^^^^^^^^

Shared contract for translating between domain objects and backends. Manages data transformation and backend invocation.

.. autoclass:: _HitenBaseInterface
   :members:
   :undoc-members:
   :exclude-members: __init__

_HitenBaseBackend()
^^^^^^^^^^^^^^^^^^^

Abstract base class for all backend implementations in the Hiten framework. Defines the common interface and lifecycle hooks for numerical computations.

.. autoclass:: _HitenBaseBackend
   :members:
   :undoc-members:
   :exclude-members: __init__

Problem and Result Types
------------------------

_HitenBaseProblem()
^^^^^^^^^^^^^^^^^^^

Marker base class for problem payloads produced by interfaces.

.. autoclass:: _HitenBaseProblem
   :members:

_HitenBaseResults()
^^^^^^^^^^^^^^^^^^^

Marker base class for user-facing results returned by engines.

.. autoclass:: _HitenBaseResults
   :members:

_HitenBaseConfig()
^^^^^^^^^^^^^^^^^^

Marker base class for configuration payloads produced by interfaces.

.. autoclass:: _HitenBaseConfig
   :members:

Backend Call
------------

_BackendCall()
^^^^^^^^^^^^^^

Describe a backend call with positional and keyword arguments.

.. autoclass:: _BackendCall
   :members:
   :undoc-members:
