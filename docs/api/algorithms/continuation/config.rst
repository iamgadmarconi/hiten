Continuation Configuration
==========================

The config module provides configuration classes for domain-specific continuation algorithms. These classes encapsulate the parameters required for different types of continuation methods (natural parameter, pseudo-arclength, etc.).

.. currentmodule:: hiten.algorithms.continuation.config

Base Configuration
------------------

.. currentmodule:: hiten.algorithms.continuation.config

_ContinuationConfig()
^^^^^^^^^^^^^^^^^^^^^

The :class:`_ContinuationConfig` class defines configuration parameters for continuation algorithms, including target bounds, step sizes, retry policies, and bounds enforcement.

.. autoclass:: _ContinuationConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__

Orbit Continuation Configuration
--------------------------------

.. currentmodule:: hiten.algorithms.continuation.config

_OrbitContinuationConfig()
^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`_OrbitContinuationConfig` class defines configuration parameters for periodic orbit continuation. This dataclass encapsulates configuration options specific to periodic orbit continuation, including state initialization, parameter extraction, and additional correction settings.

.. autoclass:: _OrbitContinuationConfig()
   :members:
   :undoc-members:
   :exclude-members: __init__
