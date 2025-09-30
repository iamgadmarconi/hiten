Integrators Utilities
======================

The utils module provides shared Numba-compatible helpers for integrators.

.. currentmodule:: hiten.algorithms.integrators.utils

Event Functions
---------------

.. autofunction:: _event_crossed()

.. autofunction:: _crossed_direction()

.. autofunction:: _bisection_update()

.. autofunction:: _bracket_converged()

Step Control Functions
----------------------

.. autofunction:: _select_initial_step()

.. autofunction:: _clamp_step()

.. autofunction:: _adjust_step_to_endpoint()

PI Controller Functions
-----------------------

.. autofunction:: _pi_accept_factor()

.. autofunction:: _pi_reject_factor()

Error Functions
---------------

.. autofunction:: _error_scale()
