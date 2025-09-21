Surface Events
==============

The events module provides surface event classes for defining Poincare section surfaces. These classes implement different types of geometric surfaces that can be used as section planes in dynamical systems analysis.

Core Events
-----------

.. currentmodule:: hiten.algorithms.poincare.core.events

.. autoclass:: _SurfaceEvent()
   :members:
   :exclude-members: __init__

Abstract base class for all Poincare section surface events. Defines the protocol for surface detection during trajectory integration.

.. autoclass:: _PlaneEvent()
   :members:
   :exclude-members: __init__

Event class for axis-aligned plane sections. Provides efficient detection of trajectory crossings through planes perpendicular to coordinate axes.

Synodic Events
--------------

.. currentmodule:: hiten.algorithms.poincare.synodic.events

.. autoclass:: _AffinePlaneEvent()
   :members:
   :exclude-members: __init__

Affine hyperplane event in the synodic frame. Extends the base surface event to provide specialized functionality for synodic Poincare sections with arbitrary hyperplane orientations. Supports both axis-aligned planes and general affine hyperplanes in 6D state space.
