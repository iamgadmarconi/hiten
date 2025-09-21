Synodic Poincare Maps
======================

The synodic module provides synodic Poincare map computation for precomputed trajectories, enabling analysis of existing orbit data. This module is designed for post-processing trajectories to extract Poincare sections and return maps from already computed orbits.

.. currentmodule:: hiten.algorithms.poincare.synodic

.. autoclass:: SynodicMap()
   :members:
   :exclude-members: __init__

Main user-facing class for synodic Poincare maps. Provides interface for detecting Poincare sections on precomputed trajectories from orbits, manifolds, or arbitrary trajectory data.

.. autoclass:: _SynodicMapConfig()
   :members:
   :exclude-members: __init__, __post_init__

Configuration class for synodic Poincare map detection and refinement. Includes geometric parameters for section definition and numerical parameters for detection algorithms.

.. autoclass:: _SynodicSectionConfig()
   :members:
   :exclude-members: __init__

Configuration for synodic Poincare sections. Defines section hyperplanes using normal vectors and offsets, with support for arbitrary section orientations.

.. autoclass:: _SynodicDetectionBackend()
   :members:
   :exclude-members: __init__

Backend for synodic Poincare section detection. Implements efficient algorithms for detecting section crossings on precomputed trajectories with support for linear and cubic interpolation.

.. autoclass:: _SynodicEngine()
   :members:
   :exclude-members: __init__

Computation engine for synodic Poincare maps. Coordinates trajectory processing, parallel detection, and result aggregation for multiple trajectories.

.. autoclass:: _SynodicEngineInterface()
   :members:
   :exclude-members: __init__, __repr__

Configuration adapter for synodic engines. Adapts synodic map configuration to the engine interface requirements.

.. autoclass:: _AffinePlaneEvent()
   :members:
   :exclude-members: __init__, __repr__

Surface event for affine hyperplanes. Implements section detection for arbitrary hyperplanes defined by normal vectors and offsets.

.. autoclass:: _NoOpStrategy()
   :members:
   :exclude-members: __init__

No-operation seeding strategy for synodic maps. Since synodic maps work with precomputed trajectories, no seeding is required.

.. autoclass:: _DetectionSettings()
   :members:
   :exclude-members: __init__

Cached numerical settings for efficient detection and refinement. Optimizes performance by pre-computing frequently used parameters.

.. autofunction:: _get_section_config()

Utility function for creating synodic section configuration objects from geometric parameters.

.. autofunction:: _project_batch()

Batch projection function for converting state vectors to 2D section coordinates. Supports both coordinate-based and custom projection functions.

.. autofunction:: _compute_event_values()       

Computes surface event values for a batch of states. Optimized for vectorized computation of section crossing detection.

.. autofunction:: _is_vectorizable_plane_event()

Checks if a plane event can be vectorized for efficient batch processing. Determines optimal computation strategy based on event type.

.. autofunction:: _on_surface_indices()

Identifies states that are already on the Poincare section surface. Handles tolerance-based detection and direction filtering.

.. autofunction:: _crossing_indices_and_alpha()

Detects sign changes and computes interpolation parameters for section crossings. Implements robust crossing detection with direction validation.

.. autofunction:: _refine_hits_linear()

Refines section crossing times using linear interpolation. Fast method for improving crossing time accuracy.

.. autofunction:: _refine_hits_cubic()

Refines section crossing times using cubic Hermite interpolation. High-accuracy method for precise crossing detection.

.. autofunction:: _order_and_dedup_hits()

Orders and deduplicates section hits. Removes duplicate crossings and sorts hits by time for proper sequence analysis.

.. autofunction:: _detect_with_segment_refine() 

Advanced detection with segment refinement. Implements dense crossing detection with configurable refinement levels for high-resolution analysis.
