r"""
hiten.system.orbits.vertical
======================

Periodic vertical orbits of the circular restricted three-body problem.

This module supplies concrete realisations of :pyclass:`hiten.system.orbits.base.PeriodicOrbit`
corresponding to the vertical family around the collinear libration points
:math:`L_1` and :math:`L_2`.  Each class provides an analytical first guess
together with a customised differential corrector that exploits the symmetries
of the family.

References
----------
Szebehely, V. (1967). "Theory of Orbits".
"""

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from hiten.algorithms.dynamics.utils.geometry import _find_z_zero_crossing
from hiten.system.libration.collinear import CollinearPoint
from hiten.system.orbits.base import (PeriodicOrbit, S, _ContinuationConfig,
                                      _CorrectionConfig)


class VerticalOrbit(PeriodicOrbit):
    r"""
    Vertical family about a collinear libration point.

    The orbit oscillates out of the synodic plane and is symmetric with
    respect to the :math:`x`-:math:`z` plane.  Initial-guess generation is not
    yet available.

    Parameters
    ----------
    libration_point : CollinearPoint
        Target :pyclass:`CollinearPoint` around
        which the orbit is computed.
    initial_state : Sequence[float] or None, optional
        Optional six-dimensional initial state vector.

    Notes
    -----
    The implementation of the analytical seed and the Jacobian adjustment for
    the vertical family is work in progress.
    """
    
    _family = "vertical"

    def __init__(self, libration_point: CollinearPoint, initial_state: Optional[Sequence[float]] = None):
        super().__init__(libration_point, initial_state)

    def _initial_guess(self) -> NDArray[np.float64]:
        raise NotImplementedError("Initial guess is not implemented for Vertical orbits.")

    @property
    def amplitude(self) -> float:
        """(Read-only) Current z-amplitude of the vertical orbit."""
        if getattr(self, "_initial_state", None) is not None:
            return float(abs(self._initial_state[S.Z]))
        return float(self._amplitude_z)

    @property
    def eccentricity(self) -> float:
        """Eccentricity is not a well-defined concept for vertical orbits."""
        return np.nan

    @property
    def _correction_config(self) -> _CorrectionConfig:
        """Provides the differential correction configuration for vertical orbits."""
        return _CorrectionConfig(
            residual_indices=(S.VX, S.Y),     # Want VX=0 and Y=0
            control_indices=(S.VZ, S.VY),     # Adjust initial VZ and VY
            target=(0.0, 0.0),
            extra_jacobian=None,
            event_func=_find_z_zero_crossing,
        )

    @property
    def _continuation_config(self) -> _ContinuationConfig:
        """Default continuation parameter: vary the out-of-plane amplitude."""
        return _ContinuationConfig(state=S.Z, amplitude=True)
