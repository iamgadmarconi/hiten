from hiten.algorithms.corrector.interfaces import \
    _PeriodicOrbitCorrectorInterface
from hiten.algorithms.corrector.newton import _NewtonCore


class _NewtonOrbitCorrector(_PeriodicOrbitCorrectorInterface, _NewtonCore):
    """Periodic-orbit corrector that combines the orbit interface with the generic
    Newton engine.  All orbit-specific bookkeeping is provided by the interface
    mix-in; numerical work is handled by :class:`_NewtonCore`."""

    pass