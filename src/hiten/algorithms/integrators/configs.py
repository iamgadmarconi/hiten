from dataclasses import dataclass


@dataclass(frozen=True)
class _EventConfig:
    """Configuration for a scalar event function g(t, y).

    Parameters
    ----------
    direction : int, default 0
        Crossing direction to detect:
        - 0: any sign change (g0 * g1 <= 0)
        - +1: only increasing crossings (g0 <= 0 and g1 >= 0)
        - -1: only decreasing crossings (g0 >= 0 and g1 <= 0)
    terminal : bool, default True
        When True, integration should stop at the first event.
    tol : float, default 1e-12
        Absolute time tolerance for root bracketing refinement.
    max_iter : int, default 50
        Maximum iterations of the bracketing refinement.
    """

    direction: int = 0
    terminal: bool = True
    tol: float = 1e-12
    max_iter: int = 50