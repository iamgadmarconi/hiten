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
    """

    direction: int = 0
    terminal: bool = True