from .base import _SeedingStrategy
from .strategies import (_AxisAlignedSeeding, _LevelSetsSeeding,
                         _RadialSeeding, _RandomSeeding, _SingleAxisSeeding)

_STRATEGY_MAP = {
    "single": _SingleAxisSeeding,
    "axis_aligned": _AxisAlignedSeeding,
    "level_sets": _LevelSetsSeeding,
    "radial": _RadialSeeding,
    "random": _RandomSeeding,
}


def _make_strategy(kind: str, section_config, **kwargs) -> _SeedingStrategy:
    try:
        cls = _STRATEGY_MAP[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown seed_strategy '{kind}'") from exc
    return cls(section_config, **kwargs)

__all__ = [
    "_SeedingStrategy",
    "_make_strategy",
]
