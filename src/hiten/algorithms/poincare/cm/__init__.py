from .config import _CenterManifoldSectionConfig
from .seeding import _CenterManifoldSeedingBase
from .strategies import (_AxisAlignedSeeding, _LevelSetsSeeding,
                         _RadialSeeding, _RandomSeeding, _SingleAxisSeeding)

_STRATEGY_MAP = {
    "single": _SingleAxisSeeding,
    "axis_aligned": _AxisAlignedSeeding,
    "level_sets": _LevelSetsSeeding,
    "radial": _RadialSeeding,
    "random": _RandomSeeding,
}


def _make_strategy(kind: str, section_config, **kwargs) -> _CenterManifoldSeedingBase:
    try:
        cls = _STRATEGY_MAP[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown seed_strategy '{kind}'") from exc
    return cls(section_config, **kwargs)

__all__ = [
    "_CenterManifoldSeedingBase",
    "_make_strategy",
    "_CenterManifoldSectionConfig",
    "_AxisAlignedSeeding",
    "_LevelSetsSeeding",
    "_RadialSeeding",
    "_RandomSeeding",
    "_SingleAxisSeeding",
]
