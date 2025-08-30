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


def _make_strategy(kind: str, section_config, map_config, **kwargs) -> _CenterManifoldSeedingBase:
    """Factory returning a concrete seeding strategy.

    Parameters
    ----------
    kind : str
        One of the keys in ``_STRATEGY_MAP``.
    section_config
        Instance of ``_CenterManifoldSectionConfig`` describing the section.
    map_config
        The map-level configuration carrying global parameters such as
        ``n_seeds`` and ``seed_axis``.
    kwargs
        Extra keyword arguments forwarded to the concrete strategy.
    """
    try:
        cls = _STRATEGY_MAP[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown seed_strategy '{kind}'") from exc
    return cls(section_config, map_config, **kwargs)

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
