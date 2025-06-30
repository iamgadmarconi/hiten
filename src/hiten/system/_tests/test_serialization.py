import math
import shutil
from pathlib import Path

import numpy as np

from hiten.system import (CenterManifold, HaloOrbit, Manifold, System,
                          _PoincareMap)
from hiten.utils.log_config import logger

TMP_DIR = Path("results") / "serialization_test"


def _reset_tmp_dir() -> None:
    """Start from a clean directory each time the script is executed."""
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def _assert_equal(name: str, left: np.ndarray, right: np.ndarray, atol: float = 1e-12) -> None:
    if not np.allclose(left, right, atol=atol):
        raise AssertionError(f"{name}: round-trip mismatch (max |Δ| = {np.abs(left-right).max():.2e})")

def test_serialization() -> None:
    _reset_tmp_dir()

    logger.info("\n[SET-UP] Building minimal CR3BP objects ...")

    # 1. Base CR3BP system & libration point
    system = System.from_bodies("earth", "moon")
    L1 = system.get_libration_point(1)

    # 2. Periodic orbit (halo) - minimal example, no correction/propagation
    orbit = HaloOrbit(L1, amplitude_z=0.01, zenith="northern")
    orbit.period = 2 * math.pi  # quick dummy value to avoid runtime checks

    orbit_path = TMP_DIR / "halo_orbit.h5"
    logger.info("[PeriodicOrbit] saving: %s", orbit_path)
    orbit.save(str(orbit_path))

    orbit_loaded = HaloOrbit(L1, amplitude_z=0.01, zenith="northern")  # placeholder instance
    orbit_loaded.load_inplace(str(orbit_path))
    _assert_equal("PeriodicOrbit.initial_state", orbit.initial_state, orbit_loaded.initial_state)
    assert math.isclose(orbit.period or 0.0, orbit_loaded.period or 0.0, rel_tol=1e-12)
    logger.info("[PeriodicOrbit] round-trip OK\n")

    manifold = Manifold(orbit)
    man_path = TMP_DIR / "manifold.h5"
    logger.info("[Manifold] saving: %s", man_path)
    manifold.save(str(man_path))

    manifold_loaded = Manifold.load(str(man_path))
    # Integrity checks
    assert manifold_loaded.stable == manifold.stable
    assert manifold_loaded.direction == manifold.direction
    assert math.isclose(manifold_loaded.mu, manifold.mu, rel_tol=1e-15)
    _assert_equal("Manifold.generating_orbit.state",
                  manifold.generating_orbit.initial_state,
                  manifold_loaded.generating_orbit.initial_state)
    logger.info("[Manifold] round-trip OK\n")

    cm = CenterManifold(L1, max_degree=4)
    # Trigger polynomial computation so we have concrete data to compare
    poly_cm_original = cm.compute()
    cm_dir = TMP_DIR / "center_manifold"
    logger.info("[CenterManifold] saving: %s", cm_dir)
    cm.save(cm_dir)

    cm_loaded = CenterManifold.load(cm_dir)
    assert cm_loaded.max_degree == cm.max_degree
    _assert_equal("CenterManifold.point.position",
                  cm.point.position,
                  cm_loaded.point.position)

    # Retrieve cached polynomials from the *loaded* instance without recomputation
    poly_cm_loaded = cm_loaded.cache_get(("hamiltonian", cm_loaded.max_degree, "center_manifold_real"))

    assert poly_cm_loaded is not None, "Loaded CenterManifold lacks cached polynomial data!"
    assert len(poly_cm_original) == len(poly_cm_loaded), "Polynomial block count mismatch"
    for i, (blk_orig, blk_load) in enumerate(zip(poly_cm_original, poly_cm_loaded)):
        _assert_equal(f"CM polynomial block {i}", blk_orig, blk_load)
    logger.info("[CenterManifold] round-trip OK\n")

    energy_level = L1.energy + 1e-5
    pmap = _PoincareMap(cm, energy=energy_level)

    pmap_path = TMP_DIR / "poincare_map.h5"
    logger.info("[_PoincareMap] saving: %s", pmap_path)
    pmap.save(str(pmap_path))

    pmap_loaded = _PoincareMap(cm, energy=0.0)  # dummy - will be overwritten
    pmap_loaded.load_inplace(str(pmap_path))
    assert math.isclose(pmap_loaded.energy, energy_level, rel_tol=1e-12)
    # Dataclass comparison works out of the box
    assert pmap_loaded.config == pmap.config
    logger.info("[_PoincareMap] round-trip OK\n")

    logger.info("\nAll serialisation tests passed")
