import math
import shutil
from pathlib import Path

import numpy as np

from hiten.system import (Body, CenterManifold, HaloOrbit, Hamiltonian, 
                          InvariantTori, LieGeneratingFunction, L1Point, 
                          L2Point, L4Point, Manifold, OrbitFamily, System,
                          CenterManifoldMap)
from hiten.utils.log_config import logger

TMP_DIR = Path("results") / "serialization_test"


def _reset_tmp_dir() -> None:
    """Start from a clean directory each time the script is executed."""
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def _assert_equal(name: str, left: np.ndarray, right: np.ndarray, atol: float = 1e-12) -> None:
    if not np.allclose(left, right, atol=atol):
        raise AssertionError(f"{name}: round-trip mismatch (max |delta | = {np.abs(left-right).max():.2e})")

def test_serialization() -> None:
    _reset_tmp_dir()

    logger.info("\n[SET-UP] Building minimal CR3BP objects ...")

    # 1. Base CR3BP system & libration point
    system = System.from_bodies("earth", "moon")
    L1 = system.get_libration_point(1)

    # 2. Periodic orbit (halo) - minimal example, no correction/propagation
    orbit = HaloOrbit(L1, amplitude_z=0.01, zenith="northern")
    orbit.period = 2 * math.pi  # quick dummy value to avoid runtime checks

    orbit_path = TMP_DIR / "halo_orbit.pkl"
    logger.info("[PeriodicOrbit] saving: %s", orbit_path)
    orbit.save(str(orbit_path))

    orbit_loaded = HaloOrbit(L1, amplitude_z=0.01, zenith="northern")  # placeholder instance
    orbit_loaded.load_inplace(str(orbit_path))
    _assert_equal("PeriodicOrbit.initial_state", orbit.initial_state, orbit_loaded.initial_state)
    assert math.isclose(orbit.period or 0.0, orbit_loaded.period or 0.0, rel_tol=1e-12)
    # Test that system and libration point relationships are preserved
    assert orbit_loaded.system.mu == orbit.system.mu
    assert orbit_loaded.system.primary.name == orbit.system.primary.name
    assert orbit_loaded.system.secondary.name == orbit.system.secondary.name
    assert orbit_loaded.libration_point.idx == orbit.libration_point.idx
    logger.info("[PeriodicOrbit] round-trip OK\n")

    manifold = Manifold(orbit)
    man_path = TMP_DIR / "manifold.pkl"
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
    # Test that generating orbit relationships are preserved
    assert manifold_loaded.generating_orbit.system.mu == manifold.generating_orbit.system.mu
    assert manifold_loaded.generating_orbit.libration_point.idx == manifold.generating_orbit.libration_point.idx
    logger.info("[Manifold] round-trip OK\n")

    cm = CenterManifold(L1, degree=6)
    # Trigger polynomial computation so we have concrete data to compare
    poly_cm_original = cm.compute()
    cm_dir = TMP_DIR / "center_manifold"
    logger.info("[CenterManifold] saving: %s", cm_dir)
    cm.save(cm_dir)

    cm_loaded = CenterManifold.load(cm_dir)
    assert cm_loaded.degree == cm.degree
    _assert_equal("CenterManifold.point.position",
                  cm.point.position,
                  cm_loaded.point.position)
    poly_cm_loaded = cm_loaded.compute()

    assert len(poly_cm_original) == len(poly_cm_loaded), "Polynomial block count mismatch"
    for i, (blk_orig, blk_load) in enumerate(zip(poly_cm_original, poly_cm_loaded)):
        _assert_equal(f"CM polynomial block {i}", blk_orig, blk_load)
    logger.info("[CenterManifold] round-trip OK\n")

    energy_level = 0.2
    pmap = CenterManifoldMap(cm, energy=energy_level)
    
    # Compute a section to have data to compare
    pmap.compute(section_coord="q3")

    pmap_path = TMP_DIR / "poincare_map.pkl"
    logger.info("[CenterManifoldMap] saving: %s", pmap_path)
    pmap.save(str(pmap_path))

    pmap_loaded = CenterManifoldMap(cm, energy=0.0)
    pmap_loaded.load_inplace(str(pmap_path))
    assert math.isclose(pmap_loaded.energy, energy_level, rel_tol=1e-12)
    # Dataclass comparison works out of the box
    assert pmap_loaded.config == pmap.config

    # Verify stored points via the new API
    _assert_equal(
        "Poincare map points",
        pmap.get_points(),
        pmap_loaded.get_points(),
    )

    logger.info("[CenterManifoldMap] round-trip OK\n")

    logger.info("\nAll serialisation tests passed")


def test_body_serialization() -> None:
    """Test Body class serialization."""
    _reset_tmp_dir()
    
    logger.info("\n[SET-UP] Testing Body serialization ...")
    
    # Create a Body instance
    earth = Body("Earth", 5.972e24, 6.371e6, color="blue")
    moon = Body("Moon", 7.342e22, 1.737e6, color="gray", parent=earth)
    
    # Test Earth serialization
    earth_path = TMP_DIR / "earth.pkl"
    logger.info("[Body] saving Earth: %s", earth_path)
    earth.save(str(earth_path))
    
    earth_loaded = Body.load(str(earth_path))
    assert earth_loaded.name == earth.name
    assert earth_loaded.mass == earth.mass
    assert earth_loaded.radius == earth.radius
    assert earth_loaded.color == earth.color
    assert earth_loaded.parent is earth_loaded  # Primary body
    logger.info("[Body] Earth round-trip OK")
    
    # Test Moon serialization
    moon_path = TMP_DIR / "moon.pkl"
    logger.info("[Body] saving Moon: %s", moon_path)
    moon.save(str(moon_path))
    
    moon_loaded = Body.load(str(moon_path))
    assert moon_loaded.name == moon.name
    assert moon_loaded.mass == moon.mass
    assert moon_loaded.radius == moon.radius
    assert moon_loaded.color == moon.color
    # Parent relationship should now be preserved with pickle serialization
    assert moon_loaded.parent.name == moon.parent.name
    logger.info("[Body] Moon round-trip OK")
    
    logger.info("[Body] serialization tests passed\n")


def test_system_serialization() -> None:
    """Test System class serialization."""
    _reset_tmp_dir()
    
    logger.info("\n[SET-UP] Testing System serialization ...")
    
    # Create a system
    system = System.from_bodies("earth", "moon")
    
    # Test System serialization
    system_path = TMP_DIR / "system.pkl"
    logger.info("[System] saving: %s", system_path)
    system.save(str(system_path))
    
    system_loaded = System.load(str(system_path))
    assert system_loaded.mu == system.mu
    assert system_loaded.distance == system.distance
    assert system_loaded.primary.name == system.primary.name
    assert system_loaded.secondary.name == system.secondary.name
    
    # Test that parent relationships are preserved
    assert system_loaded.secondary.parent.name == system.secondary.parent.name
    assert system_loaded.primary.parent is system_loaded.primary  # Primary should be self-parent
    assert system_loaded.secondary.parent is system_loaded.primary  # Secondary should reference primary
    
    logger.info("[System] round-trip OK")
    logger.info("[System] serialization tests passed\n")


def test_libration_point_serialization() -> None:
    """Test LibrationPoint classes serialization."""
    _reset_tmp_dir()
    
    logger.info("\n[SET-UP] Testing LibrationPoint serialization ...")
    
    # Create a system and libration points
    system = System.from_bodies("earth", "moon")
    L1 = system.get_libration_point(1)
    L2 = system.get_libration_point(2)
    L4 = system.get_libration_point(4)
    
    # Test L1 serialization
    l1_path = TMP_DIR / "l1.pkl"
    logger.info("[L1Point] saving: %s", l1_path)
    L1.save(str(l1_path))
    
    l1_loaded = L1Point.load(str(l1_path))
    assert l1_loaded.idx == L1.idx
    assert l1_loaded.mu == L1.mu
    _assert_equal("L1Point.position", L1.position, l1_loaded.position)
    # Test that system relationship is preserved
    assert l1_loaded.system.mu == L1.system.mu
    assert l1_loaded.system.primary.name == L1.system.primary.name
    assert l1_loaded.system.secondary.name == L1.system.secondary.name
    logger.info("[L1Point] round-trip OK")
    
    # Test L2 serialization
    l2_path = TMP_DIR / "l2.pkl"
    logger.info("[L2Point] saving: %s", l2_path)
    L2.save(str(l2_path))
    
    l2_loaded = L2Point.load(str(l2_path))
    assert l2_loaded.idx == L2.idx
    assert l2_loaded.mu == L2.mu
    _assert_equal("L2Point.position", L2.position, l2_loaded.position)
    # Test that system relationship is preserved
    assert l2_loaded.system.mu == L2.system.mu
    assert l2_loaded.system.primary.name == L2.system.primary.name
    assert l2_loaded.system.secondary.name == L2.system.secondary.name
    logger.info("[L2Point] round-trip OK")
    
    # Test L4 serialization
    l4_path = TMP_DIR / "l4.pkl"
    logger.info("[L4Point] saving: %s", l4_path)
    L4.save(str(l4_path))
    
    l4_loaded = L4Point.load(str(l4_path))
    assert l4_loaded.idx == L4.idx
    assert l4_loaded.mu == L4.mu
    _assert_equal("L4Point.position", L4.position, l4_loaded.position)
    # Test that system relationship is preserved
    assert l4_loaded.system.mu == L4.system.mu
    assert l4_loaded.system.primary.name == L4.system.primary.name
    assert l4_loaded.system.secondary.name == L4.system.secondary.name
    logger.info("[L4Point] round-trip OK")
    
    logger.info("[LibrationPoint] serialization tests passed\n")


def test_hamiltonian_serialization() -> None:
    """Test Hamiltonian class serialization."""
    _reset_tmp_dir()
    
    logger.info("\n[SET-UP] Testing Hamiltonian serialization ...")
    
    # Create a system and center manifold to get a Hamiltonian
    system = System.from_bodies("earth", "moon")
    L1 = system.get_libration_point(1)
    cm = CenterManifold(L1, degree=4)
    
    # Get a Hamiltonian from the center manifold
    hamiltonian = cm.compute()
    
    # Test Hamiltonian serialization
    ham_path = TMP_DIR / "hamiltonian.pkl"
    logger.info("[Hamiltonian] saving: %s", ham_path)
    hamiltonian.save(str(ham_path))
    
    ham_loaded = Hamiltonian.load(str(ham_path))
    assert ham_loaded.name == hamiltonian.name
    assert ham_loaded.degree == hamiltonian.degree
    assert ham_loaded.ndof == hamiltonian.ndof
    assert len(ham_loaded.poly_H) == len(hamiltonian.poly_H)
    
    # Compare polynomial coefficients
    for i, (orig, loaded) in enumerate(zip(hamiltonian.poly_H, ham_loaded.poly_H)):
        _assert_equal(f"Hamiltonian.poly_H[{i}]", orig, loaded)
    
    logger.info("[Hamiltonian] round-trip OK")
    logger.info("[Hamiltonian] serialization tests passed\n")


def test_lie_generating_function_serialization() -> None:
    """Test LieGeneratingFunction class serialization."""
    _reset_tmp_dir()
    
    logger.info("\n[SET-UP] Testing LieGeneratingFunction serialization ...")
    
    # Create a system and center manifold to get generating functions
    system = System.from_bodies("earth", "moon")
    L1 = system.get_libration_point(1)
    cm = CenterManifold(L1, degree=4)
    
    # Get generating functions from the center manifold
    generating_functions = L1.generating_functions(4)
    
    if generating_functions:  # Only test if generating functions exist
        lgf = generating_functions[0]  # Test the first one
        
        # Test LieGeneratingFunction serialization
        lgf_path = TMP_DIR / "lie_generating_function.pkl"
        logger.info("[LieGeneratingFunction] saving: %s", lgf_path)
        lgf.save(str(lgf_path))
        
        lgf_loaded = LieGeneratingFunction.load(str(lgf_path))
        assert lgf_loaded.name == lgf.name
        assert lgf_loaded.degree == lgf.degree
        assert lgf_loaded.ndof == lgf.ndof
        assert len(lgf_loaded.poly_G) == len(lgf.poly_G)
        
        # Compare polynomial coefficients
        for i, (orig, loaded) in enumerate(zip(lgf.poly_G, lgf_loaded.poly_G)):
            _assert_equal(f"LieGeneratingFunction.poly_G[{i}]", orig, loaded)
        
        logger.info("[LieGeneratingFunction] round-trip OK")
    else:
        logger.info("[LieGeneratingFunction] No generating functions available, skipping test")
    
    logger.info("[LieGeneratingFunction] serialization tests passed\n")


def test_orbit_family_serialization() -> None:
    """Test OrbitFamily class serialization."""
    _reset_tmp_dir()
    
    logger.info("\n[SET-UP] Testing OrbitFamily serialization ...")
    
    # Create a system and some orbits
    system = System.from_bodies("earth", "moon")
    L1 = system.get_libration_point(1)
    
    # Create a few orbits
    orbits = []
    for i in range(3):
        orbit = HaloOrbit(L1, amplitude_z=0.01 + i*0.005, zenith="northern")
        orbit.period = 2 * math.pi + i * 0.1  # Different periods
        orbits.append(orbit)
    
    # Create orbit family
    family = OrbitFamily(orbits, parameter_name="amplitude", 
                        parameter_values=np.array([0.01, 0.015, 0.02]))
    
    # Test OrbitFamily serialization
    family_path = TMP_DIR / "orbit_family.pkl"
    logger.info("[OrbitFamily] saving: %s", family_path)
    family.save(str(family_path))
    
    family_loaded = OrbitFamily.load(str(family_path))
    assert family_loaded.parameter_name == family.parameter_name
    assert len(family_loaded.orbits) == len(family.orbits)
    _assert_equal("OrbitFamily.parameter_values", 
                  family.parameter_values, 
                  family_loaded.parameter_values)
    
    # Compare orbits
    for i, (orig, loaded) in enumerate(zip(family.orbits, family_loaded.orbits)):
        _assert_equal(f"OrbitFamily.orbits[{i}].initial_state", 
                      orig.initial_state, 
                      loaded.initial_state)
        assert orig.period == loaded.period
        # Test that orbit relationships are preserved
        assert loaded.system.mu == orig.system.mu
        assert loaded.libration_point.idx == orig.libration_point.idx
        assert loaded.system.primary.name == orig.system.primary.name
        assert loaded.system.secondary.name == orig.system.secondary.name
    
    logger.info("[OrbitFamily] round-trip OK")
    logger.info("[OrbitFamily] serialization tests passed\n")


def test_invariant_tori_serialization() -> None:
    """Test InvariantTori class serialization."""
    _reset_tmp_dir()
    
    logger.info("\n[SET-UP] Testing InvariantTori serialization ...")
    
    # Create a system and orbit
    system = System.from_bodies("earth", "moon")
    L1 = system.get_libration_point(1)
    orbit = HaloOrbit(L1, amplitude_z=0.01, zenith="northern")
    orbit.period = 2 * math.pi
    orbit.correct()
    orbit.propagate()
    
    # Create invariant tori
    tori = InvariantTori(orbit)
    
    # Compute the torus grid
    epsilon = 0.1
    n_theta1, n_theta2 = 10, 8
    grid = tori.compute(epsilon=epsilon, n_theta1=n_theta1, n_theta2=n_theta2)
    
    # Test InvariantTori serialization
    tori_path = TMP_DIR / "invariant_tori.pkl"
    logger.info("[InvariantTori] saving: %s", tori_path)
    tori.save(str(tori_path))
    
    tori_loaded = InvariantTori.load(str(tori_path))
    assert tori_loaded.period == tori.period
    assert tori_loaded.jacobi == tori.jacobi
    assert tori_loaded.energy == tori.energy
    
    # Compare grids
    _assert_equal("InvariantTori.grid", tori.grid, tori_loaded.grid)
    
    # Test that orbit relationships are preserved
    assert tori_loaded.orbit.system.mu == tori.orbit.system.mu
    assert tori_loaded.orbit.libration_point.idx == tori.orbit.libration_point.idx
    assert tori_loaded.orbit.system.primary.name == tori.orbit.system.primary.name
    assert tori_loaded.orbit.system.secondary.name == tori.orbit.system.secondary.name
    
    logger.info("[InvariantTori] round-trip OK")
    logger.info("[InvariantTori] serialization tests passed\n")
