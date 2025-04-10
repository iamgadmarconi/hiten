from system.body import Body
from system.base import System, systemConfig
from orbits.base import orbitConfig
from orbits.halo import HaloOrbit
from utils.constants import Constants
import numpy as np

from log_config import logger


def setup_tests():
    logger.info("Setting up test system...")
    earth_mass = Constants.get_mass("earth")
    earth_radius = Constants.get_radius("earth")
    moon_mass = Constants.get_mass("moon")
    moon_radius = Constants.get_radius("moon")
    distance = Constants.get_orbital_distance("earth", "moon")

    earth = Body("Earth", earth_mass, earth_radius, color="blue")
    moon = Body("Moon", moon_mass, moon_radius, color="gray", parent=earth)

    system = System(systemConfig(primary=earth, secondary=moon, distance=distance))
    logger.info("Test system setup complete.")
    return system

def setup_halo_orbit(system, libration_point_idx):
    config = orbitConfig(
        system=system,
        orbit_family="halo",
        libration_point_idx=libration_point_idx,
        extra_params={"Az": 0.2, "Zenith": "Southern"}
    )
    return HaloOrbit(config)

def test_halo_orbit_ic():
    logger.info("Testing Halo orbit initial condition generation...")

    system = setup_tests()

    for i in range(1, 3):
        orbit = setup_halo_orbit(system, i)
        print(orbit.initial_state)

    logger.info("Finished testing Halo orbit initial condition generation.")

def test_halo_differential_correction():
    logger.info("Testing Halo orbit differential correction...")

    system = setup_tests()

    orbit = setup_halo_orbit(system, 1)
    print(orbit.initial_state)
    logger.info("Initial state: %s", np.array2string(orbit.initial_state, precision=12, suppress_small=True))

    orbit.differential_correction()
    logger.info("Final state: %s", np.array2string(orbit.initial_state, precision=12, suppress_small=True))
    logger.info("Period: %s", orbit.period)

def test_halo_orbit_propagation():
    logger.info("Testing Halo orbit propagation...")

    system = setup_tests()

    orbit = setup_halo_orbit(system, 1)
    orbit.differential_correction()
    orbit.propagate()

    logger.info("Finished testing Halo orbit propagation.")
    logger.info("Trajectory: %s", np.array2string(orbit.trajectory, precision=12, suppress_small=True))

def test_halo_orbit_plot():
    logger.info("Testing Halo orbit plotting...")

    system = setup_tests()

    for i in range(1, 3):
        orbit = setup_halo_orbit(system, i)
        orbit.differential_correction()
        orbit.propagate()
        orbit.plot()

    logger.info("Finished testing Halo orbit plotting.")

def test_halo_orbit_stability():
    logger.info("Testing Halo orbit stability...")

    system = setup_tests()

    orbit = setup_halo_orbit(system, 1)
    orbit.differential_correction()
    orbit.propagate()
    orbit.compute_stability()
    logger.info("Eigenvalues: %s", orbit.stability_info[0])
    logger.info("Eigenvectors: %s", orbit.stability_info[1])
    logger.info("Finished testing Halo orbit stability.")
    logger.info("Is stable: %s", orbit.is_stable)
    logger.info("Is unstable: %s", orbit.is_unstable)

def test_halo_base_class():
    logger.info("Testing Halo base class...")

    system = setup_tests()

    orbit = setup_halo_orbit(system, 1)
    orbit.differential_correction()
    orbit.propagate()
    orbit.compute_stability()
    logger.info("Jacobi constant: %s", orbit.jacobi_constant)
    logger.info("Energy: %s", orbit.energy)
    logger.info("Finished testing Halo base class.")

def run_all_tests():
    test_halo_orbit_ic()
    test_halo_differential_correction()
    test_halo_orbit_propagation()
    test_halo_orbit_plot()
    test_halo_orbit_stability()
    test_halo_base_class()


if __name__ == "__main__":
    run_all_tests()
