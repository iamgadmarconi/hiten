from system.body import Body
from system.base import System, systemConfig
from orbits.base import orbitConfig
from orbits.lyapunov import LyapunovOrbit
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


def setup_lyapunov_orbit(system, libration_point_idx):
    config = orbitConfig(
        system=system,
        orbit_family="lyapunov",
        libration_point_idx=libration_point_idx,
        extra_params={"Ax": 4e-3}
    )
    return LyapunovOrbit(config)

def test_lyapunov_orbit_ic():
    logger.info("Testing Lyapunov orbit initial condition generation...")

    system = setup_tests()

    for i in range(1, 4):
        orbit = setup_lyapunov_orbit(system, i)
        print(orbit.initial_state)

    logger.info("Finished testing Lyapunov orbit initial condition generation.")

def test_lyapunov_differential_correction():
    logger.info("Testing Lyapunov orbit differential correction...")

    system = setup_tests()

    orbit = setup_lyapunov_orbit(system, 1)
    print(orbit.initial_state)
    logger.info("Initial state: %s", np.array2string(orbit.initial_state, precision=12, suppress_small=True))

    orbit.differential_correction()
    logger.info("Final state: %s", np.array2string(orbit.initial_state, precision=12, suppress_small=True))
    logger.info("Period: %s", orbit.period)

def test_lyapunov_orbit_propagation():
    logger.info("Testing Lyapunov orbit propagation...")

    system = setup_tests()

    orbit = setup_lyapunov_orbit(system, 1)
    orbit.differential_correction()
    orbit.propagate()

    logger.info("Finished testing Lyapunov orbit propagation.")
    logger.info("Trajectory: %s", np.array2string(orbit.trajectory, precision=12, suppress_small=True))


def test_lyapunov_orbit_plot():
    logger.info("Testing Lyapunov orbit plotting...")

    system = setup_tests()

    for i in range(1, 3):
        orbit = setup_lyapunov_orbit(system, i)
        orbit.differential_correction()
        orbit.propagate()
        orbit.plot()

    logger.info("Finished testing Lyapunov orbit plotting.")

if __name__ == "__main__":
    test_lyapunov_orbit_ic()
    test_lyapunov_differential_correction()
    test_lyapunov_orbit_propagation()
