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


def run_all_tests():
    test_halo_orbit_ic()


if __name__ == "__main__":
    run_all_tests()
