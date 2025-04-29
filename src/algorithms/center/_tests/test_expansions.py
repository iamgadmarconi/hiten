from algorithms.expansions.center import legendre_series
from utils.constants import Constants
from system.libration import L1Point, L2Point, L3Point
from system.base import System, systemConfig
from system.body import Body
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

def test_legendre_series():
    logger.info("Testing Legendre series...")
    system = setup_tests()
    mu = system.mu
    for i in range(1, 4):
        logger.info("Testing Legendre series for libration point %s", i)
        libration_point = system.get_libration_point(i)
        state = libration_point.position
        for n in range(2, 10):
            series = legendre_series(state, n, mu, libration_point)
            logger.info("Legendre series: %s", series)
        logger.info("Finished testing Legendre series for libration point %s", i)
    logger.info("Finished testing Legendre series.")

def run_all_tests():
    test_legendre_series()


if __name__ == "__main__":
    run_all_tests()


