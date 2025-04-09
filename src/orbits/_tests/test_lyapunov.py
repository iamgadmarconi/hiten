from src.system.body import Body
from src.system.base import System, systemConfig
from src.orbits.base import orbitConfig
from src.orbits.lyapunov import LyapunovOrbit
from src.utils.constants import Constants


def test_lyapunov_orbit_ic():

    earth_mass = Constants.get_mass("earth")
    earth_radius = Constants.get_radius("earth")
    moon_mass = Constants.get_mass("moon")
    moon_radius = Constants.get_radius("moon")
    distance = Constants.get_distance("earth", "moon")

    earth = Body("Earth", earth_mass, earth_radius, color="blue")
    moon = Body("Moon", moon_mass, moon_radius, color="gray", parent=earth)

    system = System(systemConfig(primary=earth, secondary=moon, distance=distance))


    config = orbitConfig(
        system=system,
        orbit_family="lyapunov",
        libration_point_idx=2,
        extra_params={"Ax": 1e-5}
    )

    orbit = LyapunovOrbit(config)
    print(orbit.initial_state)




if __name__ == "__main__":
    test_lyapunov_orbit_ic()
