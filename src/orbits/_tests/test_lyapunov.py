from system.body import Body
from system.base import System, systemConfig
from orbits.base import orbitConfig
from orbits.lyapunov import LyapunovOrbit
from utils.constants import Constants


def test_lyapunov_orbit_ic():
    print("Testing Lyapunov orbit initial condition generation...")
    earth_mass = Constants.get_mass("earth")
    earth_radius = Constants.get_radius("earth")
    moon_mass = Constants.get_mass("moon")
    moon_radius = Constants.get_radius("moon")
    distance = Constants.get_orbital_distance("earth", "moon")

    earth = Body("Earth", earth_mass, earth_radius, color="blue")
    moon = Body("Moon", moon_mass, moon_radius, color="gray", parent=earth)

    system = System(systemConfig(primary=earth, secondary=moon, distance=distance))

    for i in range(1, 6):
        config = orbitConfig(
            system=system,
            orbit_family="lyapunov",
            libration_point_idx=i,
            extra_params={"Ax": 1e-5}
        )

        orbit = LyapunovOrbit(config)
        print(orbit.initial_state)

    print("Finished testing Lyapunov orbit initial condition generation.")


if __name__ == "__main__":
    test_lyapunov_orbit_ic()
