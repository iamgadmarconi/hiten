import numpy as np

from system.base import System, systemConfig
from system.body import Body

from algorithms.propagators import propagate_crtbp



def main():
    # Create the primary and secondary bodies
    Earth = Body(name="Earth", mass=5.972e24, radius=6378.137, color="blue")
    Moon = Body(name="Moon", mass=7.348e22, radius=1737.4, color="gray")
    Moon.parent = Earth

    # Create the system configuration
    config = systemConfig(primary=Earth, secondary=Moon, distance=384400)

    # Create the system
    system = System(config)

    # Get the libration points
    libration_point = system.get_libration_point(2)


    # Print the libration points
    libration_point.analyze_stability()
    print(libration_point.is_stable)
    print(libration_point.is_unstable)

    x0 = np.array([1.09380073, 0.0, -0.05629867, 0.0, 0.25074754, 0.0])
    t0 = 0.0
    tf = 1.5400986568902557
    mu = system.mu
    forward = 1
    steps = 1000
    solve_kwargs = {}

    sol = propagate_crtbp(x0, t0, tf, mu, forward, steps, **solve_kwargs)

    print(sol)
    


if __name__ == "__main__":
    main()





