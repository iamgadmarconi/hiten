from system.base import System, systemConfig
from system.body import Body



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
    libration_point = system.get_libration_point(4)


    # Print the libration points
    libration_point.analyze_stability()
    print(libration_point.is_stable)
    print(libration_point.is_unstable)


if __name__ == "__main__":
    main()





