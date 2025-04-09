import numpy as np

from system.base import System, systemConfig
from system.body import Body

from algorithms.propagators import propagate_crtbp

from orbits._tests.test_lyapunov import test_lyapunov_orbit_propagation,test_lyapunov_orbit_plot


def main():
    test_lyapunov_orbit_propagation()
    test_lyapunov_orbit_plot()

if __name__ == "__main__":
    main()





