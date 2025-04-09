import numpy as np

from system.base import System, systemConfig
from system.body import Body

from algorithms.propagators import propagate_crtbp

from orbits._tests.test_lyapunov import test_lyapunov_differential_correction


def main():
    test_lyapunov_differential_correction()

if __name__ == "__main__":
    main()





