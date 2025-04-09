import numpy as np

from system.base import System, systemConfig
from system.body import Body

from algorithms.propagators import propagate_crtbp

from orbits._tests.test_lyapunov import run_all_tests


def main():
    run_all_tests()

if __name__ == "__main__":
    main()





