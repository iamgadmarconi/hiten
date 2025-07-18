"""Example script: computing the centre manifold Hamiltonian for the Earth-Moon hiten.system.

Run with
    python examples/center_manifold.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import System


def main() -> None:
    """Compute and display the centre-manifold Hamiltonian."""
    system = System.from_bodies("earth", "moon")
    l_point = system.get_libration_point(5)
    print(l_point.linear_data)



if __name__ == "__main__":
    main() 