"""Example script: computing the centre manifold Hamiltonian for the Earth-Moon hiten.system.

Run with
    python examples/center_manifold.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import System


def main() -> None:
    """Compute and display the centre-manifold Hamiltonian."""
    system = System.from_bodies("sun", "earth")
    l_point = system.get_libration_point(1)
    cm = l_point.get_center_manifold(max_degree=10)
    cm.compute()

    cm.coefficients()

if __name__ == "__main__":
    main() 