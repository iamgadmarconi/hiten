"""Example script: computing the centre manifold Hamiltonian for the Earth-Moon hiten.system.

Run with
    python examples/center_manifold.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten import System
from hiten.algorithms import PoincareMap


def main() -> None:
    """Compute and display the centre-manifold Hamiltonian."""
    system = System.from_bodies("sun", "earth")
    l_point = system.get_libration_point(4)
    cm = l_point.get_center_manifold(max_degree=2)
    cm.compute()
    cm.coefficients()
    pm = PoincareMap(cm, 0.6)
    pm.compute()
    pm.plot()

if __name__ == "__main__":
    main() 