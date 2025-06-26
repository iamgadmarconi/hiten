"""Example script: generating and displaying a Poincaré map for the Earth-Moon hiten.system.

python examples/poincare_map.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.system import System
from hiten.utils.log_config import logger


def main() -> None:
    """Generate and interactively display a Poincaré map."""

    system = System.from_bodies("earth", "moon")

    l_point = system.get_libration_point(1)
    logger.info("Generating Poincaré map for L%s of the %s-%s system...", 1, "Earth", "Moon")
    cm = l_point.get_center_manifold(max_degree=6)
    cm.compute()

    pm = cm.poincare_map(
        energy=0.7,
        section_coord="p3",
        n_seeds=50,
        n_iter=100,
        seed_strategy="axis_aligned"
    )

    pm.plot(axes=("p2", "q3"))


if __name__ == "__main__":
    main() 