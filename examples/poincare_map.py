"""Example script: generating and displaying a Poincaré map for the Earth-Moon hiten.system.

python examples/poincare_map.py
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.system import Body, CenterManifold, System, systemConfig
from hiten.utils import Constants
from hiten.utils.log_config import logger


def main() -> None:
    """Generate and interactively display a Poincaré map."""

    primary = Body(
        "Earth",
        Constants.bodies["earth"]["mass"],
        Constants.bodies["earth"]["radius"],
        "blue",
    )
    secondary = Body(
        "Moon",
        Constants.bodies["moon"]["mass"],
        Constants.bodies["moon"]["radius"],
        "gray",
        primary,
    )
    system = System(
        systemConfig(
            primary,
            secondary,
            Constants.get_orbital_distance("Earth", "Moon"),
        )
    )

    l_point = system.get_libration_point(1)
    logger.info("Generating Poincaré map for L%s of the %s-%s system…", 1, "Earth", "Moon")

    cm = CenterManifold(l_point, 10)
    cm.compute()

    pm = cm.poincare_map(
        0.6,
        seed_axis="q2",
        section_coord="q3",
        n_seeds=20,
        n_iter=25,
        use_gpu=False,
    )

    pm.plot_interactive()


if __name__ == "__main__":
    main() 