"""Example script: computing the centre manifold Hamiltonian for the Earth-Moon hiten.system.

Run with
    python examples/center_manifold.py
"""

import os
import sys

# Add the project src directory to the Python path so that absolute imports work when
# the script is executed from the project root.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from hiten.system import Body, CenterManifold, System, systemConfig
from hiten.utils import Constants, format_cm_table
from hiten.utils.log_config import logger


def main() -> None:
    """Compute and display the centre-manifold Hamiltonian."""

    # Build the CRTBP system for the selected primary / secondary bodies
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

    # Locate the libration point and build the centre manifold object
    l_point = system.get_libration_point(1)
    logger.info("Computing centre manifold around L%s of the %s-%s system…", 1, "Earth", "Moon")

    cm = CenterManifold(l_point, 10)
    cm_H = cm.compute()

    logger.info(
        "\nCentre-manifold Hamiltonian (deg 2 → %d) in real NF variables (q₂, p₂, q₃, p₃)\n",
        10,
    )
    logger.info("\n%s\n", format_cm_table(cm_H, cm.clmo))


if __name__ == "__main__":
    main() 