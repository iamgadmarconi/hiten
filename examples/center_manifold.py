"""Example script: computing the centre manifold Hamiltonian for the Earth–Moon system.

Run with
    python examples/center_manifold_example.py
"""

import os
import sys

# Add the project src directory to the Python path so that absolute imports work when
# the script is executed from the project root.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from algorithms.center.base import CenterManifold
from config import L_POINT, MAX_DEG, PRIMARY, SECONDARY
from system.base import System, systemConfig
from system.body import Body
from utils.constants import Constants
from utils.log_config import logger
from utils.printing import format_cm_table


def main() -> None:
    """Compute and display the centre-manifold Hamiltonian."""

    # Build the CRTBP system for the selected primary / secondary bodies
    primary = Body(
        PRIMARY,
        Constants.bodies[PRIMARY.lower()]["mass"],
        Constants.bodies[PRIMARY.lower()]["radius"],
        "blue",
    )
    secondary = Body(
        SECONDARY,
        Constants.bodies[SECONDARY.lower()]["mass"],
        Constants.bodies[SECONDARY.lower()]["radius"],
        "gray",
        primary,
    )
    system = System(
        systemConfig(
            primary,
            secondary,
            Constants.get_orbital_distance(PRIMARY, SECONDARY),
        )
    )

    # Locate the libration point and build the centre manifold object
    l_point = system.get_libration_point(L_POINT)
    logger.info("Computing centre manifold around L%s of the %s–%s system…", L_POINT, PRIMARY, SECONDARY)

    cm = CenterManifold(l_point, MAX_DEG)
    cm_H = cm.compute()

    logger.info(
        "\nCentre-manifold Hamiltonian (deg 2 → %d) in real NF variables (q₂, p₂, q₃, p₃)\n",
        MAX_DEG,
    )
    logger.info("\n%s\n", format_cm_table(cm_H, cm.clmo))


if __name__ == "__main__":
    main() 