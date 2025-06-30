import numpy as np
import pytest

from hiten.algorithms.bifurcation.analysis import _equilibrium_with_energy
from hiten.algorithms.bifurcation.base import _BifurcationEngine
from hiten.algorithms.bifurcation.transforms import _realcenter2actionangle
from hiten.system import System


def _build_fourier_hamiltonian(max_degree: int = 4):

    system = System.from_bodies("sun", "earth")
    l1 = system.get_libration_point(1)

    # A low-order normal form is enough for the purposes of this test and makes
    # CI faster.
    cm = l1.get_center_manifold(max_degree=max_degree)
    poly_cm_real = cm.compute()  # list[np.ndarray] in (q,p) variables

    # Convert to action-angle Fourier/Taylor form.
    fourier_coeffs, _, clmoF, _ = _realcenter2actionangle(poly_cm_real, cm._clmo)
    return fourier_coeffs, clmoF


def _equilibrium_at_energy(coeffs_list, clmoF, energy_target: float, max_tries: int = 5):

    for scale in np.geomspace(1e-3, 1e-1, num=max_tries):
        # Simple heuristic: put all the action in the first planar mode (I₂)
        I_guess = np.array([0.0, scale, 0.0])
        theta_guess = np.zeros(3)
        vec_guess = np.concatenate((I_guess, theta_guess, [0.0]))

        I_eq, th_eq, ok, _ = _equilibrium_with_energy(
            coeffs_list, clmoF, vec_guess, target_energy=energy_target, tol=1e-10
        )
        if ok:
            return I_eq, th_eq

    raise RuntimeError("Could not converge to an equilibrium at the requested energy level.")


def test_halo_bifurcation_detection():

    coeffs_list, clmoF = _build_fourier_hamiltonian(max_degree=6)

    H0 = float(np.real_if_close(coeffs_list[0][0]))

    # Start exactly at the origin (I=0) where energy = H0
    I_seed = np.zeros(3)
    th_seed = np.zeros(3)

    engine = _BifurcationEngine(
        coeffs_list=coeffs_list,
        clmoF=clmoF,
        I_theta_seed=np.concatenate((I_seed, th_seed)),
        energy_seed=H0,
        energy_target=(H0, H0 + 0.45),
        step=1e-3,
        max_points=600,
    )

    events = engine.run()

    rel_energies = [ev["energy"] - H0 for ev in events]

    assert any(0.2 < rel_e < 0.4 for rel_e in rel_energies), (
        "Lyapunov-Halo bifurcation not detected within the expected energy interval"
    )
