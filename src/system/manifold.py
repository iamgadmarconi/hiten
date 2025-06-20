from dataclasses import dataclass
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from algorithms.dynamics.rtbp import compute_stm, create_rtbp_system
from algorithms.geometry import surface_of_section
from algorithms.integrators.rk import RungeKutta
from algorithms.integrators.symplectic import TaoSymplectic
from algorithms.linalg import _totime, eigenvalue_decomposition
from orbits.base import PeriodicOrbit
from plots.plots import _plot_body, _set_axes_equal
from utils.log_config import logger


@dataclass
class manifoldConfig:
    generating_orbit: PeriodicOrbit
    stable: bool = True
    direction: Literal["Positive", "Negative"] = "Positive"

    method: str = "rk8"


@dataclass
class ManifoldResult:
    ysos: List[float]
    dysos: List[float]
    states_list: List[float]
    times_list: List[float]
    _successes: int
    _attempts: int

    @property
    def success_rate(self) -> float:
        return self._successes / max(self._attempts, 1)
    
    def __iter__(self):
        return iter((self.ysos, self.dysos, self.states_list, self.times_list))


class Manifold:

    def __init__(self, config: manifoldConfig):
        self.generating_orbit = config.generating_orbit
        self.libration_point = self.generating_orbit.libration_point
        self.stable = 1 if config.stable else -1
        self.direction = 1 if config.direction == "Positive" else -1
        self.mu = self.generating_orbit.mu


        if config.method.lower().startswith("rk"):
            order = int(config.method.lower()[2:])
            self._integrator = RungeKutta(order=order)
        elif config.method.lower().startswith("symp"):
            order = int(config.method.lower()[4:])
            self._integrator = TaoSymplectic(order=order)
        else:
            raise ValueError(f"Unknown integration method '{config.method}'.")
        
        self._dynsys = create_rtbp_system(mu=self.mu)
        self._successes = 0
        self._attempts = 0

        self.manifold_result: ManifoldResult = None

    def __str__(self):
        return f"Manifold(stable={self.stable}, direction={self.direction}) of {self.libration_point}-{self.generating_orbit}"
    
    def __repr__(self):
        return self.__str__()
    
    def compute(self, step: float=0.02, integration_fraction: float=0.75, forward: bool = True, **kwargs):

        if self.manifold_result is not None:
            return self.manifold_result

        kwargs.setdefault('show_progress', True)
        kwargs.setdefault('dt', 1e-5)

        initial_state = self.generating_orbit._initial_state

        ysos, dysos, states_list, times_list = [], [], [], []

        fractions = np.arange(0, 1 + step, step)

        iterator = tqdm(fractions, desc="Computing manifold") if kwargs['show_progress'] else fractions

        for fraction in iterator:
            self._attempts += 1

            try:
                x0W = self._compute_manifold_section(initial_state, self.generating_orbit.period, fraction, forward=forward)
                x0W = x0W.flatten().astype(np.float64)
                tf = integration_fraction * 2 * np.pi

                sol = self._integrator.integrate(self._dynsys, x0W, np.arange(0, tf, kwargs['dt']))
                states, times = sol.states, sol.times

                states_list.append(states)
                times_list.append(times)

                Xy0, Ty0 = surface_of_section(states, times, self.mu, M=2, C=1)

                if len(Xy0) > 0:
                    # If intersection found, extract coordinates
                    Xy0 = Xy0.flatten()
                    ysos.append(Xy0[1])
                    dysos.append(Xy0[4])
                    self._successes += 1
                    logger.debug(f"Fraction {fraction:.3f}: Found Poincaré section point at y={Xy0[1]:.6f}, vy={Xy0[4]:.6f}")
                else:
                    logger.warning(f"No section points found for fraction {fraction:.3f}")
                    pass

            except Exception as e:
                err = f"Error computing manifold: {e}"
                logger.error(err)
                continue
            
        self.manifold_result = ManifoldResult(ysos, dysos, states_list, times_list, self._successes, self._attempts)
        return self.manifold_result

    def _compute_manifold_section(self, state: np.ndarray, period: float, fraction: float, forward: int = 1, NN: int = 1):
        xx, tt, phi_T, PHI = compute_stm(state, self.mu, period, forward=forward)

        _, eig_vals, eig_vecs = eigenvalue_decomposition(phi_T, discrete=1)

        stable_eig_vals, stable_eig_vecs = [], []
        unstable_eig_vals, unstable_eig_vecs = [], []

        for val, vec in zip(eig_vals, eig_vecs.T):
            if np.isreal(val) and not np.isclose(np.abs(val), 1.0):
                if np.abs(val) < 1.0:
                    stable_eig_vals.append(np.real(val))
                    stable_eig_vecs.append(np.real(vec))
                elif np.abs(val) > 1.0:
                    unstable_eig_vals.append(np.real(val))
                    unstable_eig_vecs.append(np.real(vec))

        snreal_vals = np.array(stable_eig_vals)
        unreal_vals = np.array(unstable_eig_vals)
        snreal_vecs = np.column_stack(stable_eig_vecs) if stable_eig_vecs else np.zeros((6, 0))
        unreal_vecs = np.column_stack(unstable_eig_vecs) if unstable_eig_vecs else np.zeros((6, 0))

        col_idx = NN - 1

        if self.stable == 1 and (snreal_vecs.shape[1] <= col_idx or col_idx < 0):
            raise ValueError(f"Requested stable eigenvector {NN} not available. Only {snreal_vecs.shape[1]} real stable eigenvectors found.")
        
        if self.stable == -1 and (unreal_vecs.shape[1] <= col_idx or col_idx < 0):
            raise ValueError(f"Requested unstable eigenvector {NN} not available. Only {unreal_vecs.shape[1]} real unstable eigenvectors found.")

        WS = snreal_vecs[:, col_idx] if self.stable == 1 else None
        WU = unreal_vecs[:, col_idx] if self.stable == -1 else None

        mfrac = _totime(tt, fraction * period)
        
        if np.isscalar(mfrac):
            mfrac_idx = mfrac
        else:
            mfrac_idx = mfrac[0]

        phi_frac_flat = PHI[mfrac_idx, :36]
        phi_frac = phi_frac_flat.reshape((6, 6))

        if self.stable == 1:
            MAN = self.direction * (phi_frac @ WS)
            logger.debug(f"Using stable manifold direction with eigenvalue {snreal_vals[col_idx]:.6f} for {NN}th eigenvector")

        if self.stable == -1:
            MAN = self.direction * (phi_frac @ WU)
            logger.debug(f"Using unstable manifold direction with eigenvalue {unreal_vals[col_idx]:.6f} for {NN}th eigenvector")

        disp_magnitude = np.linalg.norm(MAN[0:3])

        if disp_magnitude < 1e-14:
            logger.warning(f"Very small displacement magnitude: {disp_magnitude:.2e}, setting to 1.0")
            disp_magnitude = 1.0
        d = 1e-6 / disp_magnitude

        fracH = xx[mfrac_idx, :].copy()

        x0W = fracH + d * MAN.real
        x0W = x0W.flatten()
        
        if abs(x0W[2]) < 1.0e-15:
            x0W[2] = 0.0
        if abs(x0W[5]) < 1.0e-15:
            x0W[5] = 0.0

        return x0W
    
    def plot(self):
        if self.manifold_result is None:
            err = "Manifold result not computed. Please compute the manifold first."
            logger.error(err)
            raise ValueError(err)


        states_list, times_list = self.manifold_result.states_list, self.manifold_result.times_list

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Use a colormap to assign each trajectory a unique color
        num_traj = len(states_list)
        cmap = plt.get_cmap('plasma')
        for i, (xW, _) in enumerate(zip(states_list, times_list)):
            # Normalize index to range 0-1 for the colormap
            color = cmap(i / (num_traj - 1)) if num_traj > 1 else cmap(0.5)
            ax.plot(xW[:, 0], xW[:, 1], xW[:, 2], color=color, lw=2)

        mu = self.mu

        primary_center = np.array([-mu, 0, 0])
        primary_radius = self.generating_orbit._system.bodies[0].radius
        _plot_body(ax, primary_center, primary_radius / self.generating_orbit._system.distance, 'blue', self.generating_orbit._system.bodies[0].name)

        secondary_center = np.array([(1 - mu), 0, 0])
        secondary_radius = self.generating_orbit._system.bodies[1].radius
        _plot_body(ax, secondary_center, secondary_radius / self.generating_orbit._system.distance, 'grey', self.generating_orbit._system.bodies[1].name)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        _set_axes_equal(ax)
        ax.set_title('Manifold')
        plt.show()
