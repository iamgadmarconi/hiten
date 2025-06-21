import os
import pickle
from dataclasses import dataclass
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from algorithms.dynamics.rtbp import _propagate_crtbp, compute_stm
from algorithms.geometry import surface_of_section
from algorithms.linalg import _totime, eigenvalue_decomposition
from orbits.base import PeriodicOrbit
from plots.plots import _plot_body, _set_axes_equal
from utils.log_config import logger


@dataclass
class manifoldConfig:
    generating_orbit: PeriodicOrbit
    stable: bool = True
    direction: Literal["Positive", "Negative"] = "Positive"

    method: Literal["rk", "scipy", "symplectic", "adaptive"] = "scipy"
    order: int = 6


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
        self._forward = -self.stable
        self.mu = self.generating_orbit.system.mu
        self.method = config.method
        self.order = config.order
        self._successes = 0
        self._attempts = 0
        self.manifold_result: ManifoldResult = None

    def __str__(self):
        return f"Manifold(stable={self.stable}, direction={self.direction}) of {self.libration_point}-{self.generating_orbit}"
    
    def __repr__(self):
        return self.__str__()
    
    def compute(self, step: float = 0.02, integration_fraction: float = 0.75, **kwargs):

        if self.manifold_result is not None:
            return self.manifold_result

        kwargs.setdefault("show_progress", True)
        kwargs.setdefault("dt", 1e-2)

        # Stable manifolds (self.stable == 1) use backward integration (forward = -1)
        # Unstable manifolds (self.stable == -1) use forward integration (forward = 1)

        initial_state = self.generating_orbit._initial_state

        ysos, dysos, states_list, times_list = [], [], [], []

        # Sample the generating orbit *excluding* frac == 1.0 to avoid indexing
        # the STM at its last element which can raise.
        fractions = np.arange(0.0, 1.0, step)

        iterator = (
            tqdm(fractions, desc="Computing manifold")
            if kwargs["show_progress"]
            else fractions
        )

        for fraction in iterator:
            self._attempts += 1

            try:
                x0W = self._compute_manifold_section(
                    initial_state,
                    self.generating_orbit.period,
                    fraction,
                    forward=self._forward,
                )
                x0W = x0W.flatten().astype(np.float64)
                tf = integration_fraction * 2 * np.pi

                # Build signed time grid for the chosen integration direction
                dt = abs(kwargs["dt"])
                dt_signed = dt * self._forward
                t_vals = np.arange(0.0, self._forward * tf, dt_signed)
                
                # Calculate steps from the desired dt and integration time
                steps = max(int(abs(tf) / dt) + 1, 100)  # Ensure minimum steps

                # Integrate using the pre-configured dynamical system
                sol = _propagate_crtbp(
                    dynsys=self.generating_orbit.system._dynsys,
                    state0=x0W, 
                    t0=0.0, 
                    tf=tf,
                    forward=self._forward,  # Handle integration direction properly
                    steps=steps,
                    method=self.method, 
                    order=self.order
                )
                states, times = sol.states, sol.times

                states_list.append(states)
                times_list.append(times)

                Xy0, Ty0 = surface_of_section(states, times, self.mu, M=2, C=0)

                if len(Xy0) > 0:
                    # If intersection found, extract coordinates
                    Xy0 = Xy0.flatten()
                    ysos.append(Xy0[1])
                    dysos.append(Xy0[4])
                    self._successes += 1
                    logger.debug(f"Fraction {fraction:.3f}: Found Poincaré section point at y={Xy0[1]:.6f}, vy={Xy0[4]:.6f}")

            except Exception as e:
                err = f"Error computing manifold: {e}"
                logger.error(err)
                continue
        
        # Show summary warning if there were failed crossings
        if self._attempts > 0 and self._successes < self._attempts:
            failed_attempts = self._attempts - self._successes
            failure_rate = (failed_attempts / self._attempts) * 100
            logger.warning(f"Failed to find {failure_rate:.1f}% ({failed_attempts}/{self._attempts}) Poincaré section crossings")
            
        self.manifold_result = ManifoldResult(ysos, dysos, states_list, times_list, self._successes, self._attempts)
        return self.manifold_result

    def _compute_manifold_section(self, state: np.ndarray, period: float, fraction: float, NN: int = 1, forward: int = 1):
        xx, tt, phi_T, PHI = compute_stm(self.libration_point._var_eq_system, state, period, steps=1000, forward=forward, method=self.method, order=self.order)

        sn, un, _, Ws, Wu, _ = eigenvalue_decomposition(phi_T, discrete=1)

        snreal_vals = []
        snreal_vecs = []
        for k in range(len(sn)):
            if np.isreal(sn[k]):
                snreal_vals.append(sn[k])
                snreal_vecs.append(Ws[:, k])

        # 4) Collect real eigen-directions for unstable set
        unreal_vals = []
        unreal_vecs = []
        for k in range(len(un)):
            if np.isreal(un[k]):
                unreal_vals.append(un[k])
                unreal_vecs.append(Wu[:, k])

        snreal_vals = np.array(snreal_vals, dtype=np.complex128)
        unreal_vals = np.array(unreal_vals, dtype=np.complex128)
        snreal_vecs = (np.column_stack(snreal_vecs) 
                    if len(snreal_vecs) else np.zeros((6, 0), dtype=np.complex128))
        unreal_vecs = (np.column_stack(unreal_vecs) 
                    if len(unreal_vecs) else np.zeros((6, 0), dtype=np.complex128))

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
            logger.debug(f"Using stable manifold direction with eigenvalue {np.real(snreal_vals[col_idx]):.6f} for {NN}th eigenvector")

        elif self.stable == -1:
            MAN = self.direction * (phi_frac @ WU)
            logger.debug(f"Using unstable manifold direction with eigenvalue {np.real(unreal_vals[col_idx]):.6f} for {NN}th eigenvector")

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
        primary_radius = self.generating_orbit._system.primary.radius
        _plot_body(ax, primary_center, primary_radius / self.generating_orbit._system.distance, 'blue', self.generating_orbit._system.primary.name)

        secondary_center = np.array([(1 - mu), 0, 0])
        secondary_radius = self.generating_orbit._system.secondary.radius
        _plot_body(ax, secondary_center, secondary_radius / self.generating_orbit._system.distance, 'grey', self.generating_orbit._system.secondary.name)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        _set_axes_equal(ax)
        ax.set_title('Manifold')
        plt.show()

    def save(self, filepath: str, **kwargs) -> None:
        # Construct a serialisation dictionary – avoid non-pickle-friendly objects when possible.
        data = {
            "manifold_type": self.__class__.__name__,
            "stable": bool(self.stable == 1),
            "direction": "Positive" if self.direction == 1 else "Negative",
            "method": self.method,
            "order": self.order,
        }

        # Lightweight generating-orbit info (if available)
        try:
            data["generating_orbit"] = {
                "family": getattr(self.generating_orbit, "orbit_family", self.generating_orbit.__class__.__name__),
                "period": getattr(self.generating_orbit, "period", None),
                "initial_state": self.generating_orbit._initial_state.tolist(),
            }
        except Exception:
            # In case the attributes do not exist / are inaccessible just skip.
            pass

        # Store manifold Result if it exists
        if self.manifold_result is not None:
            mr = self.manifold_result
            data["manifold_result"] = {
                "ysos": mr.ysos,
                "dysos": mr.dysos,
                # numpy arrays need to be converted to (nested) lists for portability
                "states_list": [s.tolist() for s in mr.states_list],
                "times_list": [t.tolist() for t in mr.times_list],
                "_successes": mr._successes,
                "_attempts": mr._attempts,
            }
        else:
            data["manifold_result"] = None

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        with open(filepath, "wb") as fh:
            pickle.dump(data, fh)

        logger.info("Manifold saved to %s", filepath)

    def load(self, filepath: str, **kwargs) -> None:

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Manifold file not found: {filepath}")

        with open(filepath, "rb") as fh:
            data = pickle.load(fh)

        if data.get("manifold_type") != self.__class__.__name__:
            logger.warning(
                "Loading %s data into %s instance", data.get("manifold_type", "<unknown>"), self.__class__.__name__
            )

        # Update basic properties (do *not* overwrite integrator instance – keep what user supplied)
        self.stable = 1 if data.get("stable", True) else -1
        self.direction = 1 if data.get("direction", "Positive") == "Positive" else -1

        # Store generating-orbit metadata for reference
        self._loaded_generating_orbit_info = data.get("generating_orbit", {})

        # Re-create manifold_result if present
        mr_data = data.get("manifold_result")
        if mr_data is not None:
            ysos = mr_data["ysos"]
            dysos = mr_data["dysos"]
            states_list = [np.array(s, dtype=float) for s in mr_data["states_list"]]
            times_list = [np.array(t, dtype=float) for t in mr_data["times_list"]]
            _successes = mr_data.get("_successes", 0)
            _attempts = mr_data.get("_attempts", 0)
            self.manifold_result = ManifoldResult(ysos, dysos, states_list, times_list, _successes, _attempts)
        else:
            self.manifold_result = None

        logger.info("Manifold loaded from %s", filepath)
