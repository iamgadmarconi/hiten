from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Sequence

import numpy as np


if TYPE_CHECKING:
    from hiten.system.orbits.base import PeriodicOrbit


class _ContinuationOperatorsBase(Protocol):
    """Operations required for natural-parameter continuation.
    
    This protocol defines the minimal set of operations a continuation backend
    needs to perform predictor-corrector continuation without knowing about
    domain objects (orbits, manifolds), correction algorithms, or how to
    instantiate/represent solutions.
    
    Notes
    -----
    Continuation backends orchestrate:
    1. Prediction: extrapolate from last solution
    2. Correction: refine prediction to satisfy constraints
    3. Step control: adapt step size based on convergence
    
    The operators abstract domain instantiation and correction, keeping the
    backend focused on numerical continuation strategy.
    """

    @property
    def parameter_getter(self) -> Callable[[Any], float]:
        """Extract continuation parameter value from a domain object.
        
        Returns
        -------
        Callable[[Any], float]
            Function that takes a domain object and returns its parameter value.
        """
        ...

    def get_representation(self, domain_obj: Any) -> np.ndarray:
        """Convert domain object to numerical representation (state vector).
        
        Parameters
        ----------
        domain_obj : Any
            Domain object (e.g., PeriodicOrbit).
        
        Returns
        -------
        np.ndarray
            Numerical representation suitable for prediction/correction.
        """
        ...

    def instantiate_from_representation(self, representation: np.ndarray) -> Any:
        """Create domain object from numerical representation.
        
        Parameters
        ----------
        representation : np.ndarray
            Numerical representation (state vector).
        
        Returns
        -------
        Any
            Domain object instantiated from representation.
        
        Notes
        -----
        This is the inverse of get_representation. The instantiated object
        should be ready for correction but may not satisfy constraints yet.
        """
        ...

    def correct_prediction(self, prediction: np.ndarray) -> tuple[np.ndarray, float, bool] | tuple[np.ndarray, float, bool, dict]:
        """Correct a predicted solution to satisfy constraints.
        
        Parameters
        ----------
        prediction : np.ndarray
            Predicted state from continuation step.
        
        Returns
        -------
        x_corrected : np.ndarray
            Corrected state (converged or best attempt).
        residual : float
            Residual norm or correction error measure.
        converged : bool
            Whether correction converged successfully.
        aux : dict, optional
            Optional auxiliary data (e.g., period, eigenvalues, metrics).
            If present, returned as 4th element of tuple.
        
        Notes
        -----
        This wraps the domain-specific correction algorithm (e.g., Newton,
        multiple shooting) without the backend needing to know CR3BP,
        integrators, or constraint specifics.
        """
        ...

    def predict_step(self, last_repr: np.ndarray, step: np.ndarray) -> np.ndarray:
        """Predict next solution from last representation and step vector.
        
        Parameters
        ----------
        last_repr : np.ndarray
            Representation of last accepted solution.
        step : np.ndarray
            Step vector (direction and magnitude).
        
        Returns
        -------
        np.ndarray
            Predicted representation for next solution.
        
        Notes
        -----
        Simple predictor: last_repr + step (possibly with index mapping).
        More sophisticated predictors (secant, tangent) can be implemented
        by providing a different prediction strategy.
        """
        ...


class _OrbitContinuationOperators:
    """Concrete implementation of _ContinuationOperatorsBase for periodic orbits.
    
    This class adapts PeriodicOrbit domain objects to the abstract continuation
    operator protocol, handling orbit-specific instantiation, correction, and
    representation logic.
    
    Parameters
    ----------
    domain_obj : PeriodicOrbit
        The seed periodic orbit domain object.
    parameter_getter : Callable
        Function to extract continuation parameter from orbit.
    representation_fn : Callable or None
        Optional function to convert orbit to representation.
    state_indices : Sequence[int] or None
        Optional state indices for selective continuation.
    corrector_tol : float
        Tolerance for correction convergence.
    corrector_max_attempts : int
        Maximum correction attempts.
    corrector_max_delta : float
        Maximum correction step size.
    corrector_order : int
        Integration order for correction.
    corrector_steps : int
        Integration steps for correction.
    corrector_forward : int
        Integration direction for correction.
    corrector_fd_step : float
        Finite-difference step for correction.
    """

    def __init__(
        self,
        *,
        domain_obj: "PeriodicOrbit",
        parameter_getter: Callable[[Any], float],
        representation_fn: Optional[Callable[[Any], np.ndarray]],
        state_indices: Optional[Sequence[int]],
        corrector_tol: float,
        corrector_max_attempts: int,
        corrector_max_delta: float,
        corrector_order: int,
        corrector_steps: int,
        corrector_forward: int,
        corrector_fd_step: float,
    ):
        self._seed_orbit = domain_obj
        self._parameter_getter = parameter_getter
        self._representation_fn = representation_fn or (lambda obj: np.asarray(obj.initial_state, dtype=float))
        self._state_indices = None if state_indices is None else np.asarray(state_indices, dtype=int)
        self._corrector_tol = corrector_tol
        self._corrector_max_attempts = corrector_max_attempts
        self._corrector_max_delta = corrector_max_delta
        self._corrector_order = corrector_order
        self._corrector_steps = corrector_steps
        self._corrector_forward = corrector_forward
        self._corrector_fd_step = corrector_fd_step

    @property
    def parameter_getter(self) -> Callable[[Any], float]:
        return self._parameter_getter

    def get_representation(self, domain_obj: Any) -> np.ndarray:
        """Convert domain object to numerical representation."""
        return self._representation_fn(domain_obj)

    def instantiate_from_representation(self, representation: np.ndarray) -> Any:
        """Create orbit from numerical representation."""
        orbit_cls = type(self._seed_orbit)
        lp = getattr(self._seed_orbit, "libration_point", None)
        orbit = orbit_cls(
            libration_point=lp,
            initial_state=np.asarray(representation, dtype=float)
        )
        
        # Inherit seed period; post-processing will adjust if needed
        if self._seed_orbit.period is not None:
            orbit.period = self._seed_orbit.period
        
        return orbit

    def correct_prediction(
        self, prediction: np.ndarray
    ) -> tuple[np.ndarray, float, bool] | tuple[np.ndarray, float, bool, dict]:
        """Correct a predicted solution using orbit correction."""
        orbit = self.instantiate_from_representation(prediction)
        
        # Build correction options
        from hiten.algorithms.corrector.options import OrbitCorrectionOptions
        from hiten.algorithms.types.options import (ConvergenceOptions,
                                                    CorrectionOptions,
                                                    IntegrationOptions,
                                                    NumericalOptions)
        
        corrector_options = OrbitCorrectionOptions(
            base=CorrectionOptions(
                convergence=ConvergenceOptions(
                    tol=self._corrector_tol,
                    max_attempts=self._corrector_max_attempts,
                    max_delta=self._corrector_max_delta,
                ),
                integration=IntegrationOptions(
                    order=self._corrector_order,
                    steps=self._corrector_steps,
                ),
                numerical=NumericalOptions(
                    fd_step=self._corrector_fd_step,
                ),
            ),
            forward=self._corrector_forward,
        )
        
        # Perform correction
        corr_result = orbit.correct(options=corrector_options)
        x_corr = corr_result.x_corrected
        
        # Provide auxiliary metadata (period)
        aux = {"period": float(2.0 * getattr(corr_result, "half_period", np.nan))}
        
        # Compute residual as change from prediction
        residual = float(np.linalg.norm(np.asarray(x_corr, dtype=float) - prediction))
        
        return np.asarray(x_corr, dtype=float), residual, corr_result.converged, aux

    def predict_step(self, last_repr: np.ndarray, step: np.ndarray) -> np.ndarray:
        """Predict next solution from last representation and step vector."""
        if self._state_indices is None:
            # Simple prediction: last + step
            return np.asarray(last_repr, dtype=float) + np.asarray(step, dtype=float)
        
        # Selective prediction on specified indices
        last = np.asarray(last_repr, dtype=float).copy()
        step_arr = np.asarray(step, dtype=float)
        for idx, delta in zip(self._state_indices, step_arr):
            last[idx] += delta
        return last
