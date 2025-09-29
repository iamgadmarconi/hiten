"""Base types and protocols for the linear algebra module."""

from typing import Generic, Optional, Tuple

import numpy as np

from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.linalg.engine import _LinearStabilityEngine
from hiten.algorithms.linalg.interfaces import _EigenDecompositionInterface
from hiten.algorithms.linalg.types import (EigenDecompositionResults,
                                           _ProblemType, _SystemType)
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBaseFacade)


class StabilityPipeline(_HitenBaseFacade, Generic[DomainT, InterfaceT, ConfigT, ResultT]):
    """Facade exposing linear stability results on demand.
    
    Parameters
    ----------
    config : :class:`~hiten.algorithms.types.core.ConfigT`
        Configuration object.
    interface : :class:`~hiten.algorithms.types.InterfaceT`
        Interface object.
    engine : :class:`~hiten.algorithms.linalg.engine._LinearStabilityEngine`
        Engine object.
    """

    def __init__(self, config: ConfigT, interface: InterfaceT, engine: _LinearStabilityEngine = None) -> None:
        super().__init__(config, interface, engine)

    @classmethod
    def with_default_engine(cls, *, config: ConfigT, interface: Optional[InterfaceT] = None) -> "StabilityPipeline[DomainT, InterfaceT, ConfigT, ResultT]":
        """Create a facade instance with a default engine (factory).

        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
            Configuration object.
        interface : :class:`~hiten.algorithms.types.InterfaceT`
            Interface object.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.base.StabilityPipeline`
            A stability pipeline instance with a default engine injected.
        """
        backend = _LinalgBackend()
        intf = interface or _EigenDecompositionInterface()
        engine = _LinearStabilityEngine(backend=backend, interface=intf)
        return cls(config, intf, engine)

    def compute(
        self,
        domain_obj: DomainT,
        override: bool = False,
        *,
        system_type: Optional[_SystemType] = None,
        problem_type: Optional[_ProblemType] = None,
        delta: Optional[float] = None,
        tol: Optional[float] = None,
    ) -> EigenDecompositionResults:
        """Compose a problem from domain_obj and run the engine.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.algorithms.types.DomainT`
            Domain object.
        override : bool
            Whether to override configuration with provided kwargs.
        system_type : :class:`~hiten.algorithms.linalg.types._SystemType`
            System type.
        problem_type : :class:`~hiten.algorithms.linalg.types._ProblemType`
            Problem type.
        delta : float
            Delta.
        tol : float
            Tolerance.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.types.EigenDecompositionResults`
            Eigen decomposition results.
        """
        kwargs = {
            "system_type": system_type,
            "problem_type": problem_type,
            "delta": delta,
            "tol": tol,
        }

        problem = self._create_problem(domain_obj=domain_obj, override=override, **kwargs)
        engine = self._get_engine()
        self._results = engine.solve(problem)
        return self._results

    @property
    def is_stable(self) -> bool:
        """Check if the system is stable.
        
        Returns
        -------
        bool
            True if the system is stable, False otherwise.
        """
        result = self._require_result()
        return len(result.unstable) == 0

    @property
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the eigenvalues.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Stable eigenvalues, unstable eigenvalues, and center eigenvalues.
        """
        result = self._require_result()
        return result.stable, result.unstable, result.center    

    @property
    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the eigenvectors.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Stable eigenvectors, unstable eigenvectors, and center eigenvectors.
        """
        result = self._require_result()
        return result.Ws, result.Wu, result.Wc

    def get_real_eigenvectors(self, vectors: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get the real eigenvectors.
        
        Parameters
        ----------
        vectors : np.ndarray
            Eigenvectors.
        values : np.ndarray
            Eigenvalues.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Real eigenvalues and eigenvectors.

        Raises
        ------
        ValueError
            If the eigenvalues are not real.
        """
        mask = np.isreal(values)
        real_vals_arr = values[mask].astype(np.complex128)
        if np.any(mask):
            real_vecs_arr = vectors[:, mask]
        else:
            real_vecs_arr = np.zeros((vectors.shape[0], 0), dtype=np.complex128)
        return real_vals_arr, real_vecs_arr

    def _validate_config(self, config: ConfigT) -> None:
        """Validate the configuration object.
        
        This method can be overridden by concrete facades to perform
        domain-specific configuration validation.
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
            The configuration object to validate.
            
        Raises
        ------
        ValueError
            If the configuration is invalid.
        """
        super()._validate_config(config)
        
        if hasattr(config, 'tol') and config.tol <= 0:
            raise ValueError("Tolerance must be positive")
        if hasattr(config, 'delta') and config.delta <= 0:
            raise ValueError("Delta must be positive")
        if hasattr(config, 'system_type') and config.system_type is None:
            raise ValueError("System type must be specified")
        if hasattr(config, 'problem_type') and config.problem_type is None:
            raise ValueError("Problem type must be specified")

    def _require_result(self) -> EigenDecompositionResults:
        """Require the results.
        
        Returns
        -------
        :class:`~hiten.algorithms.linalg.types.EigenDecompositionResults`
            Eigen decomposition results.

        Raises
        ------
        ValueError
            If the results are not computed.
        """
        if self._results is None:
            raise ValueError("Stability results not computed; call compute() first")
        return self._results
