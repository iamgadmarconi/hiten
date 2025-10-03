"""Base types and protocols for the linear algebra module."""

from typing import TYPE_CHECKING, Generic, Optional, Tuple

import numpy as np

from hiten.algorithms.linalg.backend import _LinalgBackend
from hiten.algorithms.linalg.engine import _LinearStabilityEngine
from hiten.algorithms.linalg.interfaces import _EigenDecompositionInterface
from hiten.algorithms.linalg.types import EigenDecompositionResults
from hiten.algorithms.types.core import (ConfigT, DomainT, InterfaceT, ResultT,
                                         _HitenBasePipeline)

if TYPE_CHECKING:
    from hiten.algorithms.linalg.options import EigenDecompositionOptions


class StabilityPipeline(_HitenBasePipeline, Generic[DomainT, InterfaceT, ConfigT, ResultT]):
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

    def __init__(self, config: ConfigT, engine: _LinearStabilityEngine, interface: InterfaceT = None, backend: _LinalgBackend = None) -> None:
        super().__init__(config, engine, interface, backend)

    @classmethod
    def with_default_engine(cls, *, config: ConfigT, interface: Optional[InterfaceT] = None, backend: Optional[_LinalgBackend] = None) -> "StabilityPipeline[DomainT, InterfaceT, ConfigT, ResultT]":
        """Create a facade instance with a default engine (factory).

        Parameters
        ----------
        config : :class:`~hiten.algorithms.types.core.ConfigT`
            Configuration object.
        interface : :class:`~hiten.algorithms.types.InterfaceT`
            Interface object.
        backend : :class:`~hiten.algorithms.linalg.backend._LinalgBackend`, optional
            Backend object. If None, uses the default _LinalgBackend.
        Returns
        -------
        :class:`~hiten.algorithms.linalg.base.StabilityPipeline`
            A stability pipeline instance with a default engine injected.
        """
        backend = backend or _LinalgBackend()
        intf = interface or _EigenDecompositionInterface()
        engine = _LinearStabilityEngine(backend=backend, interface=intf)
        return cls(config, engine, intf, backend)

    def compute(
        self,
        domain_obj: DomainT,
        options: Optional["EigenDecompositionOptions"] = None,
    ) -> EigenDecompositionResults:
        """Compose a problem from domain_obj and run the engine.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.algorithms.types.DomainT`
            Domain object.
        options : :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`, optional
            Runtime options for eigenvalue decomposition. If None, uses defaults.

        Returns
        -------
        :class:`~hiten.algorithms.linalg.types.EigenDecompositionResults`
            Eigen decomposition results.
        """
        problem = self._create_problem(domain_obj=domain_obj, options=options)
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
