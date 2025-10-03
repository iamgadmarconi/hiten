"""Adapters supporting persistence and numerics for libration points."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np

from hiten.algorithms.common.energy import crtbp_energy, energy_to_jacobi
from hiten.algorithms.dynamics.base import _DynamicalSystem
from hiten.algorithms.dynamics.hamiltonian import _HamiltonianSystem
from hiten.algorithms.linalg.base import StabilityPipeline
from hiten.algorithms.linalg.config import EigenDecompositionConfig
from hiten.algorithms.linalg.interfaces import _LibrationPointInterface
from hiten.algorithms.linalg.types import _ProblemType, _SystemType
from hiten.algorithms.types.services.base import (_DynamicsServiceBase,
                                                  _PersistenceServiceBase,
                                                  _ServiceBundleBase)
from hiten.algorithms.utils.rootfinding import (expand_bracket,
                                                solve_bracketed_brent)
from hiten.system.center import CenterManifold
from hiten.system.hamiltonian import Hamiltonian, LieGeneratingFunction
from hiten.utils.io.libration import (load_libration_point,
                                      load_libration_point_inplace,
                                      save_libration_point)
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.algorithms.linalg.options import EigenDecompositionOptions
    from hiten.system.base import System
    from hiten.system.libration.base import LibrationPoint
    from hiten.system.libration.collinear import (CollinearPoint, L1Point,
                                                  L2Point, L3Point)
    from hiten.system.libration.triangular import (L4Point, L5Point,
                                                   TriangularPoint)
    from hiten.system.orbits.base import PeriodicOrbit


class _LibrationPersistenceService(_PersistenceServiceBase):
    """Encapsulate libration point IO helpers for testability.
    
    Parameters
    ----------
    save_fn : Callable[..., Any]
        The function to save the object.
    load_fn : Callable[..., Any]
        The function to load the object.
    load_inplace_fn : Callable[..., Any]
        The function to load the object in place.
    """

    def __init__(self) -> None:
        super().__init__(
            save_fn=lambda point, path, **kw: save_libration_point(point, Path(path), **kw),
            load_fn=lambda path, **kw: load_libration_point(Path(path), **kw),
            load_inplace_fn=lambda target, path, **kw: load_libration_point_inplace(target, Path(path), **kw),
        )


class _LibrationDynamicsService(_DynamicsServiceBase):
    """Provide stability analysis and geometry helpers for libration points.
    
    Parameters
    ----------
    point : :class:`~hiten.system.libration.base.LibrationPoint`
        The libration point.

    Attributes
    ----------
    generator : :class:`~hiten.algorithms.linalg.base.StabilityPipeline`
        The stability pipeline.
    """

    def __init__(self, point: "LibrationPoint") -> None:
        super().__init__(point)
        self._generator = None
        self._eigendecomposition_config = None
        self._eigendecomposition_options = None

    @property
    def generator(self) -> StabilityPipeline:
        """The stability pipeline."""
        if self._generator is None:
            self._generator = StabilityPipeline.with_default_engine(
                config=self.eigendecomposition_config, interface=_LibrationPointInterface())
        return self._generator

    @property
    def system(self) -> "System":
        """The system."""
        return self.domain_obj.system

    @property
    def mu(self) -> float:
        """The mass parameter."""
        return self.system.mu

    @property
    def dynsys(self) -> _DynamicalSystem:
        """The dynamical system."""
        return self.system.dynsys
    
    @property
    def var_dynsys(self) -> _DynamicalSystem:
        """The variational equations system."""
        return self.system.var_dynsys

    def hamsys(self, degree: int, form: str = "center_manifold_real") -> _HamiltonianSystem:
        """Get the Hamiltonian system for the given form and degree.
        
        Parameters
        ----------
        degree : int
            The maximum degree of the Hamiltonian expansion.
        form : str
            The Hamiltonian form to get coefficients for. Default is "center_manifold_real".
            
        Returns
        -------
        :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem`
            The Hamiltonian system instance.
        """
        cache_key = self.make_key(id(self.domain_obj), "hamsys", degree, form)
        
        def _factory() -> _HamiltonianSystem:
            # Get the Hamiltonian object for the specified form and degree
            hamiltonian = self.hamiltonian(degree, form)
            return hamiltonian.hamsys
        
        return self.get_or_create(cache_key, _factory)

    @property
    def jacobian_dynsys(self) -> _DynamicalSystem:
        """The Jacobian evaluation system."""
        return self.system.jacobian_dynsys

    @property
    def is_stable(self) -> bool:
        """The stability of the libration point."""
        return self.compute_stability().is_stable

    @property
    def eigenvalues(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The eigenvalues of the libration point."""
        return self.compute_stability().eigenvalues
    
    @property
    def eigenvectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The eigenvectors of the libration point."""
        return self.compute_stability().eigenvectors
    
    @property
    def energy(self) -> float:
        """The energy of the libration point.
        
        Returns
        -------
        float
            The mechanical energy in nondimensional units.
        """
        cache_key = self.make_key(id(self.domain_obj), "energy")
        
        def _factory() -> float:
            state = np.concatenate([self.domain_obj.position, np.array([0.0, 0.0, 0.0])])
            return crtbp_energy(state, self.mu)
        
        return self.get_or_create(cache_key, _factory)

    @property
    def jacobi(self) -> float:
        """Compute the Jacobi constant of the libration point.
        
        Returns
        -------
        float
            The Jacobi constant in nondimensional units.
        """
        cache_key = self.make_key(id(self.domain_obj), "jacobi_constant")
        
        def _factory() -> float:
            return energy_to_jacobi(self.energy)
        
        return self.get_or_create(cache_key, _factory)

    def compute_stability(self, options: "EigenDecompositionOptions" = None) -> StabilityPipeline:
        """Compute the stability of the libration point.
        
        Parameters
        ----------
        options : :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`, optional
            Runtime options for eigenvalue decomposition. If None, uses self.eigendecomposition_options.
            
        Returns
        -------
        :class:`~hiten.algorithms.linalg.base.StabilityPipeline`
            The stability pipeline with computed eigenvalue decomposition.
        """
        # Use self.eigendecomposition_options if options not provided
        if options is None:
            options = self.eigendecomposition_options
            
        cache_key = self.make_key(id(self.domain_obj), tuple(sorted(options.to_dict().items())))

        def _factory() -> StabilityPipeline:
            self.generator.compute(self.domain_obj, options=options)
            return self.generator

        return self.get_or_create(cache_key, _factory)

    def center_manifold(self, degree: int) -> CenterManifold:
        """Get or create a center manifold of the specified degree.
        
        Parameters
        ----------
        degree : int
            The maximum degree of the center manifold expansion.
            
        Returns
        -------
        :class:`~hiten.system.center.CenterManifold`
            The center manifold instance.
        """
        cache_key = self.make_key(id(self.domain_obj), "center_manifold", degree)
        
        def _factory() -> CenterManifold:
            return CenterManifold(self.domain_obj, degree)
        
        return self.get_or_create(cache_key, _factory)

    def hamiltonian(self, max_deg: int, form: str = "center_manifold_real") -> Hamiltonian:
        """
        Return a Hamiltonian object from the associated CenterManifold.

        Parameters
        ----------
        max_deg : int
            The maximum degree of the Hamiltonian expansion.
        form : str
            The Hamiltonian form to get coefficients for. Default is "center_manifold_real".
            Available forms: 'physical', 'real_normal', 'complex_normal', 
            'normalized', 'center_manifold_complex', 'center_manifold_real'.
            
        Returns
        -------
        :class:`~hiten.system.hamiltonian.Hamiltonian`
            The Hamiltonian object with the specified form and degree.
        """
        cache_key = self.make_key(id(self.domain_obj), "hamiltonian", max_deg, form)
        
        def _factory() -> Hamiltonian:
            center_manifold = self.center_manifold(max_deg)
            center_manifold.compute()
            
            # Get Hamiltonian from the pipeline
            try:
                hamiltonian = center_manifold.dynamics.pipeline.get_hamiltonian(form)
                if hamiltonian is None:
                    raise ValueError(f"No Hamiltonian data available for form '{form}' at degree {max_deg}")
                
                # Return the Hamiltonian object directly
                return hamiltonian
            except Exception as e:
                raise ValueError(f"No Hamiltonian data available for form '{form}' at degree {max_deg}: {e}")
        
        return self.get_or_create(cache_key, _factory)

    def hamiltonians(self, max_deg: int) -> dict[str, Hamiltonian]:
        """
        Return all Hamiltonian representations from the associated CenterManifold.

        Parameters
        ----------
        max_deg : int
            The maximum degree of the Hamiltonian expansion.
            
        Returns
        -------
        dict[str, :class:`~hiten.system.hamiltonian.Hamiltonian`]
            Dictionary with keys: 'physical', 'real_normal', 'complex_normal', 
            'normalized', 'center_manifold_complex', 'center_manifold_real'.
            Each value is a Hamiltonian object.
        """
        forms = [
            'physical',
            'real_normal', 
            'complex_normal',
            'normalized',
            'center_manifold_complex',
            'center_manifold_real',
        ]
        
        hamiltonians = {}
        for form in forms:
            try:
                hamiltonians[form] = self.hamiltonian(max_deg, form)
            except ValueError:
                continue
                
        return hamiltonians

    def generating_functions(self, max_deg: int) -> list[LieGeneratingFunction]:
        """
        Return the Lie-series generating functions from CenterManifold.
        
        Parameters
        ----------
        max_deg : int
            The maximum degree of the generating function expansion.
            
        Returns
        -------
        list[:class:`~hiten.system.hamiltonian.LieGeneratingFunction`]
            List of LieGeneratingFunction objects.
        """
        cache_key = self.make_key(id(self.domain_obj), "generating_functions", max_deg)
        
        def _factory() -> list[LieGeneratingFunction]:
            center_manifold = self.center_manifold(max_deg)
            center_manifold.compute()
            
            try:
                gen_funcs = center_manifold.dynamics.pipeline.get_generating_functions("partial")
                if gen_funcs is None:
                    return []
                
                generating_functions = []
                if gen_funcs.poly_G:
                    for i, g_data in enumerate(gen_funcs.poly_G):
                        gf = LieGeneratingFunction(
                            poly_G=[g_data.copy()],
                            poly_elim=[],
                            degree=max_deg,
                            ndof=3,
                            name=f"L{self.domain_obj.idx}_G{i}_{max_deg}"
                        )
                        generating_functions.append(gf)
                
                return generating_functions
            except Exception:
                return []
        
        return self.get_or_create(cache_key, _factory)

    def create_orbit(self, family: str | type["PeriodicOrbit"], /, **kwargs) -> "PeriodicOrbit":
        """
        Create a periodic orbit family anchored at this libration point.

        The helper transparently instantiates the appropriate concrete
        subclass of :class:`~hiten.system.orbits.base.PeriodicOrbit` and
        returns it.  The mapping is based on the family string or directly
        on a subclass type::

            L1 = system.get_libration_point(1)
            orb1 = L1.create_orbit("halo", amplitude_z=0.03, zenith="northern")
            orb2 = L1.create_orbit("lyapunov", amplitude_x=0.05)

        Parameters
        ----------
        family : str or :class:`~hiten.system.orbits.base.PeriodicOrbit` subclass
            Identifier of the orbit family or an explicit subclass type.
            Accepted strings (case-insensitive): "halo", "lyapunov",
            "vertical_lyapunov" and "generic".  If a subclass is
            passed, it is instantiated directly.
        **kwargs
            Forwarded verbatim to the underlying orbit constructor.

        Returns
        -------
        :class:`~hiten.system.orbits.base.PeriodicOrbit` 
        | :class:`~hiten.system.orbits.base.GenericOrbit` 
        | :class:`~hiten.system.orbits.base.HaloOrbit` 
        | :class:`~hiten.system.orbits.base.LyapunovOrbit` 
        | :class:`~hiten.system.orbits.base.VerticalOrbit`
            Newly created orbit instance.
        """
        from hiten.system.orbits.base import GenericOrbit, PeriodicOrbit
        from hiten.system.orbits.halo import HaloOrbit
        from hiten.system.orbits.lyapunov import LyapunovOrbit
        from hiten.system.orbits.vertical import VerticalOrbit
        from hiten.system.orbits.lissajous import LissajousOrbit

        if isinstance(family, type) and issubclass(family, PeriodicOrbit):
            orbit_cls = family
            return orbit_cls(self.domain_obj, **kwargs)

        key = family.lower().strip()
        mapping: dict[str, type[PeriodicOrbit]] = {
            "halo": HaloOrbit,
            "lyapunov": LyapunovOrbit,
            "vertical_lyapunov": VerticalOrbit,
            "vertical": VerticalOrbit,
            "generic": GenericOrbit,
            "lissajous": LissajousOrbit,
        }

        if key not in mapping:
            raise ValueError(
                f"Unknown orbit family '{family}'. Available options: {', '.join(mapping.keys())} "
                "or pass a PeriodicOrbit subclass directly."
            )

        orbit_cls = mapping[key]
        return orbit_cls(self.domain_obj, **kwargs)

    @property
    @abstractmethod
    def sign(self) -> int:
        """
        Sign convention (+-1) used for local <-> synodic transformations.

        Returns
        -------
        int
            The sign convention for coordinate transformations.
        """
        pass

    @property
    @abstractmethod
    def a(self) -> float:
        """
        Offset a used in frame changes for triangular points.

        Returns
        -------
        float
            The offset value a (dimensionless).
        """
        pass

    @property
    def position(self) -> np.ndarray:
        """Calculate the position of the triangular libration point.
        
        Returns
        -------
        numpy.ndarray, shape (3,)
            Position vector [x, y, z] in nondimensional units.
        """
        cache_key = self.make_key(id(self.domain_obj), "position")
        
        def _factory() -> np.ndarray:
            return self._compute_position()
        
        return self.get_or_create(cache_key, _factory)

    @property
    def linear_data(self):
        """Get the linear data for the collinear libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2, None, C, Cinv)
            Object containing the linear data for the libration point.
        """
        cache_key = self.make_key(id(self.domain_obj), "linear_data")
        
        def _factory():
            return self._compute_linear_data()
        
        return self.get_or_create(cache_key, _factory)

    @property
    def linear_modes(self) -> Tuple[float, float, float | None]:
        """
        Compute the linear modes (lambda1, omega1, omega2) for the collinear libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2) values in nondimensional units.
        """
        cache_key = self.make_key(id(self.domain_obj), "linear_modes")
        
        def _factory() -> Tuple[float, float, float | None]:
            return self._compute_linear_modes()
        
        return self.get_or_create(cache_key, _factory)

    @property
    def normal_form_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """The normal form transform for the collinear libration point."""
        cache_key = self.make_key(id(self.domain_obj), "normal_form_transform")
        
        def _factory() -> Tuple[np.ndarray, np.ndarray]:
            return self._build_normal_form()
        
        return self.get_or_create(cache_key, _factory)

    @abstractmethod
    def scale_factor(self, lambda1: float, omega1: float) -> Tuple[float, float]:
        """
        Compute the scale factor for the collinear libration point.
        
        Returns
        -------
        tuple
            (s1, s2) scale factors for the hyperbolic and elliptic components.
        """
        pass
    
    @property
    def eigendecomposition_config(self) -> EigenDecompositionConfig:
        """The eigen decomposition configuration for the collinear libration point.
        
        Returns
        -------
        :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
            The eigendecomposition configuration with reasonable defaults.
        """
        if self._eigendecomposition_config is None:
            self._eigendecomposition_config = EigenDecompositionConfig(
                problem_type=_ProblemType.EIGENVALUE_DECOMPOSITION,
                system_type=_SystemType.CONTINUOUS,
            )
        return self._eigendecomposition_config
    
    @eigendecomposition_config.setter
    def eigendecomposition_config(self, config: EigenDecompositionConfig) -> None:
        """Set the eigen decomposition configuration for the collinear libration point.
        
        Invalidates the generator cache to trigger recreation with the new config.
        
        Parameters
        ----------
        config : :class:`~hiten.algorithms.linalg.config.EigenDecompositionConfig`
            New eigendecomposition configuration.
        """
        self._eigendecomposition_config = config
        self._generator = None  # Invalidate cache to trigger recreation
    
    @property
    def eigendecomposition_options(self) -> "EigenDecompositionOptions":
        """Runtime options for eigenvalue decomposition.
        
        Returns
        -------
        :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`
            The eigendecomposition options with reasonable defaults.
        """
        if self._eigendecomposition_options is None:
            from hiten.algorithms.linalg.options import EigenDecompositionOptions
            self._eigendecomposition_options = EigenDecompositionOptions(
                delta=1e-6,
                tol=1e-6,
            )
        return self._eigendecomposition_options
    
    @eigendecomposition_options.setter
    def eigendecomposition_options(self, value: "EigenDecompositionOptions") -> None:
        """Set runtime options for eigenvalue decomposition.
        
        Parameters
        ----------
        value : :class:`~hiten.algorithms.linalg.options.EigenDecompositionOptions`
            New eigendecomposition options.
        """
        self._eigendecomposition_options = value


class _CollinearDynamicsService(_LibrationDynamicsService):
    """Provide stability analysis and geometry helpers for collinear libration points.
    
    Parameters
    ----------
    point : :class:`~hiten.system.libration.collinear.CollinearPoint`
        The collinear libration point.  
    """

    def __init__(self, point: "CollinearPoint") -> None:
        super().__init__(point)

    @property
    def position(self) -> np.ndarray:
        """Calculate the position of the collinear libration point.
        
        Returns
        -------
        numpy.ndarray, shape (3,)
            Position vector [x, y, z] in nondimensional units.
        """
        cache_key = self.make_key(id(self.domain_obj), "position")
        
        def _factory() -> np.ndarray:
            x = self._compute_position(self._position_search_interval)
            return np.array([x, 0, 0], dtype=np.float64)
        
        return self.get_or_create(cache_key, _factory)

    @property
    @abstractmethod
    def won(self) -> Tuple[int, float]:
        """
        Get the won value for the collinear libration point.
        
        Returns
        -------
        tuple
            (sign, won) where sign is the sign convention and won is the won value.
        """
        pass

    @property
    def gamma(self) -> float:
        """
        The gamma value for the collinear libration point.
        
        Returns
        -------
        float
            The gamma value (dimensionless).
        """
        cache_key = self.make_key(id(self.domain_obj), "gamma")
        
        def _factory() -> float:
            return self._compute_gamma()
        
        return self.get_or_create(cache_key, _factory)

    def scale_factor(self, lambda1: float, omega1: float) -> Tuple[float, float]:
        """
        The scale factor for the collinear libration point.
        
        Returns
        -------
        tuple
            (s1, s2) scale factors for the hyperbolic and elliptic components.
        """
        cache_key = self.make_key(id(self.domain_obj), "scale_factor")
        
        def _factory() -> Tuple[float, float]:
            return self._compute_scale_factor(lambda1, omega1)
        
        return self.get_or_create(cache_key, _factory)

    def cn(self, n: int) -> float:
        """
        The cn coefficient for the collinear libration point.
        
        Parameters
        ----------
        n : int
            The coefficient index (must be non-negative).
            
        Returns
        -------
        float
            The cn coefficient value (dimensionless).
        """
        cache_key = self.make_key(id(self.domain_obj), "cn", n)
        
        def _factory() -> float:
            return self._compute_cn(n)
        
        return self.get_or_create(cache_key, _factory)

    def _compute_gamma(self) -> float:
        """Compute gamma for the collinear libration point by solving the quintic polynomial.
        
        Returns
        -------
        float
            The gamma value (dimensionless).
        """
        coeffs, search_range = self._gamma_poly_def
        return self._solve_gamma_polynomial(coeffs, search_range)

    def _compute_position(self, primary_interval: list) -> float:
        """Find the x-coordinate of a collinear point using retry logic.

        Parameters
        ----------
        primary_interval : list
            Initial interval [a, b] to search for the root in nondimensional units.

        Returns
        -------
        float
            x-coordinate of the libration point in nondimensional units.

        Raises
        ------
        RuntimeError
            If both primary and fallback searches fail.
        """
        func = lambda x_val: self._dOmega_dx(x_val)
        
        # Try primary interval first
        try:
            a, b = primary_interval
            x = solve_bracketed_brent(func, a, b, xtol=1e-12, max_iter=200)

            if x is None:
                raise ValueError("Root not found in primary interval")
            return x
        except Exception as e:
            # Fallback: try a wider interval
            try:
                fallback_a = max(- self.mu + 0.001, a - 0.1)  # Ensure we don't go too close to primary
                fallback_b = min(1 - self.mu - 0.001, b + 0.1)  # Ensure we don't go too close to secondary
                
                # Try with more relaxed tolerance
                x = solve_bracketed_brent(func, fallback_a, fallback_b, xtol=1e-10, max_iter=500)
                
                if x is None:
                    raise ValueError("Root not found in fallback interval")
                return x

            except Exception as fallback_e:
                raise RuntimeError(f"{self.domain_obj.__class__.__name__}: Both primary interval {primary_interval} and fallback failed. Primary error: {e}, Fallback error: {fallback_e}") from fallback_e

    def _solve_gamma_polynomial(self, coeffs: list, gamma_range: tuple) -> float:
        """Solve the quintic polynomial for gamma with validation and fallback.
        
        Parameters
        ----------
        coeffs : list
            Polynomial coefficients from highest to lowest degree.
        gamma_range : tuple
            (min_gamma, max_gamma) valid range for this point type.
            
        Returns
        -------
        float
            The gamma value for this libration point (dimensionless).
            
        Raises
        ------
        RuntimeError
            If polynomial root finding fails or no valid root is found.
        """
        
        min_gamma, max_gamma = gamma_range
        
        def poly_func(x):
            result = 0.0
            for i, coeff in enumerate(coeffs):
                result += coeff * (x ** (len(coeffs) - 1 - i))
            return result
        
        search_points = np.linspace(min_gamma + 1e-6, max_gamma - 1e-6, 50)
        
        for x0 in search_points:
            try:
                bracket = expand_bracket(
                    poly_func, x0,
                    dx0=0.01,
                    grow=1.5,
                    max_expand=10,
                    crossing_test=lambda f_prev, f_curr: f_prev * f_curr < 0,
                    symmetric=True
                )
                
                if bracket[0] != bracket[1]:
                    root = solve_bracketed_brent(poly_func, bracket[0], bracket[1])
                    if root is not None and min_gamma < root < max_gamma:
                        return root
                        
            except Exception:
                continue
        
        n_intervals = 20
        interval_size = (max_gamma - min_gamma) / n_intervals
        
        for i in range(n_intervals):
            a = min_gamma + i * interval_size
            b = min_gamma + (i + 1) * interval_size
            
            fa = poly_func(a)
            fb = poly_func(b)
            
            if fa * fb < 0:
                try:
                    root = solve_bracketed_brent(poly_func, a, b)
                    if root is not None:
                        return root
                except Exception:
                    continue
        
        raise RuntimeError(f"No valid polynomial root found for {self.domain_obj.__class__.__name__} in range {gamma_range}")

    def _dOmega_dx(self, x: float) -> float:
        """Compute the derivative of the effective potential with respect to x.
        
        Parameters
        ----------
        x : float
            x-coordinate in the rotating frame (nondimensional units).
        
        Returns
        -------
        float
            Value of dOmega/dx at the given x-coordinate (nondimensional units).
            
        Raises
        ------
        ValueError
            If x-coordinate is too close to primary masses.
        """
        mu = self.mu

        r1_sq = (x + mu)**2
        r2_sq = (x - (1 - mu))**2

        if r1_sq < 1e-16 or r2_sq < 1e-16:
            raise ValueError(f"x-coordinate too close to primary masses: x={x}")

        r1_3 = r1_sq**1.5
        r2_3 = r2_sq**1.5

        term1 = x
        term2 = -(1 - mu) * (x + mu) / r1_3
        term3 = -mu * (x - (1 - mu)) / r2_3
        
        return term1 + term2 + term3

    def _J_hess_H2(self) -> np.ndarray:
        """Compute the 6x6 symplectic matrix for the quadratic Hamiltonian H2.
        
        Returns
        -------
        numpy.ndarray, shape (6, 6)
            The symplectic matrix J for the quadratic Hamiltonian.
        """
        c2 = self.cn(2)
        omega2 = np.sqrt(c2)

        J_planar = np.array([
            [0.0, 1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
            [2.0 * c2, 0.0, 0.0, 1.0],
            [0.0, -c2, -1.0, 0.0],
        ], dtype=np.float64)

        J_vert = np.array([[0.0, omega2], [-omega2, 0.0]], dtype=np.float64)

        J_full = np.zeros((6, 6), dtype=np.float64)
        J_full[:4, :4] = J_planar
        J_full[4:, 4:] = J_vert

        return J_full

    def _compute_linear_modes(self):
        """Compute the linear modes (lambda1, omega1, omega2) for the libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2) in nondimensional units.
            
        Raises
        ------
        RuntimeError
            If no real eigenvalues are found or expected frequencies are missing.
        """

        J_full = self._J_hess_H2()
        c2 = self.cn(2)
        omega2_expected = np.sqrt(c2)

        eigvals, _ = np.linalg.eig(J_full)

        real_mask = np.abs(eigvals.imag) < 1e-12
        imag_mask = ~real_mask

        real_eigs = eigvals[real_mask].real

        if real_eigs.size == 0:
            raise RuntimeError("No real eigen-values found while calculating linear modes.")

        lambda1 = float(np.max(np.abs(real_eigs)))

        imag_eigs = eigvals[imag_mask]
        omegas = np.unique(np.round(np.abs(imag_eigs.imag), decimals=12))

        if omegas.size < 2:
            raise RuntimeError(f"Expected two distinct imaginary frequencies, got {omegas.size}.")

        idx_vert = int(np.argmin(np.abs(omegas - omega2_expected)))
        omega2_val = float(omegas[idx_vert])
        omega1_val = float(omegas[1 - idx_vert])  # The other one

        if omega1_val < omega2_val:
            omega1_val, omega2_val = omega2_val, omega1_val

        return (float(lambda1), float(omega1_val), float(omega2_val))
    
    def _compute_linear_data(self):
        """Compute the linear data for the collinear libration point."""
        lambda1, omega1, omega2 = self.linear_modes
        C, Cinv = self.normal_form_transform
        
        return lambda1, omega1, omega2, None, C, Cinv

    def _compute_scale_factor(self, lambda1, omega1):
        """The normalization factors s1 and s2 used in the normal form transformation.
        
        Parameters
        ----------
        lambda1 : float
            The hyperbolic mode value (nondimensional units).
        omega1 : float
            The elliptic mode value (nondimensional units).
            
        Returns
        -------
        tuple
            (s1, s2) normalization factors for the hyperbolic and elliptic components.
            
        Raises
        ------
        RuntimeError
            If the expressions for s1 or s2 are negative.
        """
        c2 = self.cn(2)

        # Common terms
        term_lambda = (4.0 + 3.0 * c2) * (lambda1 ** 2.0)
        term_omega = (4.0 + 3.0 * c2) * (omega1 ** 2.0)
        base_term = 4.0 + 5.0 * c2 - 6.0 * (c2 ** 2.0)

        # Calculate expressions under square root
        expr1 = 2.0 * lambda1 * (term_lambda + base_term)
        expr2 = omega1 * (term_omega - base_term)
        
        # Validate expressions are positive
        if expr1 < 0:
            raise RuntimeError(f"Expression for s1 is negative: {expr1}.")
            
        if expr2 < 0:
            raise RuntimeError(f"Expression for s2 is negative: {expr2}.")
        
        return np.sqrt(expr1), np.sqrt(expr2)

    def _build_normal_form(self) -> Tuple[np.ndarray, np.ndarray]:
        """The normal form transformation matrices.
        
        Returns
        -------
        tuple
            (C, Cinv) where C is the symplectic transformation matrix
            and Cinv is its inverse.
        """
        
        lambda1, omega1, omega2 = self.linear_modes
        c2 = self.cn(2)
        s1, s2 = self.scale_factor(lambda1, omega1)

        if abs(omega2) < 1e-12:
            logger.warning(
                "Vertical frequency omega2 is very small (%.2e). Transformation matrix may be ill-conditioned.",
                omega2,
            )
            sqrt_omega2 = 1e-6
        else:
            sqrt_omega2 = np.sqrt(omega2)

        C = np.zeros((6, 6))

        C[0, 0] = 2 * lambda1 / s1
        C[0, 3] = -2 * lambda1 / s1
        C[0, 4] = 2 * omega1 / s2

        C[1, 0] = (lambda1**2 - 2 * c2 - 1) / s1
        C[1, 1] = (-omega1**2 - 2 * c2 - 1) / s2
        C[1, 3] = (lambda1**2 - 2 * c2 - 1) / s1

        C[2, 2] = 1 / sqrt_omega2

        C[3, 0] = (lambda1**2 + 2 * c2 + 1) / s1
        C[3, 1] = (-omega1**2 + 2 * c2 + 1) / s2
        C[3, 3] = (lambda1**2 + 2 * c2 + 1) / s1

        C[4, 0] = (lambda1**3 + (1 - 2 * c2) * lambda1) / s1
        C[4, 3] = (-lambda1**3 - (1 - 2 * c2) * lambda1) / s1
        C[4, 4] = (-omega1**3 + (1 - 2 * c2) * omega1) / s2

        C[5, 5] = sqrt_omega2

        Cinv = np.linalg.inv(C)
        return C, Cinv

    @property
    @abstractmethod
    def _position_search_interval(self) -> list:
        """
        The search interval for finding the x-position.
        
        Returns
        -------
        list
            [min_x, max_x] interval for root finding in nondimensional units.
        """
        pass

    @property
    @abstractmethod
    def _gamma_poly_def(self) -> Tuple[list, tuple]:
        """
        The quintic polynomial for gamma calculation.
        
        Returns
        -------
        tuple
            (coefficients, search_range) where coefficients is a list of
            polynomial coefficients and search_range is (min_gamma, max_gamma).
        """
        pass


class _L1DynamicsService(_CollinearDynamicsService):
    """Provide stability analysis and geometry helpers for L1 libration points."""

    def __init__(self, point: "L1Point") -> None:
        super().__init__(point)

    @property
    def sign(self) -> int:
        """The sign convention for L1 libration points."""
        return -1
    
    @property
    def a(self) -> float:
        """The offset along the x-axis used in frame changes for L1 libration points."""
        return -1 + self.gamma

    @property
    def won(self) -> Tuple[int, float]:
        """The won value for L1 libration points."""
        return (+1, 1 - self.mu)

    @property
    def _position_search_interval(self) -> list:
        """
        The search interval for L1's x-position.
        
        Returns
        -------
        list
            [min_x, max_x] interval for root finding in nondimensional units.
            L1 is between the primaries: -mu < x < 1-mu.
        """
        # L1 is between the primaries: -mu < x < 1-mu
        return [-self.mu + 0.01, 1 - self.mu - 0.01]

    @property
    def _gamma_poly_def(self) -> Tuple[list, tuple]:
        """
        The quintic polynomial definition for L1's gamma value.
        
        Returns
        -------
        tuple
            (coefficients, search_range) for the L1 quintic polynomial.
            The polynomial is: x^5 - (3-mu)x^4 + (3-2mu)x^3 - mux^2 + 2mux - mu = 0.
        """
        mu = self.mu
        # Coefficients for L1 quintic: x^5 - (3-mu)x^4 + (3-2mu)x^3 - mux^2 + 2mux - mu = 0
        coeffs = [1, -(3 - mu), (3 - 2 * mu), -mu, 2 * mu, -mu]
        return coeffs, (0, 1)

    def _compute_cn(self, n: int) -> float:
        """The cn coefficient for L1 using Jorba & Masdemont (1999), eq. (3).
        
        Parameters
        ----------
        n : int
            The coefficient index (must be non-negative).
            
        Returns
        -------
        float
            The cn coefficient value (dimensionless).
        """
        gamma = self.gamma
        mu = self.mu
        
        term1 = 1 / (gamma**3)
        term2 = mu
        term3 = ((-1)**n) * (1 - mu) * (gamma**(n+1)) / ((1 - gamma)**(n+1))
        
        return term1 * (term2 + term3)


class _L2DynamicsService(_CollinearDynamicsService):
    """Provide stability analysis and geometry helpers for L2 libration points."""

    def __init__(self, point: "L2Point") -> None:
        super().__init__(point)

    @property
    def sign(self) -> int:
        """The sign convention for L2 libration points."""
        return -1
    
    @property
    def a(self) -> float:
        """The offset along the x-axis used in frame changes for L2 libration points."""
        return -1 - self.gamma

    @property
    def won(self) -> Tuple[int, float]:
        """The won value for L2 libration points."""
        return (-1, 1 - self.mu)

    @property
    def _position_search_interval(self) -> list:
        """
        The search interval for L2's x-position.
        
        Returns
        -------
        list
            [min_x, max_x] interval for root finding in nondimensional units.
            L2 is beyond the smaller primary: x > 1-mu.
        """
        # L2 is beyond the smaller primary: x > 1-mu
        return [1 - self.mu + 0.001, 2.0]

    @property
    def _gamma_poly_def(self) -> Tuple[list, tuple]:
        """
        The quintic polynomial definition for L2's gamma value.
        
        Returns
        -------
        tuple
            (coefficients, search_range) for the L2 quintic polynomial.
            The polynomial is: x^5 + (3-mu)x^4 + (3-2mu)x^3 - mux^2 - 2mux - mu = 0.
        """
        mu = self.mu
        # Coefficients for L2 quintic: x^5 + (3-mu)x^4 + (3-2mu)x^3 - mux^2 - 2mux - mu = 0
        coeffs = [1, (3 - mu), (3 - 2 * mu), -mu, -2 * mu, -mu]
        return coeffs, (0, 1)

    def _compute_cn(self, n: int) -> float:
        """
        The cn coefficient for L2 using Jorba & Masdemont (1999), eq. (3).
        
        Parameters
        ----------
        n : int
            The coefficient index (must be non-negative).
            
        Returns
        -------
        float
            The cn coefficient value (dimensionless).
        """
        gamma = self.gamma
        mu = self.mu
        
        term1 = 1 / (gamma**3)
        term2 = ((-1)**n) * mu
        term3 = ((-1)**n) * (1 - mu) * (gamma**(n+1)) / ((1 + gamma)**(n+1))
        
        return term1 * (term2 + term3)

class _L3DynamicsService(_CollinearDynamicsService):
    """Provide stability analysis and geometry helpers for L3 libration points."""

    def __init__(self, point: "L3Point") -> None:
        super().__init__(point)

    @property
    def sign(self) -> int:
        """The sign convention for L3 libration points."""
        return 1
    
    @property
    def a(self) -> float:
        """The offset along the x-axis used in frame changes for L3 libration points."""
        return self.gamma

    @property
    def won(self) -> Tuple[int, float]:
        """The won value for L3 libration points."""
        return (+1, -self.mu)

    @property
    def _position_search_interval(self) -> list:
        """
        The search interval for L3's x-position.
        
        Returns
        -------
        list
            [min_x, max_x] interval for root finding in nondimensional units.
            L3 is beyond the larger primary: x < -mu.
        """
        return [-1.5, -self.mu - 0.001]

    @property
    def _gamma_poly_def(self) -> Tuple[list, tuple]:
        """
        The quintic polynomial definition for L3's gamma value.
        
        Returns
        -------
        tuple
            (coefficients, search_range) for the L3 quintic polynomial.
            The polynomial is: x^5 + (2+mu)x^4 + (1+2mu)x^3 - mu_1x^2 - 2mu_1x - mu_1 = 0.
        """
        mu = self.mu
        mu1 = 1 - mu
        coeffs = [1, (2 + mu), (1 + 2 * mu), -mu1, -2 * mu1, -mu1]
        return coeffs, (0.5, 1.5)

    def _compute_cn(self, n: int) -> float:
        """
        The cn coefficient for L3 using Jorba & Masdemont (1999), eq. (3).
        
        Parameters
        ----------
        n : int
            The coefficient index (must be non-negative).
            
        Returns
        -------
        float
            The cn coefficient value (dimensionless).
        """
        gamma = self.gamma
        mu = self.mu
        
        term1 = ((-1)**n) / (gamma**3)
        term2 = (1 - mu)
        term3 = mu * (gamma**(n+1)) / ((1 + gamma)**(n+1))
        
        return term1 * (term2 + term3)


class _TriangularDynamicsService(_LibrationDynamicsService):
    """Provide stability analysis and geometry helpers for triangular libration points."""

    def __init__(self, point: "TriangularPoint") -> None:
        super().__init__(point)

    def scale_factor(self, idx: int) -> Tuple[float, float]:
        """
        Compute the scale factor for the collinear libration point.
        
        Returns
        -------
        tuple
            (s1, s2) scale factors for the hyperbolic and elliptic components.
        """
        cache_key = self.make_key(id(self.domain_obj), "scale_factor", idx)
        
        def _factory() -> Tuple[float, float]:
            return self._compute_scale_factor(idx)
        
        return self.get_or_create(cache_key, _factory)

    def _compute_position(self):
        x = 0.5 - self.mu
        y = self.sign * np.sqrt(3) / 2.0
        return np.array([x, y, 0], dtype=np.float64)

    def _compute_linear_modes(self):
        """Compute the three frequencies (omega_1, omega_2, omega_z) following the convention:
        omega_1 > 0 with omega_1^2 < 1/2, omega_2 < 0, omega_z is vertical frequency = 1.
        
        Returns
        -------
        tuple
            (omega_1, omega_2, omega_z) in nondimensional units.
            
        Raises
        ------
        RuntimeError
            If the expected number of eigenvalues or frequency groups are not found.
        """
        J_full = self._J_hess_H2()
        eigvals = np.linalg.eigvals(J_full)

        imag_eigs = eigvals[np.abs(eigvals.real) < 1e-12]
        omegas_with_sign = imag_eigs.imag  # Keep the signs

        omegas_unique = []
        for omega in omegas_with_sign:
            if not any(np.isclose(omega, existing, atol=1e-12) for existing in omegas_unique):
                omegas_unique.append(omega)

        if len(omegas_unique) != 6:
            raise RuntimeError(f"Expected 6 eigenvalues (+-3 frequencies), got {len(omegas_unique)}.")

        freq_groups = {}
        for omega in omegas_unique:
            abs_omega = abs(omega)
            found_group = False
            for key in freq_groups:
                if np.isclose(abs_omega, key, rtol=1e-10):
                    freq_groups[key].append(omega)
                    found_group = True
                    break
            if not found_group:
                freq_groups[abs_omega] = [omega]
        
        vertical_group_key = min(freq_groups.keys(), key=lambda x: abs(x - 1.0))
        if not np.isclose(vertical_group_key, 1.0, rtol=1e-2):
            raise RuntimeError(f"No frequency group found near 1.0, closest is {vertical_group_key}")
        
        omega_z = vertical_group_key

        planar_omegas = []
        for key, omegas_list in freq_groups.items():
            if not np.isclose(key, vertical_group_key, rtol=1e-10):
                planar_omegas.extend(omegas_list)
        
        planar_freq_groups = {}
        for omega in planar_omegas:
            abs_omega = abs(omega)
            found_group = False
            for key in planar_freq_groups:
                if np.isclose(abs_omega, key, rtol=1e-10):
                    planar_freq_groups[key].append(omega)
                    found_group = True
                    break
            if not found_group:
                planar_freq_groups[abs_omega] = [omega]
        
        if len(planar_freq_groups) != 2:
            raise RuntimeError(f"Expected 2 distinct planar frequency groups, got {len(planar_freq_groups)} groups with magnitudes {list(planar_freq_groups.keys())}")

        planar_mags = sorted(planar_freq_groups.keys())
        smaller_mag, larger_mag = planar_mags
        

        omega1 = larger_mag           # positive, expected > sqrt(1/2)
        omega2 = -smaller_mag         # negative, expected < -sqrt(1/2)

        if not (omega1**2 > 0.5 and omega2**2 < 0.5):
            raise RuntimeError(f"Computed planar frequencies do not strictly satisfy the requested ordering: omega_1={omega1:.4f}, omega_2={omega2:.4f}.")

        return (float(omega1), float(omega2), float(omega_z))

    def _compute_linear_data(self):
        """Compute the linear data for the triangular libration point.
        
        Returns
        -------
        tuple
            (lambda1, omega1, omega2, omega_z, C, Cinv)
            Object containing the linear data for the libration point.
        """
        omega1, omega2, omega_z = self.linear_modes
        C, Cinv = self.normal_form_transform
        
        return None, omega1, omega2, omega_z, C, Cinv

    def _J_hess_H2(self) -> np.ndarray:
        """The 6x6 symplectic matrix for the quadratic Hamiltonian H2.
        
        Returns
        -------
        numpy.ndarray, shape (6, 6)
            The symplectic matrix J for the quadratic Hamiltonian.
        """
        a = self.a
        
        J_planar = np.array([
            [0.0, 1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
            [-0.25, a, 0.0, 1.0],
            [a, 1.25, -1.0, 0.0],
        ], dtype=np.float64)

        J_vert = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)

        J_full = np.zeros((6, 6), dtype=np.float64)
        J_full[:4, :4] = J_planar
        J_full[4:, 4:] = J_vert
        return J_full

    def _compute_scale_factor(self, idx: int) -> float:
        """The scaling factor for the given mode index.
        
        Parameters
        ----------
        idx : int
            The mode index (0, 1 for planar modes, 2 for vertical).
            
        Returns
        -------
        float
            The scaling factor (dimensionless).
        """
        if idx == 2:
            return 1.0
        return np.sqrt(self._d_omega(idx))

    def _build_normal_form(self) -> Tuple[np.ndarray, np.ndarray]:
        """The normal form transformation matrices.
        
        Returns
        -------
        tuple
            (C, Cinv) where C is the symplectic transformation matrix
            and Cinv is its inverse.
        """
        eigvs = self._get_eigvs().T
        s = np.array([
            self.scale_factor(0),
            self.scale_factor(1),
            self.scale_factor(2),
        ])
        S = np.diag(np.concatenate([s, s]))
        C = eigvs @ np.linalg.inv(S)
        Cinv = np.linalg.inv(C)
        return C, Cinv

    def _get_eigvs(self):
        """The six real eigenvectors (u_1, u_2, u_3, v_1, v_2, v_3)
        providing a canonical basis of the centre sub-space.

        For triangular points the flow decomposes into a planar 2-DOF part
        and a vertical 1-DOF harmonic oscillator uncoupled from the plane.
        The planar eigenvectors are constructed analytically following the
        classical derivation (see Gomez et al., 1993, par3.2).  They live in
        the first four coordinates (x, y, p_x, p_y).  We simply append two
        zeros to embed them in the full 6-D phase-space.  The vertical pair
        is trivial thanks to the decoupling: (z, p_z) already form a
        canonical coordinate pair.
        
        Returns
        -------
        numpy.ndarray, shape (6, 6)
            Matrix of eigenvectors as rows.
        """
        a = self.a
        omega1, omega2, omega_z = self.linear_modes  # omega_z == 1

        # The vectors are written in the (x, y, z, p_x, p_y, p_z) ordering used by
        # _J_hess_H2.  They are then embedded into 6-D by appending zeros
        # for the vertical coordinates (z, p_z). 
        u1_planar = np.array([a, -omega1**2 - 0.75, -omega1**2 + 0.75, a])
        u2_planar = np.array([a, -omega2**2 - 0.75, -omega2**2 + 0.75, a])
        v1_planar = np.array([2 * omega1, 0.0, a * omega1, -omega1**3 + 1.25 * omega1])
        v2_planar = np.array([2 * omega2, 0.0, a * omega2, -omega2**3 + 1.25 * omega2])

        u1 = np.zeros(6)
        u2 = np.zeros(6)
        v1 = np.zeros(6)
        v2 = np.zeros(6)

        # Assign planar components.
        u1[[0, 1, 3, 4]] = u1_planar
        u2[[0, 1, 3, 4]] = u2_planar
        v1[[0, 1, 3, 4]] = v1_planar
        v2[[0, 1, 3, 4]] = v2_planar

        sqrt_omega_z = np.sqrt(abs(omega_z))  # positive by construction
        u3 = np.zeros(6)
        v3 = np.zeros(6)
        u3[2] = 1.0 / sqrt_omega_z  # z coordinate
        v3[5] = sqrt_omega_z        # p_z coordinate

        # Stack as rows.
        eigv_matrix = np.vstack([u1, u2, u3, v1, v2, v3])
        return eigv_matrix

    def _d_omega(self, idx: int) -> float:
        """The derivative term for the given mode index.
        
        Parameters
        ----------
        idx : int
            The mode index.
            
        Returns
        -------
        float
            The derivative term (dimensionless).
        """
        omegas = self.linear_modes
        omega = omegas[idx]

        return omega * (2*omega**4+0.5*omega**2-0.75)


class _L4DynamicsService(_TriangularDynamicsService):
    """Provide stability analysis and geometry helpers for L4 libration points."""

    def __init__(self, point: "L4Point") -> None:
        super().__init__(point)

    @property
    def sign(self) -> int:
        """The sign convention for L4 libration points."""
        return 1

    @property
    def a(self) -> float:
        """The offset along the x-axis used in frame changes for L4 libration points."""
        return self.sign * 3 * np.sqrt(3) / 4 * (1 - 2 * self.mu)


class _L5DynamicsService(_TriangularDynamicsService):
    """Provide stability analysis and geometry helpers for L5 libration points."""

    def __init__(self, point: "L5Point") -> None:
        super().__init__(point)

    @property
    def sign(self) -> int:
        """The sign convention for L5 libration points."""
        return -1

    @property
    def a(self) -> float:
        """The offset along the x-axis used in frame changes for L5 libration points."""
        return self.sign * 3 * np.sqrt(3) / 4 * (1 - 2 * self.mu)


class _LibrationServices(_ServiceBundleBase):
    """Provide stability analysis and geometry helpers for libration points
    
    Parameters
    ----------
    domain_obj : :class:`~hiten.system.libration.base.LibrationPoint`
        The libration point.
    persistence : :class:`~hiten.algorithms.types.services.libration._LibrationPersistenceService`
        The persistence service.
    dynamics : :class:`~hiten.algorithms.types.services.libration._LibrationDynamicsService`
        The dynamics service.
    """

    def __init__(self, domain_obj: "LibrationPoint", persistence: _LibrationPersistenceService, dynamics: _LibrationDynamicsService) -> None:
        super().__init__(domain_obj)
        self.persistence = persistence
        self.dynamics = dynamics

    @classmethod
    def default(cls, domain_obj: "LibrationPoint") -> "_LibrationServices":
        """Create a default service bundle for a libration point.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.libration.base.LibrationPoint`
            The libration point.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.libration._LibrationServices`
            The service bundle.
        """
        dynamics = _LibrationServices._check_point_type(domain_obj)
        return cls(
            domain_obj=domain_obj,
            persistence=_LibrationPersistenceService(),
            dynamics=dynamics
        )

    @classmethod
    def with_shared_dynamics(cls, dynamics: _LibrationDynamicsService) -> "_LibrationServices":
        """Create a service bundle with a shared dynamics service.
        
        Parameters
        ----------
        dynamics : :class:`~hiten.algorithms.types.services.libration._LibrationDynamicsService`
            The dynamics service.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.libration._LibrationServices`
            The service bundle.
        """
        return cls(
            domain_obj=dynamics.domain_obj,
            persistence=_LibrationPersistenceService(),
            dynamics=dynamics
        )

    @staticmethod
    def _check_point_type(domain_obj: "LibrationPoint") -> _LibrationDynamicsService:
        """Check the type of the libration point and return the corresponding dynamics service.
        
        Parameters
        ----------
        domain_obj : :class:`~hiten.system.libration.base.LibrationPoint`
            The libration point.

        Returns
        -------
        :class:`~hiten.algorithms.types.services.libration._LibrationDynamicsService`
            The dynamics service.   
        """
        from hiten.system.libration.collinear import L1Point, L2Point, L3Point
        from hiten.system.libration.triangular import L4Point, L5Point

        mapping = {
            L1Point: _L1DynamicsService,
            L2Point: _L2DynamicsService,
            L3Point: _L3DynamicsService,
            L4Point: _L4DynamicsService,
            L5Point: _L5DynamicsService,
        }
        
        return mapping[type(domain_obj)](domain_obj)