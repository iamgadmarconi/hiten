"""Hamiltonian transformation pipeline for the CR3BP.

This module provides the HamiltonianPipeline class that manages the transformation
pipeline for different Hamiltonian representations in the circular restricted
three-body problem. It handles caching, conversion between representations,
and computation of Lie generating functions.

The pipeline uses Lie generating functions G(q, p) to perform canonical
transformations that preserve the Hamiltonian structure while simplifying
the dynamics.


Notes
-----
The pipeline caches all computed Hamiltonians to avoid redundant computation.
Conversion between representations uses a registry-based system with
automatic path finding for multi-step transformations.
"""

from collections import deque
from typing import TYPE_CHECKING, Dict, Optional

from hiten.algorithms.hamiltonian.center._lie import _lie_expansion
from hiten.algorithms.hamiltonian.center._lie import \
    _lie_transform as _lie_transform_partial
from hiten.algorithms.hamiltonian.hamiltonian import (
    _build_physical_hamiltonian_collinear,
    _build_physical_hamiltonian_triangular)
from hiten.algorithms.hamiltonian.normal._lie import \
    _lie_transform as _lie_transform_full
from hiten.algorithms.types.serialization import _SerializeBase
from hiten.algorithms.types.services.hamiltonian import _PipelineService
from hiten.algorithms.types.services import get_hamiltonian_services
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.hamiltonian import Hamiltonian, LieGeneratingFunction
    from hiten.system.libration.base import LibrationPoint


class HamiltonianPipeline(_SerializeBase):
    """
    Manages the transformation pipeline for Hamiltonian representations.

    This class provides a comprehensive pipeline for computing and converting
    between different Hamiltonian representations in the circular restricted
    three-body problem. It handles caching, automatic path finding for
    multi-step conversions, and computation of Lie generating functions.

    Parameters
    ----------
    point : :class:`~hiten.system.libration.base.LibrationPoint`
        The libration point about which the normal form is computed
    degree : int
        Maximum total degree of the polynomial truncation

    Attributes
    ----------
    point : :class:`~hiten.system.libration.base.LibrationPoint`
        The libration point about which the normal form is computed
    degree : int
        The maximum total degree of the polynomial truncation
    _hamiltonian_cache : dict
        Cache of computed Hamiltonian objects keyed by form name
    _generating_function_cache : dict
        Cache of computed Lie generating functions

    Notes
    -----
    All heavy computations are cached and subsequent calls with the same
    parameters are inexpensive. The pipeline automatically finds the shortest
    conversion path between any two Hamiltonian representations.
    """

    def __init__(
        self,
        point: "LibrationPoint",
        degree: int):
        if not isinstance(degree, int) or degree <= 0:
            raise ValueError("degree must be a positive integer")

        self._point = point
        self._max_degree = degree
        self._hamiltonian_cache: Dict[str, "Hamiltonian"] = {}
        self._generating_function_cache: Dict[str, "LieGeneratingFunction"] = {}
        self._registry = get_hamiltonian_services()

        from hiten.system.libration.collinear import CollinearPoint
        from hiten.system.libration.triangular import TriangularPoint

        if isinstance(self._point, CollinearPoint):
            self._build_hamiltonian = _build_physical_hamiltonian_collinear
            self._mix_pairs = (1, 2)
        elif isinstance(self._point, TriangularPoint):
            self._build_hamiltonian = _build_physical_hamiltonian_triangular
            self._mix_pairs = (0, 1, 2)
        else:
            raise ValueError(f"Unsupported libration point type: {type(self._point)}")

    @property
    def registry(self) -> _PipelineService:
        return self._registry

    @property
    def point(self) -> "LibrationPoint":
        """The libration point about which the normal form is computed."""
        return self._point

    @property
    def degree(self) -> int:
        """The maximum total degree of the polynomial truncation."""
        return self._max_degree

    def __str__(self) -> str:
        return f"HamiltonianPipeline(point={self._point}, degree={self._max_degree})"

    def __repr__(self) -> str:
        return f"HamiltonianPipeline(point={self._point!r}, degree={self._max_degree})"

    def get_hamiltonian(self, form: str) -> "Hamiltonian":
        """
        Get a specific Hamiltonian representation.

        This method retrieves a Hamiltonian in the requested representation,
        computing it if necessary and caching the result for future use.

        Parameters
        ----------
        form : str
            The desired Hamiltonian form. Supported forms include:
            - "physical": Physical coordinates in the rotating frame
            - "real_modal": Real modal coordinates (diagonalized linear system)
            - "complex_modal": Complex modal coordinates
            - "complex_partial_normal": Complex partial normal form
            - "real_partial_normal": Real partial normal form
            - "center_manifold_complex": Complex center manifold
            - "center_manifold_real": Real center manifold
            - "complex_full_normal": Complex full normal form
            - "real_full_normal": Real full normal form

        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            The requested Hamiltonian representation

        Raises
        ------
        ValueError
            If the requested form is not supported
        NotImplementedError
            If no conversion path exists to the requested form

        Notes
        -----
        The method automatically finds the shortest conversion path from
        any available form to the requested form. All computations are
        cached to avoid redundant work.
        """
        if form not in self._hamiltonian_cache:
            self._hamiltonian_cache[form] = self._compute_hamiltonian(form)

        return self._hamiltonian_cache[form]

    def _store_generating_functions(self, form_name: str, generating_functions: "LieGeneratingFunction"):
        """
        Store generating functions in the cache.
        
        Parameters
        ----------
        form_name : str
            The Hamiltonian form name (e.g., "complex_partial_normal")
        generating_functions : :class:`~hiten.system.hamiltonians.base.LieGeneratingFunction`
            The LieGeneratingFunction object to cache
            
        Notes
        -----
        This method is called internally by the conversion system when
        generating functions are computed as part of a transformation.
        Only partial and full normal form generating functions are cached.
        """
        # Map form names to cache keys
        if form_name == "complex_partial_normal":
            cache_key = "generating_functions_partial"
        elif form_name == "complex_full_normal":
            cache_key = "generating_functions_full"
        else:
            return  # Don't cache for other forms
        
        self._generating_function_cache[cache_key] = generating_functions
        logger.debug(f"Stored generating functions for {form_name} in cache")

    def _compute_hamiltonian(self, form: str) -> "Hamiltonian":
        """
        Compute a Hamiltonian representation, using conversion if needed.

        This method handles the computation of Hamiltonian representations,
        either by building them directly or by converting from existing forms.

        Parameters
        ----------
        form : str
            The desired Hamiltonian form

        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            The computed Hamiltonian representation

        Raises
        ------
        NotImplementedError
            If no conversion path exists to the requested form

        Notes
        -----
        The method first checks if the form can be built directly (e.g., "physical").
        If not, it finds a suitable source form and performs the conversion,
        potentially through multiple intermediate steps.
        """
        # Check if we can build this form directly
        if form == "physical":
            return self._build_physical_hamiltonian()

        # Try to find a source form in cache or registry
        source_form = self._find_conversion_source(form)
        if source_form is None:
            raise NotImplementedError(
                f"No conversion path found to form '{form}'. "
                f"Available forms: {list(self._get_available_forms())}"
            )

        # Get the source Hamiltonian and convert
        source_ham = self.get_hamiltonian(source_form)

        try:
            result = source_ham.to_state(form, point=self._point, _pipeline=self)
            # Handle tuple returns (Hamiltonian, LieGeneratingFunction)
            if isinstance(result, tuple):
                new_ham, generating_functions = result
                # Store generating functions if we have a pipeline reference
                self._store_generating_functions(form, generating_functions)
                return new_ham
            else:
                return result
        except NotImplementedError:
            pass

        # Otherwise, follow the multi-step conversion path
        return self._follow_conversion_path(source_form, form)

    def _build_physical_hamiltonian(self) -> "Hamiltonian":
        """
        Build the physical Hamiltonian from scratch.

        This method constructs the physical Hamiltonian in the rotating
        synodic frame using the appropriate builder function for the
        libration point type.

        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            The physical Hamiltonian representation

        Notes
        -----
        The physical Hamiltonian is built using the point-specific builder
        function determined during initialization. It represents the full
        6D system in the rotating synodic frame.
        """
        from hiten.system.hamiltonian import Hamiltonian
        logger.debug(f"Building physical Hamiltonian for {self._point}")
        poly_H = self._build_hamiltonian(self._point, self._max_degree)
        return Hamiltonian(poly_H, self._max_degree, ndof=3, name="physical")

    def _find_conversion_source(self, target_form: str) -> Optional[str]:
        """
        Find a source form that can be converted to the target form.

        This method uses breadth-first search to find the shortest conversion
        path from any available form to the target form.

        Parameters
        ----------
        target_form : str
            The target form to find a conversion source for

        Returns
        -------
        str or None
            The source form that can be converted to target_form, or None if
            no conversion path exists

        Notes
        -----
        The method first checks for direct conversions from cached forms,
        then explores the conversion graph using BFS to find the shortest path.
        """
        # Check if we have a direct conversion from any existing form
        for source_form in self._hamiltonian_cache:
            if self._can_convert(source_form, target_form):
                return source_form

        # If no direct conversion, try building from physical
        if self._can_convert("physical", target_form):
            return "physical"

        # debug: Print available conversions
        logger.debug(f"Looking for conversion path to '{target_form}'")
        queue = deque([("physical", ["physical"])])
        visited = {"physical"}
        
        while queue:
            current_form, path = queue.popleft()
            logger.debug(f"Exploring from '{current_form}' with path {path}")
            
            # Check all possible conversions from current_form using registry
            for (src, dst), (_, required_context, _) in self.registry._CONVERSION_REGISTRY.items():
                if src == current_form and dst not in visited:
                    # Check if we have the required context (point is always available)
                    if not required_context or "point" in required_context:
                        visited.add(dst)
                        new_path = path + [dst]
                        logger.debug(f"  Found conversion: {src} -> {dst} (context: {required_context})")
                        
                        # If we found the target, return the first step in the path
                        if dst == target_form:
                            logger.debug(f"  Found target '{target_form}' via path {new_path}")
                            return new_path[0]  # Return "physical"
                        
                        # Continue exploring from this form
                        queue.append((dst, new_path))
        
        logger.debug(f"No conversion path found to '{target_form}'. Visited: {visited}")
        return None

    def _can_convert(self, src_form: str, dst_form: str) -> bool:
        """
        Check if a conversion from src_form to dst_form is possible without computing it.
        
        Parameters
        ----------
        src_form : str
            Source form name
        dst_form : str
            Destination form name
            
        Returns
        -------
        bool
            True if conversion is possible, False otherwise
        """
        # Check direct conversion in registry
        if (src_form, dst_form) in self.registry._CONVERSION_REGISTRY:
            return True
            
        # Check if conversion adapter has the conversion
        if self.registry.conversion is not None:
            return self.registry.conversion.get(src_form, dst_form) is not None
            
        return False

    def _follow_conversion_path(self, start_form: str, target_form: str) -> "Hamiltonian":
        """
        Follow a multi-step conversion path from start_form to target_form.
        
        This method uses breadth-first search to find and execute the shortest
        conversion path between two Hamiltonian representations.
        
        Parameters
        ----------
        start_form : str
            The starting form (e.g., "physical")
        target_form : str
            The target form (e.g., "center_manifold_real")
            
        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            The converted Hamiltonian
            
        Raises
        ------
        NotImplementedError
            If no conversion path exists from start_form to target_form
            
        Notes
        -----
        The method finds the shortest path using BFS and then executes each
        conversion step, caching intermediate results for efficiency.
        """
        # Use BFS to find the shortest path
        from collections import deque
        
        queue = deque([(start_form, [start_form])])
        visited = {start_form}
        
        while queue:
            current_form, path = queue.popleft()
            
            # Check all possible conversions from current_form
            for (src, dst), (_, required_context, _) in self.registry._CONVERSION_REGISTRY.items():
                if src == current_form and dst not in visited:
                    # Check if we have the required context (point is always available)
                    if not required_context or "point" in required_context:
                        visited.add(dst)
                        new_path = path + [dst]
                        
                        # If we found the target, follow the path
                        if dst == target_form:
                            logger.debug(f"Following conversion path: {new_path}")
                            return self._execute_conversion_path(new_path)
                        
                        # Continue exploring from this form
                        queue.append((dst, new_path))
        
        raise NotImplementedError(f"No conversion path found from {start_form} to {target_form}")

    def _execute_conversion_path(self, path: list[str]) -> "Hamiltonian":
        """
        Execute a series of conversions along the given path.
        
        This method performs the actual conversions along a pre-computed
        path, caching intermediate results for efficiency.
        
        Parameters
        ----------
        path : list[str]
            List of form names representing the conversion path
            
        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            The final converted Hamiltonian
            
        Notes
        -----
        Each conversion step is performed using the to_state method of the
        current Hamiltonian, and intermediate results are cached to avoid
        redundant computation.
        """
        # Start with the first form
        current_ham = self.get_hamiltonian(path[0])
        
        # Convert step by step along the path
        for i in range(len(path) - 1):
            current_form = path[i]
            next_form = path[i + 1]
            
            logger.info(f"Converting {current_form} -> {next_form}")
            result = current_ham.to_state(next_form, point=self._point, _pipeline=self)
            
            # Handle tuple returns (Hamiltonian, LieGeneratingFunction)
            if isinstance(result, tuple):
                current_ham, generating_functions = result
                # Store generating functions if we have a pipeline reference
                if hasattr(self, '_store_generating_functions'):
                    self._store_generating_functions(next_form, generating_functions)
            else:
                current_ham = result
            
            # Cache the intermediate result
            self._hamiltonian_cache[next_form] = current_ham
        
        return current_ham

    def _get_available_forms(self) -> set[str]:
        """
        Get all available Hamiltonian forms.

        This method determines which Hamiltonian forms can be computed
        based on the conversion registry and available context using
        breadth-first search to find all reachable forms.

        Returns
        -------
        set[str]
            Set of all available form names

        Notes
        -----
        The method uses BFS to find all forms that can be reached from
        "physical" through any number of conversion steps, not just direct
        conversions. This ensures that multi-step conversion paths like
        physical -> real_modal -> complex_modal -> ... -> center_manifold_real
        are properly detected.
        """
        from collections import deque
        
        available = {"physical"}  # Always available
        queue = deque(["physical"])
        visited = {"physical"}
        
        while queue:
            current_form = queue.popleft()
            
            # Check all possible conversions from current_form using registry
            for (src, dst), (_, required_context, _) in self.registry._CONVERSION_REGISTRY.items():
                if src == current_form and dst not in visited:
                    # Check if we have the required context (point is always available)
                    if not required_context or "point" in required_context:
                        visited.add(dst)
                        available.add(dst)
                        queue.append(dst)

        return available

    def compute(self, form: str = "center_manifold_real") -> "Hamiltonian":
        """
        Compute and return a specific Hamiltonian representation.

        This method provides backward compatibility with the old CenterManifold
        interface, but returns Hamiltonian objects instead of raw coefficient lists.

        Parameters
        ----------
        form : str, optional
            The desired Hamiltonian form. Defaults to "center_manifold_real"

        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.Hamiltonian`
            The requested Hamiltonian representation

        Notes
        -----
        This method is equivalent to 
        :meth:`~hiten.system.hamiltonians.pipeline.HamiltonianPipeline.get_hamiltonian` 
        but provides a familiar interface for users migrating from the old CenterManifold.
        It is maintained for backward compatibility.
        """
        return self.get_hamiltonian(form)

    def get_hamsys(self, form: str):
        """
        Get the runtime Hamiltonian system for a specific form.

        This method retrieves the runtime Hamiltonian system that can be used
        for evaluation and integration of the Hamiltonian equations of motion.

        Parameters
        ----------
        form : str
            The Hamiltonian form to get the system for

        Returns
        -------
        :class:`~hiten.algorithms.dynamics.hamiltonian._HamiltonianSystem`
            The runtime Hamiltonian system

        Notes
        -----
        The runtime system is built lazily and cached for efficiency.
        It provides the actual integration interface for the Hamiltonian.
        """
        return self.get_hamiltonian(form).hamsys

    def cache_clear(self):
        """
        Clear the Hamiltonian cache.

        This forces recomputation of all Hamiltonian representations on
        the next call to :meth:`~hiten.system.hamiltonians.pipeline.HamiltonianPipeline.get_hamiltonian`.

        Notes
        -----
        This method clears both the Hamiltonian cache and the generating
        function cache. Use this when you need to force recomputation
        with different parameters or after modifying the pipeline state.
        """
        logger.debug("Clearing Hamiltonian and generating function caches")
        self._hamiltonian_cache.clear()
        self._generating_function_cache.clear()

    def list_forms(self) -> list[str]:
        """
        List all available Hamiltonian forms.

        This method returns a list of all Hamiltonian forms that can be
        computed with the current pipeline configuration.

        Returns
        -------
        list[str]
            List of available form names

        Notes
        -----
        The list includes all forms that can be converted from "physical"
        with the available context (point).
        """
        return list(self._get_available_forms())

    def has_form(self, form: str) -> bool:
        """
        Check if a specific form is available.

        This method checks whether a given Hamiltonian form can be computed
        with the current pipeline configuration.

        Parameters
        ----------
        form : str
            The form to check

        Returns
        -------
        bool
            True if the form is available, False otherwise

        Notes
        -----
        The method checks if the form can be converted from "physical"
        with the available context (point).
        """
        return form in self._get_available_forms()

    def get_generating_functions(self, transform_type: str = "partial", **kwargs) -> "LieGeneratingFunction":
        """
        Get Lie generating functions for coordinate transformations.

        This method retrieves the Lie generating functions used in canonical
        transformations for normal form calculations. The functions are cached
        for efficiency.

        Parameters
        ----------
        transform_type : str, optional
            Type of Lie transform. Options:
            - "partial": Partial normal form (center manifold)
            - "full": Full normal form
        **kwargs
            Additional parameters for the Lie transform:
            - tol_lie: float, default 1e-30
            - resonance_tol: float, default 1e-30 (for full transform only)

        Returns
        -------
        :class:`~hiten.system.hamiltonians.base.LieGeneratingFunction`
            The generating functions and eliminated terms

        Raises
        ------
        ValueError
            If transform_type is not supported

        Notes
        -----
        The generating functions are computed using Lie transform methods
        and are essential for canonical transformations that preserve the
        Hamiltonian structure.
        """
        if transform_type not in ["partial", "full"]:
            raise ValueError(f"Unsupported transform_type: {transform_type}. Use 'partial' or 'full'.")

        cache_key = f"generating_functions_{transform_type}"
        
        # Check if already cached
        if cache_key in self._generating_function_cache:
            logger.debug(f"Using cached generating functions for {transform_type} transform")
            return self._generating_function_cache[cache_key]
        
        # Try to trigger computation by requesting the corresponding Hamiltonian
        if transform_type == "partial":
            # Request complex_partial_normal to trigger Lie transform computation
            self.get_hamiltonian("complex_partial_normal")
        elif transform_type == "full":
            # Request complex_full_normal to trigger Lie transform computation
            self.get_hamiltonian("complex_full_normal")
        
        # Check cache again after potential computation
        if cache_key in self._generating_function_cache:
            return self._generating_function_cache[cache_key]
        
        # If still not cached, compute explicitly
        logger.debug(f"Computing generating functions for {transform_type} transform explicitly")
        self._generating_function_cache[cache_key] = self._compute_generating_functions(transform_type, **kwargs)
        return self._generating_function_cache[cache_key]

    def _compute_generating_functions(self, transform_type: str, **kwargs) -> "LieGeneratingFunction":
        """
        Compute Lie generating functions.

        This method performs the actual computation of Lie generating functions
        using the appropriate Lie transform algorithm.

        Parameters
        ----------
        transform_type : str
            Type of Lie transform ("partial" or "full")
        **kwargs
            Additional parameters for the Lie transform

        Returns
        -------
        :class:`~hiten.system.hamiltonian.LieGeneratingFunction`
            The computed generating functions

        Notes
        -----
        The method uses the complex modal Hamiltonian as the starting point
        and applies the appropriate Lie transform algorithm based on the
        transform type.
        """
        from hiten.system.hamiltonian import LieGeneratingFunction

        # Get the complex modal Hamiltonian as starting point
        complex_modal_ham = self.get_hamiltonian("complex_modal")
        
        if transform_type == "partial":
            tol_lie = kwargs.get("tol_lie", 1e-30)
            
            logger.debug(f"Computing partial Lie generating functions (tol_lie={tol_lie})")
            poly_trans, poly_G_total, poly_elim_total = _lie_transform_partial(
                self._point, 
                complex_modal_ham.poly_H, 
                complex_modal_ham.dynamics.psi, 
                complex_modal_ham.dynamics.clmo, 
                complex_modal_ham.degree, 
                tol=tol_lie
            )
            
        elif transform_type == "full":
            tol_lie = kwargs.get("tol_lie", 1e-30)
            resonance_tol = kwargs.get("resonance_tol", 1e-30)
            
            logger.debug(f"Computing full Lie generating functions (tol_lie={tol_lie}, resonance_tol={resonance_tol})")
            poly_trans, poly_G_total, poly_elim_total = _lie_transform_full(
                self._point,
                complex_modal_ham.poly_H,
                complex_modal_ham.dynamics.psi,
                complex_modal_ham.dynamics.clmo,
                complex_modal_ham.degree,
                tol=tol_lie,
                resonance_tol=resonance_tol
            )
        
        return LieGeneratingFunction(
            poly_G=poly_G_total,
            poly_elim=poly_elim_total,
            degree=complex_modal_ham.degree,
            ndof=complex_modal_ham.ndof
        )

    def get_lie_expansions(self, inverse: bool = False, tol: float = 1e-16) -> list:
        """
        Get Lie coordinate expansions for forward/inverse transformations.

        This method computes the polynomial expansions for coordinate
        transformations using Lie generating functions.

        Parameters
        ----------
        inverse : bool, default False
            If True, return inverse expansions (for coordinate reconstruction).
            If False, return forward expansions (for initial condition generation)
        tol : float, default 1e-16
            Numerical tolerance for the expansion computation

        Returns
        -------
        list
            List of polynomial expansions for each coordinate

        Notes
        -----
        The expansions are computed using the partial Lie generating functions
        and provide the coordinate transformation polynomials for both forward
        and inverse transformations.
        """
        # Get generating functions
        gen_funcs = self.get_generating_functions("partial")
        
        # Compute expansions
        sign = -1 if inverse else 1
        expansions = _lie_expansion(
            gen_funcs.poly_G,
            gen_funcs.degree,
            gen_funcs.dynamics.psi,
            gen_funcs.dynamics.clmo,
            tol,
            inverse=inverse,
            sign=sign,
            restrict=False,
        )
        
        return expansions
