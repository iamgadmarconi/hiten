from typing import Dict, Optional
from collections import deque

from hiten.algorithms.hamiltonian.center._lie import _lie_expansion
from hiten.algorithms.hamiltonian.center._lie import \
    _lie_transform as _lie_transform_partial
from hiten.algorithms.hamiltonian.hamiltonian import (
    _build_physical_hamiltonian_collinear,
    _build_physical_hamiltonian_triangular)
from hiten.algorithms.hamiltonian.normal._lie import \
    _lie_transform as _lie_transform_full
from hiten.system.hamiltonians.base import (_CONVERSION_REGISTRY, Hamiltonian,
                                            LieGeneratingFunction)
from hiten.system.libration.base import LibrationPoint
from hiten.system.libration.collinear import CollinearPoint
from hiten.system.libration.triangular import TriangularPoint
from hiten.utils.log_config import logger


class HamiltonianPipeline:
    r"""
    Manages the transformation pipeline for Hamiltonian representations.

    Parameters
    ----------
    point : LibrationPoint
        The libration point about which the normal form is computed.
    degree : int
        Maximum total degree of the polynomial truncation.

    Attributes
    ----------
    point : LibrationPoint
        The libration point about which the normal form is computed.
    degree : int
        The maximum total degree of the polynomial truncation.
    _hamiltonian_cache : dict
        Cache of computed Hamiltonian objects keyed by form name.

    Notes
    -----
    All heavy computations are cached and subsequent calls with the same
    parameters are inexpensive.
    """

    def __init__(self, point: LibrationPoint, degree: int):
        if not isinstance(degree, int) or degree <= 0:
            raise ValueError("degree must be a positive integer")

        self._point = point
        self._max_degree = degree
        self._hamiltonian_cache: Dict[str, Hamiltonian] = {}
        self._generating_function_cache: Dict[str, LieGeneratingFunction] = {}

        # Determine point-specific parameters
        if isinstance(self._point, CollinearPoint):
            self._build_hamiltonian = _build_physical_hamiltonian_collinear
            self._mix_pairs = (1, 2)
        elif isinstance(self._point, TriangularPoint):
            self._build_hamiltonian = _build_physical_hamiltonian_triangular
            self._mix_pairs = (0, 1, 2)
        else:
            raise ValueError(f"Unsupported libration point type: {type(self._point)}")

    @property
    def point(self) -> LibrationPoint:
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

    def get_hamiltonian(self, form: str) -> Hamiltonian:
        """
        Get a specific Hamiltonian representation.

        Parameters
        ----------
        form : str
            The desired Hamiltonian form. Supported forms include:
            - "physical": Physical coordinates
            - "real_modal": Real modal coordinates
            - "complex_modal": Complex modal coordinates
            - "complex_partial_normal": Complex partial normal form
            - "real_partial_normal": Real partial normal form
            - "center_manifold_complex": Complex center manifold
            - "center_manifold_real": Real center manifold
            - "complex_full_normal": Complex full normal form
            - "real_full_normal": Real full normal form

        Returns
        -------
        Hamiltonian
            The requested Hamiltonian representation.

        Raises
        ------
        ValueError
            If the requested form is not supported.
        NotImplementedError
            If no conversion path exists to the requested form.
        """
        if form not in self._hamiltonian_cache:
            self._hamiltonian_cache[form] = self._compute_hamiltonian(form)

        return self._hamiltonian_cache[form]

    def _store_generating_functions(self, form_name: str, generating_functions: LieGeneratingFunction):
        """
        Store generating functions in the cache.
        
        Parameters
        ----------
        form_name : str
            The Hamiltonian form name (e.g., "complex_partial_normal").
        generating_functions : LieGeneratingFunction
            The LieGeneratingFunction object to cache.
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

    def _compute_hamiltonian(self, form: str) -> Hamiltonian:
        """
        Compute a Hamiltonian representation, using conversion if needed.

        Parameters
        ----------
        form : str
            The desired Hamiltonian form.

        Returns
        -------
        Hamiltonian
            The computed Hamiltonian representation.
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
        
        # If we found a direct conversion, use it
        if (source_form, form) in _CONVERSION_REGISTRY:
            return source_ham.to_state(form, point=self._point, _pipeline=self)
        
        # Otherwise, follow the multi-step conversion path
        return self._follow_conversion_path(source_form, form)

    def _build_physical_hamiltonian(self) -> Hamiltonian:
        """
        Build the physical Hamiltonian from scratch.

        Returns
        -------
        Hamiltonian
            The physical Hamiltonian representation.
        """
        logger.debug(f"Building physical Hamiltonian for {self._point}")
        poly_H = self._build_hamiltonian(self._point, self._max_degree)
        return Hamiltonian(poly_H, self._max_degree, ndof=3, name="physical")

    def _find_conversion_source(self, target_form: str) -> Optional[str]:
        """
        Find a source form that can be converted to the target form.

        Parameters
        ----------
        target_form : str
            The target form to find a conversion source for.

        Returns
        -------
        str or None
            The source form that can be converted to target_form, or None if
            no conversion path exists.
        """
        # Check if we have a direct conversion from any existing form
        for source_form in self._hamiltonian_cache:
            if (source_form, target_form) in _CONVERSION_REGISTRY:
                return source_form

        # If no direct conversion, try building from physical
        if ("physical", target_form) in _CONVERSION_REGISTRY:
            return "physical"

        # debug: Print available conversions
        logger.debug(f"Looking for conversion path to '{target_form}'")
        logger.debug(f"Available conversions in registry: {list(_CONVERSION_REGISTRY.keys())}")
        
        queue = deque([("physical", ["physical"])])
        visited = {"physical"}
        
        while queue:
            current_form, path = queue.popleft()
            logger.debug(f"Exploring from '{current_form}' with path {path}")
            
            # Check all possible conversions from current_form
            for (src, dst), (_, required_context, _) in _CONVERSION_REGISTRY.items():
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

    def _follow_conversion_path(self, start_form: str, target_form: str) -> Hamiltonian:
        """
        Follow a multi-step conversion path from start_form to target_form.
        
        Parameters
        ----------
        start_form : str
            The starting form (e.g., "physical").
        target_form : str
            The target form (e.g., "center_manifold_real").
            
        Returns
        -------
        Hamiltonian
            The converted Hamiltonian.
        """
        # Use BFS to find the shortest path
        from collections import deque
        
        queue = deque([(start_form, [start_form])])
        visited = {start_form}
        
        while queue:
            current_form, path = queue.popleft()
            
            # Check all possible conversions from current_form
            for (src, dst), (_, required_context, _) in _CONVERSION_REGISTRY.items():
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

    def _execute_conversion_path(self, path: list[str]) -> Hamiltonian:
        """
        Execute a series of conversions along the given path.
        
        Parameters
        ----------
        path : list[str]
            List of form names representing the conversion path.
            
        Returns
        -------
        Hamiltonian
            The final converted Hamiltonian.
        """
        # Start with the first form
        current_ham = self.get_hamiltonian(path[0])
        
        # Convert step by step along the path
        for i in range(len(path) - 1):
            current_form = path[i]
            next_form = path[i + 1]
            
            logger.debug(f"Converting {current_form} -> {next_form}")
            current_ham = current_ham.to_state(next_form, point=self._point, _pipeline=self)
            
            # Cache the intermediate result
            self._hamiltonian_cache[next_form] = current_ham
        
        return current_ham

    def _get_available_forms(self) -> set[str]:
        """
        Get all available Hamiltonian forms.

        Returns
        -------
        set[str]
            Set of all available form names.
        """
        available = {"physical"}  # Always available
        
        # Add forms that can be converted from physical
        for (src, dst), (_, required_context, _) in _CONVERSION_REGISTRY.items():
            if src == "physical" and not required_context:
                available.add(dst)
            elif src == "physical" and "point" in required_context:
                available.add(dst)

        return available

    def compute(self, form: str = "center_manifold_real") -> Hamiltonian:
        """
        Compute and return a specific Hamiltonian representation.

        This method provides backward compatibility with the old CenterManifold
        interface, but returns Hamiltonian objects instead of raw coefficient lists.

        Parameters
        ----------
        form : str, optional
            The desired Hamiltonian form. Defaults to "center_manifold_real".

        Returns
        -------
        Hamiltonian
            The requested Hamiltonian representation.

        Notes
        -----
        This method is equivalent to :meth:`get_hamiltonian` but provides
        a familiar interface for users migrating from the old CenterManifold.
        """
        return self.get_hamiltonian(form)

    def get_hamsys(self, form: str):
        """
        Get the runtime Hamiltonian system for a specific form.

        Parameters
        ----------
        form : str
            The Hamiltonian form to get the system for.

        Returns
        -------
        _HamiltonianSystem
            The runtime Hamiltonian system.
        """
        return self.get_hamiltonian(form).hamsys

    def coefficients(self, form: str = "center_manifold_real", degree=None) -> str:
        """
        Get a formatted string representation of the Hamiltonian coefficients.

        Parameters
        ----------
        form : str, optional
            The Hamiltonian form to get coefficients for.
        degree : int or iterable, optional
            Degree filter for the coefficients.

        Returns
        -------
        str
            Formatted coefficient table.
        """
        from hiten.utils.printing import _format_poly_table
        
        ham = self.get_hamiltonian(form)
        table = _format_poly_table(ham.poly_H, ham._clmo, degree)
        logger.debug(f'{form} coefficients:\n\n{table}\n\n')
        return table

    def cache_clear(self):
        """
        Clear the Hamiltonian cache.

        This forces recomputation of all Hamiltonian representations on
        the next call to :meth:`get_hamiltonian`.
        """
        logger.debug("Clearing Hamiltonian and generating function caches")
        self._hamiltonian_cache.clear()
        self._generating_function_cache.clear()

    def list_forms(self) -> list[str]:
        """
        List all available Hamiltonian forms.

        Returns
        -------
        list[str]
            List of available form names.
        """
        return list(self._get_available_forms())

    def has_form(self, form: str) -> bool:
        """
        Check if a specific form is available.

        Parameters
        ----------
        form : str
            The form to check.

        Returns
        -------
        bool
            True if the form is available, False otherwise.
        """
        return form in self._get_available_forms()

    def get_generating_functions(self, transform_type: str = "partial", **kwargs) -> LieGeneratingFunction:
        """
        Get Lie generating functions for coordinate transformations.

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
        LieGeneratingFunction
            The generating functions and eliminated terms.

        Raises
        ------
        ValueError
            If transform_type is not supported.
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

    def _compute_generating_functions(self, transform_type: str, **kwargs) -> LieGeneratingFunction:
        """
        Compute Lie generating functions.

        Parameters
        ----------
        transform_type : str
            Type of Lie transform ("partial" or "full").
        **kwargs
            Additional parameters.

        Returns
        -------
        LieGeneratingFunction
            The computed generating functions.
        """
        # Get the complex modal Hamiltonian as starting point
        complex_modal_ham = self.get_hamiltonian("complex_modal")
        
        if transform_type == "partial":
            tol_lie = kwargs.get("tol_lie", 1e-30)
            
            logger.debug(f"Computing partial Lie generating functions (tol_lie={tol_lie})")
            poly_trans, poly_G_total, poly_elim_total = _lie_transform_partial(
                self._point, 
                complex_modal_ham.poly_H, 
                complex_modal_ham._psi, 
                complex_modal_ham._clmo, 
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
                complex_modal_ham._psi,
                complex_modal_ham._clmo,
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

        Parameters
        ----------
        inverse : bool, default False
            If True, return inverse expansions (for coordinate reconstruction).
            If False, return forward expansions (for initial condition generation).
        tol : float, default 1e-16
            Numerical tolerance for the expansion computation.

        Returns
        -------
        list
            List of polynomial expansions for each coordinate.
        """
        # Get generating functions
        gen_funcs = self.get_generating_functions("partial")
        
        # Compute expansions
        sign = -1 if inverse else 1
        expansions = _lie_expansion(
            gen_funcs.poly_G,
            gen_funcs.degree,
            gen_funcs._psi,
            gen_funcs._clmo,
            tol,
            inverse=inverse,
            sign=sign,
            restrict=False,
        )
        
        return expansions
