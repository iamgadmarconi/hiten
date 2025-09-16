"""
hiten.system.libration.triangular
==========================

Triangular Libration points (L4 and L5) of the Circular Restricted Three-Body Problem (CR3BP).

This module provides concrete implementations of the triangular libration points
in the Circular Restricted Three-Body Problem. These points form equilateral
triangles with the two primary bodies and are characterized by center-type
stability when the mass ratio is below Routh's critical value.

Classes
-------
:class:`~hiten.system.libration.triangular.TriangularPoint`
    Abstract base class for triangular libration points.
:class:`~hiten.system.libration.triangular.L4Point`
    L4 libration point located above the x-axis (positive y).
:class:`~hiten.system.libration.triangular.L5Point`
    L5 libration point located below the x-axis (negative y).

Notes
-----
All positions and distances are expressed in nondimensional units where the
distance between the primaries is unity and the orbital period is 2*pi.
"""

from typing import TYPE_CHECKING

import numpy as np

from hiten.system.libration.base import LibrationPoint, LinearData
from hiten.utils.log_config import logger

if TYPE_CHECKING:
    from hiten.system.base import System


class TriangularPoint(LibrationPoint):
    """
    Abstract helper for the triangular Libration points.

    The triangular points form equilateral triangles with the two primary
    bodies. They behave as centre-type equilibria when the mass ratio
    mu is below Routh's critical value.

    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        CR3BP model supplying the mass parameter mu.

    Attributes
    ----------
    mu : float
        Mass ratio mu = m2 / (m1 + m2) taken from system (dimensionless).
    ROUTH_CRITICAL_MU : float
        Critical value mu_R delimiting linear stability (dimensionless).
    sign : int
        +1 for :class:`~hiten.system.libration.triangular.L4Point`, -1 
        for :class:`~hiten.system.libration.triangular.L5Point`.
    a : float
        Offset used by local <-> synodic frame transformations (dimensionless).

    Notes
    -----
    A warning is logged if mu > mu_R.
    """
    ROUTH_CRITICAL_MU = (1.0 - np.sqrt(1.0 - (1.0/27.0))) / 2.0 # approx 0.03852
    
    def __init__(self, system: "System"):
        super().__init__(system)
        # Log stability warning based on mu
        if system.mu > self.ROUTH_CRITICAL_MU:
            logger.warning(f"Triangular points are potentially unstable for mu > {self.ROUTH_CRITICAL_MU:.6f} (current mu = {system.mu})")

    @property
    def sign(self) -> int:
        """
        Sign convention distinguishing L4 and L5.

        Returns
        -------
        int
            +1 for :class:`~hiten.system.libration.triangular.L4Point`, -1 for :class:`~hiten.system.libration.triangular.L5Point`.
        """
        return 1 if isinstance(self, L4Point) else -1
    
    @property
    def a(self) -> float:
        """
        Offset a along the x axis used in frame changes.
        
        Returns
        -------
        float
            The offset value a (dimensionless).
        """
        return self.sign * 3 * np.sqrt(3) / 4 * (1 - 2 * self.mu)

    def _calculate_position(self) -> np.ndarray:
        """
        Calculate the position of a triangular point (L4 or L5).
        
        Returns
        -------
        numpy.ndarray, shape (3,)
            3D vector [x, y, 0] giving the position in nondimensional units.
        """
        point_name = self.__class__.__name__
        logger.debug(f"Calculating {point_name} position directly.")
        
        x = 0.5 - self.mu
        y = self.sign * np.sqrt(3) / 2.0
        
        logger.info(f"{point_name} position calculated: x = {x:.6f}, y = {y:.6f}")
        return np.array([x, y, 0], dtype=np.float64)

    def _get_linear_data(self) -> LinearData:
        """
        Get the linear data for the Libration point.
        
        Returns
        -------
        :class:`~hiten.system.libration.LinearData`
            Object containing the linear data for the Libration point.
        """
        # Frequencies and canonical transform
        omega1, omega2, omega_z = self.linear_modes
        C, Cinv = self.normal_form_transform()
        
        # Create and return the LinearData object
        return LinearData(
            mu=self.mu,
            point=type(self).__name__[:2],  # 'L1', 'L2', 'L3'
            lambda1=None, 
            omega1=omega1, 
            omega2=omega2,
            omega3=omega_z,
            C=C, 
            Cinv=Cinv
        )

    def normal_form_transform(self):
        """
        Build the 6x6 symplectic matrix C that sends H2 to normal form.

        Returns
        -------
        tuple
            (C, Cinv) where C is the symplectic transformation matrix and Cinv is its inverse.
        """
        cache_key = ('normal_form_transform',)
        cached = self.cache_get(cache_key)
        if cached is not None:
            return cached

        # Canonical eigenvectors (rows) -> columns after transpose
        eigvs = self._get_eigvs().T  # shape (6,6), columns are the vectors

        # Scaling factors s_j
        s = np.array([self._scaling_factor(0),
                      self._scaling_factor(1),
                      self._scaling_factor(2)])

        # Build diagonal scaling matrix for u and v blocks
        S = np.diag(np.concatenate([s, s]))

        C = eigvs @ np.linalg.inv(S)  # divide each column by its s_j

        # Pre-compute inverse once - safer & cheaper later on
        Cinv = np.linalg.inv(C)

        self.cache_set(cache_key, (C, Cinv))
        return C, Cinv
    
    def _compute_linear_modes(self):
        """
        Compute the three frequencies (omega_1, omega_2, omega_z) following the convention:
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

        # Extract purely imaginary eigenvalues (should be 6 values: +- i omega)
        imag_eigs = eigvals[np.abs(eigvals.real) < 1e-12]
        omegas_with_sign = imag_eigs.imag  # Keep the signs

        # Get unique frequencies (positive and negative)
        omegas_unique = []
        for omega in omegas_with_sign:
            if not any(np.isclose(omega, existing, atol=1e-12) for existing in omegas_unique):
                omegas_unique.append(omega)

        if len(omegas_unique) != 6:
            raise RuntimeError(f"Expected 6 eigenvalues (+-3 frequencies), got {len(omegas_unique)}.")

        # Identify the vertical frequency (close to +-1)
        # Group frequencies by their absolute values to handle numerical duplicates
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
        
        # Find the group closest to 1.0 (vertical frequency)
        vertical_group_key = min(freq_groups.keys(), key=lambda x: abs(x - 1.0))
        if not np.isclose(vertical_group_key, 1.0, rtol=1e-2):
            raise RuntimeError(f"No frequency group found near 1.0, closest is {vertical_group_key}")
        
        omega_z = vertical_group_key
        vertical_omegas = freq_groups[vertical_group_key]
        
        # Get planar frequencies (all other groups)
        planar_omegas = []
        for key, omegas_list in freq_groups.items():
            if not np.isclose(key, vertical_group_key, rtol=1e-10):
                planar_omegas.extend(omegas_list)
        
        # Get the distinct planar frequency magnitudes (should be 2 groups, each with +- pairs)
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

        # Get the two planar frequency magnitudes and sort them
        planar_mags = sorted(planar_freq_groups.keys())
        smaller_mag, larger_mag = planar_mags
        
        # Apply the updated convention described above.
        # Ideally we have |smaller| < sqrt(1/2) < |larger|, but we still
        # enforce sign ordering even if the threshold split fails (rare).

        omega1 = larger_mag           # positive, expected > sqrt(1/2)
        omega2 = -smaller_mag         # negative, expected < -sqrt(1/2)

        # Sanity check - warn (do not fail) if the magnitudes violate the desired split
        if not (omega1**2 > 0.5 and omega2**2 < 0.5):
            raise RuntimeError(f"Computed planar frequencies do not strictly satisfy the requested ordering: omega_1={omega1:.4f}, omega_2={omega2:.4f}.")

        return (float(omega1), float(omega2), float(omega_z))

    @property
    def linear_modes(self):
        """
        Get the linear modes for the Libration point.
        
        Returns
        -------
        tuple
            (omega_1, omega_2, omega_z) where:
            - omega_1 > 0 with omega_1^2 < 1/2 (small positive planar frequency)
            - omega_2 < 0 (negative planar frequency)  
            - omega_z = 1.0 (vertical frequency)
            For triangular points all eigenvalues are purely imaginary so no
            hyperbolic mode is present.
        """
        cached = self.cache_get(('linear_modes',))
        if cached is not None:
            return cached
            
        result = self._compute_linear_modes()
        self.cache_set(('linear_modes',), result)
        return result

    def _J_hess_H2(self) -> np.ndarray:
        """
        Compute the 6x6 symplectic matrix for the quadratic Hamiltonian H2.
        
        Returns
        -------
        numpy.ndarray, shape (6, 6)
            The symplectic matrix J for the quadratic Hamiltonian.
        """
        # Planar 4x4 block (x, y, p_x, p_y)
        J_planar = np.array([
            [0.0, 1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
            [-0.25, self.a, 0.0, 1.0],
            [self.a, 1.25, -1.0, 0.0],
        ], dtype=np.float64)

        # Vertical 2x2 block: simple harmonic oscillator with omega_z = 1
        J_vert = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)

        # Assemble full 6x6 Jacobian 
        J_full = np.zeros((6, 6), dtype=np.float64)
        J_full[:4, :4] = J_planar
        J_full[4:, 4:] = J_vert
        return J_full # x, y, p_x, p_y, z, p_z
    
    def _rs(self, idx):
        """
        Compute the scaling factor for the given mode index.
        
        Parameters
        ----------
        idx : int
            The mode index (0 or 1 for planar modes).
            
        Returns
        -------
        float
            The scaling factor (dimensionless).
            
        Raises
        ------
        ValueError
            If idx is not 0 or 1.
        """
        if idx not in [0, 1]:
            raise ValueError(f"Invalid index {idx} for scaling factor calculation")
        return np.sqrt(self.linear_modes[idx] * (4*self.linear_modes[idx]**4 + self.linear_modes[idx]**2 - 1.5))

    def _get_eigvs(self):
        """
        Return the six real eigenvectors (u_1, u_2, u_3, v_1, v_2, v_3)
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

        # The vectors are written in the (x, y, p_x, p_y) ordering used by
        # _J_hess_H2.  They are then embedded into 6-D by appending zeros
        # for the vertical coordinates (z, p_z). 
        u1_planar = np.array([a, -omega1**2 - 0.75, -omega1**2 + 0.75, a])
        u2_planar = np.array([a, -omega2**2 - 0.75, -omega2**2 + 0.75, a])
        v1_planar = np.array([2 * omega1, 0.0, a * omega1, -omega1**3 + 1.25 * omega1])
        v2_planar = np.array([2 * omega2, 0.0, a * omega2, -omega2**3 + 1.25 * omega2])

        # Build full 6-D vectors with zeros in the vertical coordinates
        # (z, p_z) = (index 2, 5).
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
    
    def _d_omega(self, idx):
        """
        Compute the derivative term for the given mode index.
        
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
    
    def _scaling_factor(self, idx):
        """
        Compute the scaling factor for the given mode index.
        
        Parameters
        ----------
        idx : int
            The mode index (0, 1 for planar modes, 2 for vertical).
            
        Returns
        -------
        float
            The scaling factor (dimensionless).
        """
        # Planar modes: use Gomez et al. formula.  Vertical mode: already
        # rescaled directly in the eigenvectors, so the scaling factor is 1.
        if idx == 2:
            return 1.0
        return np.sqrt(self._d_omega(idx))


class L4Point(TriangularPoint):
    """
    L4 Libration point, forming an equilateral triangle with the two primary bodies,
    located above the x-axis (positive y).
    
    The L4 point is situated above the x-axis, forming an equilateral triangle
    with the two primary bodies. It is characterized by center-type stability
    when the mass ratio is below Routh's critical value.
    
    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system containing the mass parameter mu.
    """
    
    def __init__(self, system: "System"):
        super().__init__(system)
    
    @property
    def idx(self) -> int:
        """Get the libration point index.
        
        Returns
        -------
        int
            The libration point index (4 for L4).
        """
        return 4


class L5Point(TriangularPoint):
    """
    L5 Libration point, forming an equilateral triangle with the two primary bodies,
    located below the x-axis (negative y).
    
    The L5 point is situated below the x-axis, forming an equilateral triangle
    with the two primary bodies. It is characterized by center-type stability
    when the mass ratio is below Routh's critical value.
    
    Parameters
    ----------
    system : :class:`~hiten.system.base.System`
        The CR3BP system containing the mass parameter mu.
    """
    
    def __init__(self, system: "System"):
        super().__init__(system)
    
    @property
    def idx(self) -> int:
        """Get the libration point index.
        
        Returns
        -------
        int
            The libration point index (5 for L5).
        """
        return 5
