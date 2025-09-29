"""Tests for the LibrationPoint class API in libration/."""

import numpy as np
import pytest
from pathlib import Path

from hiten.system.base import System
from hiten.system.body import Body
from hiten.system.libration.collinear import L1Point, L2Point, L3Point
from hiten.system.libration.triangular import L4Point, L5Point
from hiten.utils.constants import Constants

TEST_MU_EARTH_MOON = 0.01215  # Earth-Moon system
TEST_MU_SUN_EARTH = 3.00348e-6  # Sun-Earth system
TEST_MU_SUN_JUPITER = 9.5387e-4  # Sun-Jupiter system


def is_symplectic(matrix, tol=1e-10):
    """
    Check if a 6x6 matrix is symplectic by verifying M^T J M = J
    where J is the standard symplectic matrix.
    """
    # Standard symplectic matrix J
    J = np.zeros((6, 6))
    n = 3  # 3 degrees of freedom
    for i in range(n):
        J[i, i+n] = 1
        J[i+n, i] = -1
    
    # Calculate M^T J M
    M_T_J_M = matrix.T @ J @ matrix
    
    # Check if M^T J M = J
    return np.allclose(M_T_J_M, J, atol=tol)


@pytest.fixture
def system_earth_moon():   
    """Create an Earth-Moon system for testing."""
    earth_mass = Constants.get_mass("earth")
    earth_radius = Constants.get_radius("earth")
    moon_mass = Constants.get_mass("moon")
    moon_radius = Constants.get_radius("moon")
    distance = Constants.get_orbital_distance("earth", "moon")

    earth = Body("Earth", earth_mass, earth_radius, color="blue")
    moon = Body("Moon", moon_mass, moon_radius, "gray", earth)

    return System(earth, moon, distance)


@pytest.fixture
def system_sun_earth():
    """Create a Sun-Earth system for testing."""
    sun_mass = Constants.get_mass("sun")
    sun_radius = Constants.get_radius("sun")
    earth_mass = Constants.get_mass("earth")
    earth_radius = Constants.get_radius("earth")
    distance = Constants.get_orbital_distance("sun", "earth")

    sun = Body("Sun", sun_mass, sun_radius, color="yellow")
    earth = Body("Earth", earth_mass, earth_radius, "blue", sun)

    return System(sun, earth, distance)


@pytest.fixture
def system_sun_jupiter():
    """Create a Sun-Jupiter system for testing."""
    sun_mass = Constants.get_mass("sun")
    sun_radius = Constants.get_radius("sun")
    jupiter_mass = Constants.get_mass("jupiter")
    jupiter_radius = Constants.get_radius("jupiter")
    distance = Constants.get_orbital_distance("sun", "jupiter")

    sun = Body("Sun", sun_mass, sun_radius, color="yellow")
    jupiter = Body("Jupiter", jupiter_mass, jupiter_radius, "gray", sun)
    return System(sun, jupiter, distance)


@pytest.fixture
def l1_earth_moon(system_earth_moon):
    """Create L1 point for Earth-Moon system."""
    return system_earth_moon.get_libration_point(1)


@pytest.fixture
def l2_earth_moon(system_earth_moon):
    """Create L2 point for Earth-Moon system."""
    return system_earth_moon.get_libration_point(2)


@pytest.fixture
def l3_earth_moon(system_earth_moon):
    """Create L3 point for Earth-Moon system."""
    return system_earth_moon.get_libration_point(3)


@pytest.fixture
def l4_earth_moon(system_earth_moon):
    """Create L4 point for Earth-Moon system."""
    return system_earth_moon.get_libration_point(4)


@pytest.fixture
def l5_earth_moon(system_earth_moon):
    """Create L5 point for Earth-Moon system."""
    return system_earth_moon.get_libration_point(5)


@pytest.fixture
def l1_sun_earth(system_sun_earth):
    """Create L1 point for Sun-Earth system."""
    return system_sun_earth.get_libration_point(1)


@pytest.fixture
def l2_sun_earth(system_sun_earth):
    """Create L2 point for Sun-Earth system."""
    return system_sun_earth.get_libration_point(2)


@pytest.fixture
def l3_sun_jupiter(system_sun_jupiter):
    """Create L3 point for Sun-Jupiter system."""
    return system_sun_jupiter.get_libration_point(3)


class TestLibrationPointInitialization:
    """Test LibrationPoint class initialization and basic properties."""
    
    def test_libration_point_initialization(self):
        """Test initialization of different libration points."""
        
        def create_mock_system(mu):
            """Create a mock system with the given mu."""
            primary = Body("p", 1 - mu, 0.1)
            secondary = Body("s", mu, 0.1, "gray", primary)
            return System(primary, secondary, 1.0)

        l1_earth_moon = L1Point(create_mock_system(TEST_MU_EARTH_MOON))
        assert l1_earth_moon.mu == pytest.approx(TEST_MU_EARTH_MOON, rel=1e-5)
        
        l2_sun_earth = L2Point(create_mock_system(TEST_MU_SUN_EARTH))
        assert l2_sun_earth.mu == pytest.approx(TEST_MU_SUN_EARTH, rel=1e-5)
        
        l3_sun_jupiter = L3Point(create_mock_system(TEST_MU_SUN_JUPITER))
        assert l3_sun_jupiter.mu == pytest.approx(TEST_MU_SUN_JUPITER, rel=1e-5)
        
        l4_earth_moon = L4Point(create_mock_system(TEST_MU_EARTH_MOON))
        assert l4_earth_moon.mu == pytest.approx(TEST_MU_EARTH_MOON, rel=1e-5)
        
        l5_sun_earth = L5Point(create_mock_system(TEST_MU_SUN_EARTH))
        assert l5_sun_earth.mu == pytest.approx(TEST_MU_SUN_EARTH, rel=1e-5)
    
    def test_libration_point_from_system(self, system_earth_moon):
        """Test creating libration points from a system."""
        l1 = system_earth_moon.get_libration_point(1)
        l2 = system_earth_moon.get_libration_point(2)
        l3 = system_earth_moon.get_libration_point(3)
        l4 = system_earth_moon.get_libration_point(4)
        l5 = system_earth_moon.get_libration_point(5)
        
        assert l1 is not None
        assert l2 is not None
        assert l3 is not None
        assert l4 is not None
        assert l5 is not None
        
        # All should have the same mu as the system
        assert l1.mu == system_earth_moon.mu
        assert l2.mu == system_earth_moon.mu
        assert l3.mu == system_earth_moon.mu
        assert l4.mu == system_earth_moon.mu
        assert l5.mu == system_earth_moon.mu
    
    def test_libration_point_system_reference(self, system_earth_moon):
        """Test that libration points reference their parent system correctly."""
        l1 = system_earth_moon.get_libration_point(1)
        
        # Should reference the System object, not the dynamics service
        assert l1.system is system_earth_moon
        assert isinstance(l1.system, System)


class TestLibrationPointProperties:
    """Test LibrationPoint class properties."""
    
    def test_mu_property(self, l1_earth_moon, l2_sun_earth):
        """Test mu property access."""
        assert 0 < l1_earth_moon.mu < 1
        assert 0 < l2_sun_earth.mu < 1
        assert isinstance(l1_earth_moon.mu, float)
        assert isinstance(l2_sun_earth.mu, float)
    
    def test_position_property(self, l1_earth_moon, l2_earth_moon, l3_earth_moon):
        """Test position property access."""
        pos_l1 = l1_earth_moon.position
        pos_l2 = l2_earth_moon.position
        pos_l3 = l3_earth_moon.position
        
        # All positions should be numpy arrays of shape (3,)
        assert isinstance(pos_l1, np.ndarray)
        assert isinstance(pos_l2, np.ndarray)
        assert isinstance(pos_l3, np.ndarray)
        
        assert pos_l1.shape == (3,)
        assert pos_l2.shape == (3,)
        assert pos_l3.shape == (3,)
    
    def test_energy_property(self, l1_earth_moon, l2_earth_moon):
        """Test energy property."""
        energy_l1 = l1_earth_moon.energy
        energy_l2 = l2_earth_moon.energy
        
        assert isinstance(energy_l1, float)
        assert isinstance(energy_l2, float)
    
    def test_jacobi_property(self, l1_earth_moon, l2_earth_moon):
        """Test Jacobi constant property."""
        cj_l1 = l1_earth_moon.jacobi
        cj_l2 = l2_earth_moon.jacobi
        
        assert isinstance(cj_l1, float)
        assert isinstance(cj_l2, float)
        
        # CJ = -2E
        assert np.isclose(cj_l1, -2 * l1_earth_moon.energy)
        assert np.isclose(cj_l2, -2 * l2_earth_moon.energy)
    
    def test_dynsys_property(self, l1_earth_moon):
        """Test dynsys property access."""
        dynsys = l1_earth_moon.dynsys
        assert dynsys is not None
        assert hasattr(dynsys, 'mu')
    
    def test_is_stable_property(self, l1_earth_moon, l2_earth_moon, l4_earth_moon):
        """Test is_stable property."""
        # Collinear points are unstable
        assert l1_earth_moon.is_stable == False
        assert l2_earth_moon.is_stable == False
        
        # Triangular points may be stable or unstable depending on mu
        is_stable_l4 = l4_earth_moon.is_stable
        assert isinstance(is_stable_l4, bool)
    
    def test_eigenvalues_property(self, l1_earth_moon):
        """Test eigenvalues property."""
        evals_stable, evals_unstable, evals_center = l1_earth_moon.eigenvalues
        
        # For L1, should have stable, unstable, and center eigenvalues
        assert isinstance(evals_stable, np.ndarray)
        assert isinstance(evals_unstable, np.ndarray)
        assert isinstance(evals_center, np.ndarray)
    
    def test_eigenvectors_property(self, l1_earth_moon):
        """Test eigenvectors property."""
        evecs_stable, evecs_unstable, evecs_center = l1_earth_moon.eigenvectors
        
        # Should return arrays of eigenvectors
        assert isinstance(evecs_stable, np.ndarray)
        assert isinstance(evecs_unstable, np.ndarray)
        assert isinstance(evecs_center, np.ndarray)


class TestCollinearPoints:
    """Test collinear libration points (L1, L2, L3)."""
    
    def test_positions(self, l1_earth_moon, l2_earth_moon, l3_earth_moon, l4_earth_moon, l5_earth_moon):
        """Test computation of libration point positions."""
        mu = TEST_MU_EARTH_MOON
        
        # L1: Between primaries
        pos_l1 = l1_earth_moon.position
        assert -mu < pos_l1[0] < 1 - mu
        assert np.isclose(pos_l1[1], 0)
        assert np.isclose(pos_l1[2], 0)
        
        # L2: Beyond secondary
        pos_l2 = l2_earth_moon.position
        assert pos_l2[0] > 1 - mu
        assert np.isclose(pos_l2[1], 0)
        assert np.isclose(pos_l2[2], 0)
        
        # L3: Beyond primary
        pos_l3 = l3_earth_moon.position
        assert pos_l3[0] < -mu
        assert np.isclose(pos_l3[1], 0)
        assert np.isclose(pos_l3[2], 0)
        
        # L4: Leading triangular point
        pos_l4 = l4_earth_moon.position
        assert np.isclose(pos_l4[0], 0.5 - mu)
        assert np.isclose(pos_l4[1], np.sqrt(3)/2)
        assert np.isclose(pos_l4[2], 0)
        
        # L5: Trailing triangular point
        pos_l5 = l5_earth_moon.position
        assert np.isclose(pos_l5[0], 0.5 - mu)
        assert np.isclose(pos_l5[1], -np.sqrt(3)/2)
        assert np.isclose(pos_l5[2], 0)
    
    def test_gamma_values(self, l1_earth_moon, l2_earth_moon, l3_earth_moon):
        """Test gamma (distance ratio) calculations for collinear points."""
        gamma_l1 = l1_earth_moon.gamma
        assert gamma_l1 > 0
        assert gamma_l1 < 1.0
        
        gamma_l2 = l2_earth_moon.gamma
        assert gamma_l2 > 0
        assert gamma_l2 < 1.0
        
        gamma_l3 = l3_earth_moon.gamma
        assert gamma_l3 > 0
        expected_gamma_l3 = 1.0 - (7.0/12.0) * TEST_MU_EARTH_MOON
        assert np.isclose(gamma_l3, expected_gamma_l3, rtol=0.1)
    
    def test_cn_coefficients(self, l1_earth_moon, l2_earth_moon, l3_earth_moon, l1_sun_earth, l2_sun_earth, l3_sun_jupiter):
        """Test calculation of cn coefficients for collinear points."""
        c2_l1_em = l1_earth_moon.dynamics.cn(2)
        c2_l2_em = l2_earth_moon.dynamics.cn(2)
        c2_l3_em = l3_earth_moon.dynamics.cn(2)

        c2_l1_se = l1_sun_earth.dynamics.cn(2)
        c2_l2_se = l2_sun_earth.dynamics.cn(2)
        c2_l3_sj = l3_sun_jupiter.dynamics.cn(2)

        # c2 should be greater than 1 for collinear points
        assert c2_l1_em > 1.0
        assert c2_l2_em > 1.0
        assert c2_l3_em > 1.0

        assert c2_l1_se > 1.0
        assert c2_l2_se > 1.0
        assert c2_l3_sj > 1.0
    
    def test_linear_modes(self, l1_earth_moon, l2_earth_moon, l3_earth_moon):
        """Test calculation of linear modes for collinear points."""
        lambda1, omega1, omega2 = l1_earth_moon.linear_modes

        assert lambda1 > 0
        assert omega1 > 0
        assert omega2 > 0

        c2 = l1_earth_moon.dynamics.cn(2)
        
        discriminant = 9 * c2**2 - 8 * c2
        eta1 = (c2 - 2 - np.sqrt(discriminant)) / 2
        eta2 = (c2 - 2 + np.sqrt(discriminant)) / 2
        
        assert eta1 < 0, "Expected eta1 < 0 for collinear points"
        assert eta2 > 0, "Expected eta2 > 0 for collinear points"
        
        expected_lambda1 = np.sqrt(eta2)
        expected_omega1 = np.sqrt(-eta1)
        expected_omega2 = np.sqrt(c2)
        
        assert np.isclose(lambda1, expected_lambda1, rtol=1e-5), f"lambda1 should be {expected_lambda1}, got {lambda1}"
        assert np.isclose(omega1, expected_omega1, rtol=1e-5), f"omega1 should be {expected_omega1}, got {omega1}"
        assert np.isclose(omega2, expected_omega2, rtol=1e-5), f"omega2 should be {expected_omega2}, got {omega2}"
        
        # Test L2
        lambda1_l2, omega1_l2, omega2_l2 = l2_earth_moon.linear_modes

        assert lambda1_l2 > 0
        assert omega1_l2 > 0
        assert omega2_l2 > 0

        c2_l2 = l2_earth_moon.dynamics.cn(2)
        
        discriminant_l2 = 9 * c2_l2**2 - 8 * c2_l2
        eta1_l2 = (c2_l2 - 2 - np.sqrt(discriminant_l2)) / 2
        eta2_l2 = (c2_l2 - 2 + np.sqrt(discriminant_l2)) / 2
        
        assert eta1_l2 < 0
        assert eta2_l2 > 0
        
        expected_lambda1_l2 = np.sqrt(eta2_l2)
        expected_omega1_l2 = np.sqrt(-eta1_l2)
        expected_omega2_l2 = np.sqrt(c2_l2)
        
        assert np.isclose(lambda1_l2, expected_lambda1_l2, rtol=1e-5)
        assert np.isclose(omega1_l2, expected_omega1_l2, rtol=1e-5)
        assert np.isclose(omega2_l2, expected_omega2_l2, rtol=1e-5)
        
        # Test L3
        lambda1_l3, omega1_l3, omega2_l3 = l3_earth_moon.linear_modes

        assert lambda1_l3 > 0
        assert omega1_l3 > 0
        assert omega2_l3 > 0

        c2_l3 = l3_earth_moon.dynamics.cn(2)
        
        discriminant_l3 = 9 * c2_l3**2 - 8 * c2_l3
        eta1_l3 = (c2_l3 - 2 - np.sqrt(discriminant_l3)) / 2
        eta2_l3 = (c2_l3 - 2 + np.sqrt(discriminant_l3)) / 2
        
        assert eta1_l3 < 0
        assert eta2_l3 > 0
        
        expected_lambda1_l3 = np.sqrt(eta2_l3)
        expected_omega1_l3 = np.sqrt(-eta1_l3)
        expected_omega2_l3 = np.sqrt(c2_l3)
        
        assert np.isclose(lambda1_l3, expected_lambda1_l3, rtol=1e-5)
        assert np.isclose(omega1_l3, expected_omega1_l3, rtol=1e-5)
        assert np.isclose(omega2_l3, expected_omega2_l3, rtol=1e-5)
    
    def test_scale_factors(self, l1_earth_moon, l2_earth_moon, l3_earth_moon):
        """Test that scale factors s1 and s2 are always positive."""
        lambda1, omega1, omega2 = l1_earth_moon.linear_modes
        s1, s2 = l1_earth_moon.dynamics.scale_factor(lambda1, omega1)
        
        assert s1 > 0, "s1 scale factor should be positive"
        assert s2 > 0, "s2 scale factor should be positive"
        
        lambda1_l2, omega1_l2, omega2_l2 = l2_earth_moon.linear_modes
        s1_l2, s2_l2 = l2_earth_moon.dynamics.scale_factor(lambda1_l2, omega1_l2)
        
        assert s1_l2 > 0, "s1 scale factor should be positive for L2"
        assert s2_l2 > 0, "s2 scale factor should be positive for L2"
        
        lambda1_l3, omega1_l3, omega2_l3 = l3_earth_moon.linear_modes
        s1_l3, s2_l3 = l3_earth_moon.dynamics.scale_factor(lambda1_l3, omega1_l3)
        
        assert s1_l3 > 0, "s1 scale factor should be positive for L3"
        assert s2_l3 > 0, "s2 scale factor should be positive for L3"


class TestNormalFormTransform:
    """Test normal form transformations."""
    
    def test_normal_form_transform(self, l1_earth_moon, l2_earth_moon, l3_earth_moon, l4_earth_moon, l5_earth_moon):
        """Test normal form transform for all libration points."""
        C_l1, Cinv_l1 = l1_earth_moon.normal_form_transform
        assert is_symplectic(C_l1)
        assert is_symplectic(Cinv_l1)

        C_l2, Cinv_l2 = l2_earth_moon.normal_form_transform
        assert is_symplectic(C_l2)
        assert is_symplectic(Cinv_l2)

        C_l3, Cinv_l3 = l3_earth_moon.normal_form_transform
        assert is_symplectic(C_l3)
        assert is_symplectic(Cinv_l3)

        C_l4, Cinv_l4 = l4_earth_moon.normal_form_transform
        assert is_symplectic(C_l4)
        assert is_symplectic(Cinv_l4)

        C_l5, Cinv_l5 = l5_earth_moon.normal_form_transform
        assert is_symplectic(C_l5)
        assert is_symplectic(Cinv_l5)
    
    def test_normal_form_inverse(self, l1_earth_moon):
        """Test that normal form transform and its inverse are truly inverses."""
        C, Cinv = l1_earth_moon.normal_form_transform
        
        # C * Cinv should be identity
        product = C @ Cinv
        identity = np.eye(6)
        
        assert np.allclose(product, identity, atol=1e-10)
        
        # Cinv * C should also be identity
        product_inv = Cinv @ C
        assert np.allclose(product_inv, identity, atol=1e-10)


class TestTriangularPoints:
    """Test triangular libration points (L4, L5)."""
    
    def test_triangular_positions_symmetry(self, l4_earth_moon, l5_earth_moon):
        """Test that L4 and L5 are symmetric about the x-axis."""
        pos_l4 = l4_earth_moon.position
        pos_l5 = l5_earth_moon.position
        
        # x and z coordinates should be the same
        assert np.isclose(pos_l4[0], pos_l5[0])
        assert np.isclose(pos_l4[2], pos_l5[2])
        
        # y coordinates should be opposite
        assert np.isclose(pos_l4[1], -pos_l5[1])
    
    def test_triangular_equilateral_triangle(self, system_earth_moon, l4_earth_moon):
        """Test that L4 forms an equilateral triangle with the primaries."""
        mu = system_earth_moon.mu
        pos_l4 = l4_earth_moon.position
        
        # Position of primary
        pos_primary = np.array([-mu, 0, 0])
        # Position of secondary
        pos_secondary = np.array([1 - mu, 0, 0])
        
        # Distance from L4 to primary
        dist_to_primary = np.linalg.norm(pos_l4 - pos_primary)
        # Distance from L4 to secondary
        dist_to_secondary = np.linalg.norm(pos_l4 - pos_secondary)
        # Distance between primaries
        dist_primaries = np.linalg.norm(pos_secondary - pos_primary)
        
        # All distances should be equal (equilateral triangle)
        assert np.isclose(dist_to_primary, 1.0, atol=1e-10)
        assert np.isclose(dist_to_secondary, 1.0, atol=1e-10)
        assert np.isclose(dist_primaries, 1.0, atol=1e-10)


class TestLibrationPointStringRepresentations:
    """Test LibrationPoint string representations."""
    
    def test_str_representation(self, l1_earth_moon, l4_earth_moon):
        """Test __str__ representation."""
        str_l1 = str(l1_earth_moon)
        str_l4 = str(l4_earth_moon)
        
        assert "L1Point" in str_l1
        assert "L4Point" in str_l4
        assert "mu=" in str_l1
        assert "mu=" in str_l4
    
    def test_repr_representation(self, l1_earth_moon, l2_sun_earth):
        """Test __repr__ representation."""
        repr_l1 = repr(l1_earth_moon)
        repr_l2 = repr(l2_sun_earth)
        
        assert "L1Point" in repr_l1
        assert "L2Point" in repr_l2
        assert "mu=" in repr_l1
        assert "mu=" in repr_l2


class TestLibrationPointSerialization:
    """Test LibrationPoint serialization methods."""
    
    def test_libration_point_serialization_l1(self, l1_earth_moon):
        """Test L1 point serialization and deserialization."""
        temp_path = Path("temp_l1.pkl")
        
        try:
            l1_earth_moon.save(str(temp_path))
            l1_loaded = l1_earth_moon.__class__.load(str(temp_path))
            
            # Verify properties are preserved
            assert np.allclose(l1_loaded.position, l1_earth_moon.position)
            assert l1_loaded.mu == l1_earth_moon.mu
            
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def test_libration_point_serialization_l4(self, l4_earth_moon):
        """Test L4 point serialization and deserialization."""
        temp_path = Path("temp_l4.pkl")
        
        try:
            l4_earth_moon.save(str(temp_path))
            l4_loaded = l4_earth_moon.__class__.load(str(temp_path))
            
            # Verify properties are preserved
            assert np.allclose(l4_loaded.position, l4_earth_moon.position)
            assert l4_loaded.mu == l4_earth_moon.mu
            
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestLibrationPointCaching:
    """Test that libration points are properly cached by the system."""
    
    def test_libration_point_caching(self, system_earth_moon):
        """Test that requesting the same libration point returns the same instance."""
        l1_first = system_earth_moon.get_libration_point(1)
        l1_second = system_earth_moon.get_libration_point(1)
        
        # Should be the same instance (cached)
        assert l1_first is l1_second
    
    def test_different_libration_points_different_instances(self, system_earth_moon):
        """Test that different libration points are different instances."""
        l1 = system_earth_moon.get_libration_point(1)
        l2 = system_earth_moon.get_libration_point(2)
        
        assert l1 is not l2
        assert type(l1) != type(l2)


class TestLibrationPointIntegration:
    """Integration tests for LibrationPoint class."""
    
    def test_all_five_libration_points(self, system_earth_moon):
        """Test that all five libration points can be created and accessed."""
        libration_points = []
        
        for i in range(1, 6):
            point = system_earth_moon.get_libration_point(i)
            libration_points.append(point)
        
        assert len(libration_points) == 5
        
        # All should have valid positions
        for point in libration_points:
            pos = point.position
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (3,)
            assert not np.any(np.isnan(pos))
    
    def test_libration_points_with_different_systems(self):
        """Test creating libration points for different systems."""
        # Earth-Moon system
        em_system = System.from_bodies("earth", "moon")
        l1_em = em_system.get_libration_point(1)
        
        # Sun-Earth system
        se_system = System.from_bodies("sun", "earth")
        l1_se = se_system.get_libration_point(1)
        
        # They should have different properties
        assert l1_em.mu != l1_se.mu
        assert not np.allclose(l1_em.position, l1_se.position)
    
    def test_libration_point_properties_consistency(self, l1_earth_moon):
        """Test that libration point properties are consistent across accesses."""
        # Multiple accesses should return the same values
        pos1 = l1_earth_moon.position
        pos2 = l1_earth_moon.position
        
        assert np.allclose(pos1, pos2)
        
        energy1 = l1_earth_moon.energy
        energy2 = l1_earth_moon.energy
        
        assert energy1 == energy2


class TestLibrationPointEdgeCases:
    """Test LibrationPoint edge cases and error handling."""
    
    def test_invalid_libration_point_index(self, system_earth_moon):
        """Test that invalid libration point indices raise errors."""
        with pytest.raises(ValueError):
            system_earth_moon.get_libration_point(0)
        
        with pytest.raises(ValueError):
            system_earth_moon.get_libration_point(6)
        
        with pytest.raises(ValueError):
            system_earth_moon.get_libration_point(-1)
    
    def test_libration_point_with_extreme_mu(self):
        """Test libration points with extreme mass ratios."""
        # Very small mu (Sun-like system)
        system_small_mu = System.from_mu(1e-6)
        l1_small = system_small_mu.get_libration_point(1)
        
        assert l1_small.mu == 1e-6
        assert l1_small.position is not None
        
        # Larger mu (but still < 0.5)
        system_large_mu = System.from_mu(0.3)
        l1_large = system_large_mu.get_libration_point(1)
        
        assert l1_large.mu == 0.3
        assert l1_large.position is not None