import numpy as np
import pytest

from hiten.system.base import System, systemConfig
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
    earth_mass = Constants.get_mass("earth")
    earth_radius = Constants.get_radius("earth")
    moon_mass = Constants.get_mass("moon")
    moon_radius = Constants.get_radius("moon")
    distance = Constants.get_orbital_distance("earth", "moon")

    earth = Body("Earth", earth_mass, earth_radius, color="blue")
    moon = Body("Moon", moon_mass, moon_radius, color="gray", parent=earth)

    return System(systemConfig(primary=earth, secondary=moon, distance=distance))

@pytest.fixture
def system_sun_earth():
    sun_mass = Constants.get_mass("sun")
    sun_radius = Constants.get_radius("sun")
    earth_mass = Constants.get_mass("earth")
    earth_radius = Constants.get_radius("earth")
    distance = Constants.get_orbital_distance("sun", "earth")

    sun = Body("Sun", sun_mass, sun_radius, color="yellow")
    earth = Body("Earth", earth_mass, earth_radius, color="blue", parent=sun)

    return System(systemConfig(primary=sun, secondary=earth, distance=distance))

@pytest.fixture
def system_sun_jupiter():
    sun_mass = Constants.get_mass("sun")
    sun_radius = Constants.get_radius("sun")
    jupiter_mass = Constants.get_mass("jupiter")
    jupiter_radius = Constants.get_radius("jupiter")
    distance = Constants.get_orbital_distance("sun", "jupiter")

    sun = Body("Sun", sun_mass, sun_radius, color="yellow")
    jupiter = Body("Jupiter", jupiter_mass, jupiter_radius, color="gray", parent=sun)
    return System(systemConfig(primary=sun, secondary=jupiter, distance=distance))

@pytest.fixture
def l1_earth_moon(system_earth_moon):
    return L1Point(system_earth_moon)

@pytest.fixture
def l2_earth_moon(system_earth_moon):
    return L2Point(system_earth_moon)

@pytest.fixture
def l3_earth_moon(system_earth_moon):
    return L3Point(system_earth_moon)

@pytest.fixture
def l4_earth_moon(system_earth_moon):
    return L4Point(system_earth_moon)

@pytest.fixture
def l5_earth_moon(system_earth_moon):
    return L5Point(system_earth_moon)

@pytest.fixture
def l1_sun_earth(system_sun_earth):
    return L1Point(system_sun_earth)

@pytest.fixture
def l2_sun_earth(system_sun_earth):
    return L2Point(system_sun_earth)

@pytest.fixture
def l3_sun_jupiter(system_sun_jupiter):
    return L3Point(system_sun_jupiter)


def test_libration_point_initialization():
    """Test initialization of different libration points."""
    
    def create_mock_system(mu):
        # Create a mock system with the given mu. The bodies are placeholders.
        primary = Body("p", 1 - mu, 0.1)
        secondary = Body("s", mu, 0.1, parent=primary)
        config = systemConfig(primary=primary, secondary=secondary, distance=1.0)
        return System(config)

    l1_earth_moon = L1Point(create_mock_system(TEST_MU_EARTH_MOON))
    assert l1_earth_moon.mu == TEST_MU_EARTH_MOON
    
    l2_sun_earth = L2Point(create_mock_system(TEST_MU_SUN_EARTH))
    assert l2_sun_earth.mu == TEST_MU_SUN_EARTH
    
    l3_sun_jupiter = L3Point(create_mock_system(TEST_MU_SUN_JUPITER))
    assert l3_sun_jupiter.mu == TEST_MU_SUN_JUPITER
    
    l4_earth_moon = L4Point(create_mock_system(TEST_MU_EARTH_MOON))
    assert l4_earth_moon.mu == TEST_MU_EARTH_MOON
    
    l5_sun_earth = L5Point(create_mock_system(TEST_MU_SUN_EARTH))
    assert l5_sun_earth.mu == TEST_MU_SUN_EARTH

def test_positions(l1_earth_moon, l2_earth_moon, l3_earth_moon, l4_earth_moon, l5_earth_moon):
    """Test computation of libration point positions."""
    pos_l1 = l1_earth_moon.position
    assert -TEST_MU_EARTH_MOON < pos_l1[0] < 1-TEST_MU_EARTH_MOON
    assert np.isclose(pos_l1[1], 0)
    assert np.isclose(pos_l1[2], 0)
    
    pos_l2 = l2_earth_moon.position
    assert pos_l2[0] > 1-TEST_MU_EARTH_MOON
    assert np.isclose(pos_l2[1], 0)
    assert np.isclose(pos_l2[2], 0)
    
    pos_l3 = l3_earth_moon.position
    assert pos_l3[0] < -TEST_MU_EARTH_MOON
    assert np.isclose(pos_l3[1], 0)
    assert np.isclose(pos_l3[2], 0)
    
    pos_l4 = l4_earth_moon.position
    assert np.isclose(pos_l4[0], 0.5-TEST_MU_EARTH_MOON)
    assert np.isclose(pos_l4[1], np.sqrt(3)/2)
    assert np.isclose(pos_l4[2], 0)
    
    pos_l5 = l5_earth_moon.position
    assert np.isclose(pos_l5[0], 0.5-TEST_MU_EARTH_MOON)
    assert np.isclose(pos_l5[1], -np.sqrt(3)/2)
    assert np.isclose(pos_l5[2], 0)

def test_gamma_values(l1_earth_moon, l2_earth_moon, l3_earth_moon):
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

def test_cn_coefficients(l1_earth_moon, l2_earth_moon, l3_earth_moon, l1_sun_earth, l2_sun_earth, l3_sun_jupiter):
    """Test calculation of cn coefficients for collinear points."""
    c2_l1_em = l1_earth_moon._cn(2)
    c2_l2_em = l2_earth_moon._cn(2)
    c2_l3_em = l3_earth_moon._cn(2)

    c2_l1_se = l1_sun_earth._cn(2)
    c2_l2_se = l2_sun_earth._cn(2)
    c2_l3_sj = l3_sun_jupiter._cn(2)

    assert c2_l1_em > 1.0
    assert c2_l2_em > 1.0
    assert c2_l3_em > 1.0

    assert c2_l1_se > 1.0
    assert c2_l2_se > 1.0
    assert c2_l3_sj > 1.0

def test_linear_modes(l1_earth_moon, l2_earth_moon, l3_earth_moon):
    """Test calculation of linear modes for collinear points.
    """
    lambda1, omega1, omega2 = l1_earth_moon.linear_modes

    assert lambda1 > 0
    assert omega1 > 0
    assert omega2 > 0

    c2 = l1_earth_moon._cn(2)
    
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
    
    lambda1_l2, omega1_l2, omega2_l2 = l2_earth_moon.linear_modes

    assert lambda1 > 0
    assert omega1 > 0
    assert omega2 > 0

    c2_l2 = l2_earth_moon._cn(2)
    
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
    
    lambda1_l3, omega1_l3, omega2_l3 = l3_earth_moon.linear_modes

    assert lambda1 > 0
    assert omega1 > 0
    assert omega2 > 0

    c2_l3 = l3_earth_moon._cn(2)
    
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

def test_scale_factors(l1_earth_moon, l2_earth_moon, l3_earth_moon):
    """Test that scale factors s1 and s2 are always positive."""
    lambda1, omega1, omega2 = l1_earth_moon.linear_modes
    s1, s2 = l1_earth_moon._scale_factor(lambda1, omega1)
    
    assert s1 > 0, "s1 scale factor should be positive"
    assert s2 > 0, "s2 scale factor should be positive"
    
    lambda1_l2, omega1_l2, omega2_l2 = l2_earth_moon.linear_modes
    s1_l2, s2_l2 = l2_earth_moon._scale_factor(lambda1_l2, omega1_l2)
    
    assert s1_l2 > 0, "s1 scale factor should be positive for L2"
    assert s2_l2 > 0, "s2 scale factor should be positive for L2"
    
    lambda1_l3, omega1_l3, omega2_l3 = l3_earth_moon.linear_modes
    s1_l3, s2_l3 = l3_earth_moon._scale_factor(lambda1_l3, omega1_l3)
    
    assert s1_l3 > 0, "s1 scale factor should be positive for L3"
    assert s2_l3 > 0, "s2 scale factor should be positive for L3"

def test_normal_form_transform(l1_earth_moon, l2_earth_moon, l3_earth_moon):
    """Test normal form transform for collinear points."""
    C_l1, Cinv_l1 = l1_earth_moon.normal_form_transform()
    assert is_symplectic(C_l1)
    assert is_symplectic(Cinv_l1)

    C_l2, Cinv_l2 = l2_earth_moon.normal_form_transform()
    assert is_symplectic(C_l2)
    assert is_symplectic(Cinv_l2)

    C_l3, Cinv_l3 = l3_earth_moon.normal_form_transform()
    assert is_symplectic(C_l3)
    assert is_symplectic(Cinv_l3)
