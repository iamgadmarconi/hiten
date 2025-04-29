from system.body import Body
from system.base import System, systemConfig
from orbits.base import orbitConfig
from orbits.lyapunov import LyapunovOrbit
from utils.constants import Constants
import numpy as np
import pytest

from log_config import logger

@pytest.fixture
def system():
    """Fixture that sets up the Earth-Moon system."""
    earth_mass = Constants.get_mass("earth")
    earth_radius = Constants.get_radius("earth")
    moon_mass = Constants.get_mass("moon")
    moon_radius = Constants.get_radius("moon")
    distance = Constants.get_orbital_distance("earth", "moon")

    earth = Body("Earth", earth_mass, earth_radius, color="blue")
    moon = Body("Moon", moon_mass, moon_radius, color="gray", parent=earth)

    return System(systemConfig(primary=earth, secondary=moon, distance=distance))

@pytest.fixture
def l1_orbit(system):
    """Fixture that creates a L1 Lyapunov orbit."""
    config = orbitConfig(
        system=system,
        orbit_family="lyapunov",
        libration_point_idx=1,
        extra_params={"Ax": 4e-3}
    )
    return LyapunovOrbit(config)

@pytest.fixture
def l2_orbit(system):
    """Fixture that creates a L2 Lyapunov orbit."""
    config = orbitConfig(
        system=system,
        orbit_family="lyapunov",
        libration_point_idx=2,
        extra_params={"Ax": 4e-3}
    )
    return LyapunovOrbit(config)

def test_lyapunov_orbit_ic(l1_orbit, l2_orbit):
    """Test initial condition generation for Lyapunov orbits."""
    # Check that initial conditions have the right shape
    assert l1_orbit.initial_state.shape == (6,), "Initial state should be a 6-element vector"
    assert l2_orbit.initial_state.shape == (6,), "Initial state should be a 6-element vector"
    
    # For L1 orbit, x should be less than the L1 point (between primary and L1)
    assert l1_orbit.initial_state[0] < l1_orbit.system.libration_points[0][0]
    
    # For L2 orbit, x should be greater than the L2 point (beyond L2)
    assert l2_orbit.initial_state[0] > l2_orbit.system.libration_points[1][0]
    
    # For Lyapunov orbits, y should be near zero
    assert abs(l1_orbit.initial_state[1]) < 1e-10, "Y coordinate should be approximately zero for planar Lyapunov orbit"
    assert abs(l2_orbit.initial_state[1]) < 1e-10, "Y coordinate should be approximately zero for planar Lyapunov orbit"

def test_lyapunov_differential_correction(l1_orbit):
    """Test differential correction for Lyapunov orbits."""
    # Store initial state before correction
    initial_state_before = l1_orbit.initial_state.copy()
    
    # Perform differential correction
    l1_orbit.differential_correction()
    
    # Check that the state has been updated
    assert not np.array_equal(l1_orbit.initial_state, initial_state_before), "Initial state should change after correction"
    
    # Check that period is positive
    assert l1_orbit.period > 0, "Period should be positive after correction"
    
    # Y coordinate should remain near zero
    assert abs(l1_orbit.initial_state[1]) < 1e-10, "Y coordinate should still be approximately zero after correction"

def test_lyapunov_orbit_propagation(l1_orbit):
    """Test propagation of Lyapunov orbits."""
    l1_orbit.differential_correction()
    l1_orbit.propagate()
    
    # Check that trajectory has been generated
    assert l1_orbit.trajectory is not None, "Trajectory should be generated after propagation"
    assert len(l1_orbit.trajectory) > 0, "Trajectory should not be empty"
    
    # Verify that the trajectory starts with the initial condition
    assert np.allclose(l1_orbit.trajectory[0, :6], l1_orbit.initial_state), "Trajectory should start at initial state"
    
    # Verify that the trajectory returns to near the initial point after one period
    final_state = l1_orbit.trajectory[-1, :6]
    initial_state = l1_orbit.initial_state
    # We mainly care about position similarity, not velocity
    position_close = np.allclose(final_state[:3], initial_state[:3], rtol=1e-2, atol=1e-2)
    assert position_close, "Trajectory should approximately return to initial position after one period"

def test_lyapunov_orbit_stability(l1_orbit):
    """Test stability calculation for Lyapunov orbits."""
    l1_orbit.differential_correction()
    l1_orbit.propagate()
    l1_orbit.compute_stability()
    
    # Check that stability info is computed
    assert l1_orbit.stability_info is not None, "Stability info should be computed"
    assert len(l1_orbit.stability_info) == 2, "Stability info should contain eigenvalues and eigenvectors"
    
    eigenvalues, eigenvectors = l1_orbit.stability_info
    
    # Check eigenvalue and eigenvector dimensions
    assert len(eigenvalues) == 6, "Should have 6 eigenvalues"
    assert eigenvectors.shape == (6, 6), "Should have 6 eigenvectors of dimension 6"
    
    # Check stability properties
    assert isinstance(l1_orbit.is_stable, bool), "is_stable should be a boolean"
    assert isinstance(l1_orbit.is_unstable, bool), "is_unstable should be a boolean"

def test_lyapunov_base_class(l1_orbit):
    """Test base class properties for Lyapunov orbits."""
    l1_orbit.differential_correction()
    l1_orbit.propagate()
    l1_orbit.compute_stability()
    
    # Check that Jacobi constant is computed
    assert isinstance(l1_orbit.jacobi_constant, float), "Jacobi constant should be a float"
    
    # Check that energy is computed
    assert isinstance(l1_orbit.energy, float), "Energy should be a float"
    
    # Energy should be negative for a bound orbit
    assert l1_orbit.energy < 0, "Energy should be negative for a bound orbit"
