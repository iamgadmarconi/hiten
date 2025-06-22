import numpy as np
import pytest

from hiten.system.orbits.base import orbitConfig
from hiten.system.orbits.halo import HaloOrbit
from hiten.system.base import System, systemConfig
from hiten.system.body import Body
from hiten.utils.constants import Constants


@pytest.fixture
def system():   
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
    config = orbitConfig(
        orbit_family="halo",
        libration_point=system.get_libration_point(1),
        extra_params={"Az": 0.2, "Zenith": "Southern"}
    )
    return HaloOrbit(config)

@pytest.fixture
def l2_orbit(system):
    config = orbitConfig(
        orbit_family="halo",
        libration_point=system.get_libration_point(2),
        extra_params={"Az": 0.2, "Zenith": "Southern"}
    )
    return HaloOrbit(config)

def test_halo_orbit_ic(l1_orbit, l2_orbit):
    assert l1_orbit.initial_state.shape == (6,), "Initial state should be a 6-element vector"
    assert l2_orbit.initial_state.shape == (6,), "Initial state should be a 6-element vector"
    
    l1_position = l1_orbit.libration_point.position[0]
    assert l1_orbit.initial_state[0] < l1_position, f"L1 orbit x ({l1_orbit.initial_state[0]}) should be less than L1 position ({l1_position})"
    
    l2_position = l2_orbit.libration_point.position[0]
    assert abs(l2_orbit.initial_state[0] - l2_position) < 0.1, f"L2 orbit x ({l2_orbit.initial_state[0]}) should be within 0.1 of L2 position ({l2_position})"

def test_halo_differential_correction(l1_orbit):
    initial_state_before = l1_orbit.initial_state.copy()
    
    l1_orbit.differential_correction()
    
    assert not np.array_equal(l1_orbit.initial_state, initial_state_before), "Initial state should change after correction"
    
    assert l1_orbit.period > 0, "Period should be positive after correction"

def test_halo_orbit_propagation(l1_orbit):
    l1_orbit.differential_correction()
    l1_orbit.propagate()
    
    assert l1_orbit.trajectory is not None, "Trajectory should be generated after propagation"
    assert len(l1_orbit.trajectory) > 0, "Trajectory should not be empty"
    
    assert np.allclose(l1_orbit.trajectory[0, :6], l1_orbit.initial_state), "Trajectory should start at initial state"
    
    final_state = l1_orbit.trajectory[-1, :6]
    initial_state = l1_orbit.initial_state
    position_close = np.allclose(final_state[:3], initial_state[:3], rtol=1e-2, atol=1e-2)
    assert position_close, "Trajectory should approximately return to initial position after one period"

def test_halo_orbit_stability(l1_orbit):
    l1_orbit.differential_correction()
    l1_orbit.propagate()
    l1_orbit.compute_stability()
    
    assert l1_orbit.stability_info is not None, "Stability info should be computed"
    stability_indices, stability_eigvals = l1_orbit.stability_info
    
    assert len(stability_indices) >= 1, "Should have at least one stability index"
    
    assert isinstance(stability_indices[0], complex), "Stability indices should be complex numbers"
    
    assert isinstance(l1_orbit.is_stable, (bool, np.bool_)), "is_stable should be convertible to a boolean"

def test_halo_base_class(l1_orbit):
    l1_orbit.differential_correction()
    l1_orbit.propagate()
    l1_orbit.compute_stability()
    
    assert isinstance(l1_orbit.jacobi_constant, float), "Jacobi constant should be a float"
    
    assert isinstance(l1_orbit.energy, float), "Energy should be a float"
    
    assert l1_orbit.energy < 0, "Energy should be negative for a bound orbit"
