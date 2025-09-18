import numpy as np

from hiten.algorithms.dynamics.rhs import create_rhs_system
from hiten.algorithms.integrators import AdaptiveRK
from hiten.algorithms.integrators.configs import _EventConfig


def test_dop853_event_positive_crossing():
    # dy/dt = 1, y(t) = y0 + t; event at y = 1 -> t_hit = 1 - y0
    def rhs(t, y):
        return np.array([1.0])

    sys = create_rhs_system(rhs, dim=1, name="unit_slope")
    y0 = np.array([0.0])
    t_vals = np.array([0.0, 2.0])

    # g(t,y) = y - 1; positive crossing
    def g(t, y):
        return float(y[0] - 1.0)

    ev_cfg = _EventConfig(direction=+1, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    # Early termination exactly at the event
    assert sol.times.size == 2
    t_hit = sol.times[-1]
    y_hit = sol.states[-1, 0]
    assert abs(t_hit - 1.0) < 1e-10
    assert abs(y_hit - 1.0) < 1e-10


def test_dop853_event_negative_crossing():
    # dy/dt = -1, y(t) = y0 - t; event at y = 1 -> t_hit = y0 - 1
    def rhs(t, y):
        return np.array([-1.0])

    sys = create_rhs_system(rhs, dim=1, name="neg_slope")
    y0 = np.array([1.5])
    t_vals = np.array([0.0, 2.0])

    def g(t, y):
        return float(y[0] - 1.0)

    ev_cfg = _EventConfig(direction=-1, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    assert sol.times.size == 2
    t_hit = sol.times[-1]
    y_hit = sol.states[-1, 0]
    assert abs(t_hit - 0.5) < 1e-10
    assert abs(y_hit - 1.0) < 1e-10


def test_dop853_event_strict_direction_no_hit():
    # dy/dt = 1, starting below plane; request decreasing direction -> no hit
    def rhs(t, y):
        return np.array([1.0])

    sys = create_rhs_system(rhs, dim=1, name="unit_slope_nohit")
    y0 = np.array([0.0])
    t_vals = np.array([0.0, 1.5])

    def g(t, y):
        return float(y[0] - 1.0)

    ev_cfg = _EventConfig(direction=-1, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    # Should run to tmax with no event
    assert sol.times.size == 2
    assert abs(sol.times[-1] - t_vals[-1]) < 1e-15
    assert abs(sol.states[-1, 0] - (y0[0] + t_vals[-1])) < 1e-8


def test_dop853_start_on_plane_moving_away_no_hit():
    # Start exactly on plane and move away in + direction -> no strict crossing
    def rhs(t, y):
        return np.array([+1.0])

    sys = create_rhs_system(rhs, dim=1, name="start_on_plane")
    y0 = np.array([1.0])
    t_vals = np.array([0.0, 0.5])

    def g(t, y):
        return float(y[0] - 1.0)

    ev_cfg = _EventConfig(direction=+1, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    assert sol.times.size == 2
    # No event: should end at tmax
    assert abs(sol.times[-1] - t_vals[-1]) < 1e-15
    assert abs(sol.states[-1, 0] - (1.0 + t_vals[-1])) < 1e-8
