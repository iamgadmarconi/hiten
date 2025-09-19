import numpy as np

from hiten.algorithms.dynamics.rhs import create_rhs_system
from hiten.algorithms.integrators import AdaptiveRK
from hiten.algorithms.integrators import RungeKutta
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
    # Constrain step size to improve event time accuracy on stiff relaxation
    dop853 = AdaptiveRK(order=8, max_step=1e-3)
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


def test_dop853_event_any_direction_crossing():
    # Any-direction detection should catch either sign change
    def rhs(t, y):
        return np.array([1.0])

    sys = create_rhs_system(rhs, dim=1, name="any_dir")
    y0 = np.array([0.25])
    t_vals = np.array([0.0, 5.0])

    def g(t, y):
        return float(y[0] - 1.0)

    ev_cfg = _EventConfig(direction=0, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    assert sol.times.size == 2
    assert abs(sol.times[-1] - 0.75) < 1e-10
    assert abs(sol.states[-1, 0] - 1.0) < 1e-10


def test_dop853_event_filtered_by_direction_no_hit():
    # Decreasing crossing should be ignored when direction=+1
    def rhs(t, y):
        return np.array([-1.0])

    sys = create_rhs_system(rhs, dim=1, name="filtered_dir")
    y0 = np.array([2.0])
    t_vals = np.array([0.0, 3.0])

    def g(t, y):
        return float(y[0] - 1.0)

    ev_cfg = _EventConfig(direction=+1, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    # No event expected; should reach tmax
    assert sol.times.size == 2
    assert abs(sol.times[-1] - t_vals[-1]) < 1e-12


def test_dop853_event_always_positive_no_hit():
    # Event function never changes sign (always > 0)
    def rhs(t, y):
        return np.array([0.0])

    sys = create_rhs_system(rhs, dim=1, name="always_pos")
    y0 = np.array([2.0])
    t_vals = np.array([0.0, 1.0])

    def g(t, y):
        return float(y[0] - 1.0)  # always positive for y0=2 and dy/dt=0

    ev_cfg = _EventConfig(direction=0, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    assert sol.times.size == 2
    assert abs(sol.times[-1] - t_vals[-1]) < 1e-12
    assert abs(sol.states[-1, 0] - 2.0) < 1e-12


def test_dop853_event_numba_compiled_function():
    # Passing a pre-compiled numba event function should also work
    try:
        import numba
    except Exception:
        # If numba unavailable in test env, skip silently
        return

    def rhs(t, y):
        return np.array([1.0])

    sys = create_rhs_system(rhs, dim=1, name="numba_evt")
    y0 = np.array([0.0])
    t_vals = np.array([0.0, 3.0])

    @numba.njit(cache=False)
    def g(t, y):
        return y[0] - 1.5

    ev_cfg = _EventConfig(direction=+1, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    assert sol.times.size == 2
    assert abs(sol.times[-1] - 1.5) < 1e-9
    assert abs(sol.states[-1, 0] - 1.5) < 1e-9


def test_dop853_event_stiff_relaxation_positive_crossing():
    # Stiff(ish) relaxation to y=1 from y0=0: y(t) = 1 - exp(-lambda t)
    lam = 100.0

    def rhs(t, y):
        return np.array([lam * (1.0 - y[0])])

    sys = create_rhs_system(rhs, dim=1, name="stiff_relax")
    y0 = np.array([0.0])
    t_vals = np.array([0.0, 1.0])

    y_target = 0.999
    def g(t, y):
        return float(y[0] - y_target)

    # Analytic hit time
    t_expected = -np.log(1.0 - y_target) / lam

    ev_cfg = _EventConfig(direction=+1, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    assert sol.times.size == 2
    t_hit = sol.times[-1]
    y_hit = sol.states[-1, 0]
    assert abs(t_hit - t_expected) < 1e-6
    assert abs(y_hit - y_target) < 1e-8


def test_dop853_event_crossing_very_early_from_near_plane():
    # Start extremely close to the plane from below; hit almost immediately
    def rhs(t, y):
        return np.array([1.0])

    sys = create_rhs_system(rhs, dim=1, name="near_plane")
    eps = 1e-9
    y0 = np.array([1.0 - eps])
    t_vals = np.array([0.0, 1.0])

    def g(t, y):
        return float(y[0] - 1.0)

    ev_cfg = _EventConfig(direction=+1, terminal=True)
    dop853 = AdaptiveRK(order=8)
    sol = dop853.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)

    assert sol.times.size == 2
    t_hit = sol.times[-1]
    y_hit = sol.states[-1, 0]
    assert abs(t_hit - eps) < 1e-9
    assert abs(y_hit - 1.0) < 1e-10


def test_endpoint_zero_is_detected_rk45_and_fixed_rk():
    # Construct a step where g hits exactly zero at the right endpoint.
    # dy/dt = 1, y0 = 0, event at y=1 → at t=1 exactly.
    def rhs(t, y):
        return np.array([1.0])

    sys = create_rhs_system(rhs, dim=1, name="endpoint_zero")
    y0 = np.array([0.0])
    t_vals = np.array([0.0, 1.0])

    def g(t, y):
        return float(y[0] - 1.0)

    ev_cfg = _EventConfig(direction=+1, terminal=True)

    # RK45 should detect the event at the right endpoint exactly at t=1
    rk45 = AdaptiveRK(order=5)
    sol45 = rk45.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)
    assert sol45.times.size == 2
    assert abs(sol45.times[-1] - 1.0) < 1e-12
    assert abs(sol45.states[-1, 0] - 1.0) < 1e-12

    # Fixed RK4 should detect it as well when stepping exactly to t=1
    rk4 = RungeKutta(order=4)
    sol4 = rk4.integrate(sys, y0, t_vals, event_fn=g, event_cfg=ev_cfg)
    assert sol4.times.size == 2
    assert abs(sol4.times[-1] - 1.0) < 1e-12
    assert abs(sol4.states[-1, 0] - 1.0) < 1e-12
