"""
Implements a family of nonseparable Hamiltonian symplectic integrators, where the family is parameterized by the
coefficients which define the weights for each update step.

References

    https://en.wikipedia.org/wiki/Symplectic_integrator
    https://en.wikipedia.org/wiki/Energy_drift
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.043303 - Molei Tao - Explicit symplectic approximation of nonseparable Hamiltonians: Algorithm and long time performance
"""

import numpy as np
from .. import apply_along_axes
from . import exceptions

def heuristic_estimate_for_omega (*, delta, order, c=10.0):
    """
    Uses a heuristic to give an estimate for the binding coefficient omega in the nonseparable
    Hamiltonian integrator.  The rule from the Tao paper is that the timestep `delta` must be much
    smaller than `omega**(-1/order)`.  If we define `x is much smaller than y` to be `c*x <= y` for
    a large positive constant c, then the condition is equivalent to

        omega <= (c*delta)**(-order),

    and we will use that bound as the heuristic assignment.
    """
    return (c*delta)**(-order)

def integrate (*, initial_coordinates, t_v, dH_dq, dH_dp, order, omega):
    """
    This function computes multiple timesteps of the nonseparable Hamiltonian symplectic integrator.

    Let N denote the dimension of the configuration space (i.e. the number of components of the q coordinate).

    A single set of coordinates shall be represented with a numpy array of shape (2,N).

    Parameters:

    -   initial_coordinates specify the coordinates from which to begin integrating.  This should have
        the shape (A_1,A_2,...,A_M,2,N), where M might be zero (in which case the shape is (2,N)).
        The indices A_1,A_2,...,A_M (of which there can be none) may index some other parameter to
        the initial conditions, such that many integral curves will be computed in parallel (one for
        each assignment of A_1,A_2,...,A_M index).

    -   t_v specifies a list of the time values at which to integrate the system.  The first value corresponds
        to the initial condition, so the length of t_v must be at least 1.  The timesteps are computed as the
        difference between successive elements.  The timesteps can be negative; see
        https://en.wikipedia.org/wiki/Symplectic_integrator#A_second-order_example

    -   dH_dq and dH_dp should be functions of the respective forms

        lambda q,p : <expression evaluating \partial H / \partial q>
        lambad q,p : <expression evaluating \partial H / \partial p>

        and should each accept a pair of (N,)-shaped numpy.ndarray and return an (N,)-shaped numpy.ndarray.

    -   order specifies the order of the integrator; must be a positive, even integer.

    -   omega is the coupling constant between the two copies of phase space and must be a positive value.
        See heuristic_estimate_for_omega for computing omega using a heuristic suggested by the Tao paper.

    Return values:

    -   integrated_coordinates is a numpy.ndarray having shape (len(t_v),A_1,A_2,...,A_M,2,N), containing the coordinates of
        each integrator step starting with initial_coordinates.
    """

    initial_coordinates_shape = np.shape(initial_coordinates)

    assert len(initial_coordinates_shape) >= 2
    assert initial_coordinates_shape[-2] == 2
    assert len(t_v) >= 1
    assert order > 0, 'order (which is {0}) must be a positive, even integer'.format(order)
    assert order%2 == 0, 'order (which is {0}) must be a positive, even integer'.format(order)

    # N is the dimension of the underlying configuration space.  Thus 2*N is the dimension of the phase space,
    # hence a coordinate of the phase space having shape (2,N).
    N = initial_coordinates_shape[-1]
    # get the axes not corresponding to the final (2,N) part of the shape.  This can be the empty tuple.
    non_coordinate_shape = initial_coordinates_shape[:-2]
    non_coordinate_axis_v = tuple(range(len(non_coordinate_shape)))
    # T is the number of timesteps
    T = len(t_v)

    # Create the return value
    integrated_coordinates = np.ndarray((T,)+non_coordinate_shape+(2,N), dtype=initial_coordinates.dtype)
    integrated_coordinates.fill(np.nan)
    # Create a buffer for intermediate coordinates; this will represent an element of "extended phase space"
    # (see Tao's paper).  The added `2` axis after (2,N) is to index the coordinate pairs (q,p) (index 0)
    # and (x,y) (index 1).
    current_coordinates = np.ndarray(non_coordinate_shape+(2,N,2), dtype=initial_coordinates.dtype)

    # Create slices to address the q, p, x, y components of current_coordinates.
    qp = current_coordinates[...,0]
    xy = current_coordinates[...,1]
    q = current_coordinates[...,0,:,0]
    p = current_coordinates[...,1,:,0]
    x = current_coordinates[...,0,:,1]
    y = current_coordinates[...,1,:,1]

    # Set the values of qp and xy to initial_coordinates.
    qp[...] = xy[...] = initial_coordinates

    def phi_H_a_update (delta):
        nonlocal q
        nonlocal p
        nonlocal x
        nonlocal y
        p -= delta*apply_along_axes(dH_dq, (-1,), (q,y), output_axis_v=(-1,), func_output_shape=(N,))
        x += delta*apply_along_axes(dH_dp, (-1,), (q,y), output_axis_v=(-1,), func_output_shape=(N,))

    def phi_H_b_update (delta):
        nonlocal q
        nonlocal p
        nonlocal x
        nonlocal y
        q += delta*apply_along_axes(dH_dp, (-1,), (x,p), output_axis_v=(-1,), func_output_shape=(N,))
        y -= delta*apply_along_axes(dH_dq, (-1,), (x,p), output_axis_v=(-1,), func_output_shape=(N,))

    def phi_omega_H_c_update (delta):
        nonlocal q
        nonlocal p
        nonlocal x
        nonlocal y
        # This could certainly be implemented better, since it's a linear operator.
        c           = np.cos(2*omega*delta)
        s           = np.sin(2*omega*delta)
        q_plus_x    = q + x
        q_minus_x   = q - x
        p_plus_y    = p + y
        p_minus_y   = p - y
        # The ellipsises are necessary in order to assign into the existing slices.
        q[...]      = 0.5*(q_plus_x + c*q_minus_x + s*p_minus_y)
        p[...]      = 0.5*(p_plus_y - s*q_minus_x + c*p_minus_y)
        x[...]      = 0.5*(q_plus_x - c*q_minus_x - s*p_minus_y)
        y[...]      = 0.5*(p_plus_y + s*q_minus_x - c*p_minus_y)

    def update (timestep, order):
        # Base case for inductively defined update step (induction parameter is order)
        if order == 2:
            phi_H_a_update(0.5*timestep)
            phi_H_b_update(0.5*timestep)
            phi_omega_H_c_update(timestep)
            phi_H_b_update(0.5*timestep)
            phi_H_a_update(0.5*timestep)
        # Recursive case for inductively defined update step (induction parameter is order)
        else:
            # gamma is gamma_{\ell} in the Tao paper.  Would it be faster to use a list of these gammas
            # instead of computing them each time?  Because of the recursive definition, they are computed
            # an exponential number of times.
            gamma = 1.0 / (2.0 - 2.0**(1.0/(order+1)))
            lower_order = order-2
            update(gamma*timestep, lower_order)
            update((1.0-2.0*gamma)*timestep, lower_order)
            update(gamma*timestep, lower_order)

    # Only store the (q,p) half of extended phase space in integrated_coordinates.
    integrated_coordinates[0,...] = qp

    for step_index,timestep in enumerate(np.diff(t_v)):
        try:
            # Perform update steps
            update(timestep, order)
            # Only store the (q,p) half of extended phase space in integrated_coordinates.
            integrated_coordinates[step_index+1,...] = qp
        except Exception as e:
            # If a non-system-exiting or user-defined exception was encountered, then salvage the part
            # of the curve that was computed without error.
            raise exceptions.SalvagedResultException(
                original_exception=e,
                salvaged_t_v=np.copy(t_v[:step_index+1]),
                salvaged_qp_v=np.copy(integrated_coordinates[:step_index+1,...])
            ) from e

    return integrated_coordinates
