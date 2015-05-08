import numpy as np
from scipy.integrate import odeint
from functools import partial


def integrate(flow, state_init, dt, n):
    """
    Integrates the given flow over the velocity field.

    state_init: the initial condition
    dt : time step
    n: number of integration steps.

    return : a [ n x len(state_init) ] vector
    """
    state = odeint(flow.velocity, state_init, np.arange(0, dt*n, dt))
    return state


def jacobian_velocity(state_ext, t, flow=None):
    """
    Velocity function for the Jacobian integration.

    state_vec : (d+d^2)x1 dimensional state space vector including both the
                state space itself and the tangent space
    t : [not used] For compatibility with integrators from scipy.integrate

    return : (d+d^2)x1 dimensional velocity vector
    """
    if flow is None:
        return None
    state_dim = int((-1 + np.sqrt(1 + np.size(state_ext) * 4)) / 2)
    state_vec = state_ext[0:state_dim]
    jacobian = state_ext[state_dim:].reshape((state_dim, state_dim))

    jacobian_velocity = np.zeros(np.size(state_ext))
    jacobian_velocity[0:state_dim] = flow.velocity(state_vec, t)

    # Last dxd elements of the jacobian_velocity are determined by the
    # action of stability matrix on the current value of the Jacobian
    velocity_tangent = np.dot(flow.stability_matrix(state_vec), jacobian)
    jacobian_velocity[state_dim:] = np.reshape(velocity_tangent, state_dim ** 2)
    return jacobian_velocity


def integrate_with_jacobian(flow, state_init, dt, n):
    """
    integrate the orbit and the Jacobian as well. The meaning
    of input parameters are the same as 'integrate()'.

    return :
            state: a [ n x len(state) ] state vector
            Jacob: [ len(state) x len(state) ] Jacobian matrix
    """

    state_dim = np.size(state_init)
    jacobian_init = np.zeros(state_dim + state_dim ** 2)
    jacobian_init[0:state_dim] = state_init
    jacobian_init[state_dim:] = np.reshape(np.identity(state_dim),
                                           state_dim ** 2)

    time_arr = np.linspace(0, n * dt, n)

    jacobian_velocity_tmp = partial(jacobian_velocity, flow=flow)
    jacobian_solution = odeint(jacobian_velocity_tmp, jacobian_init, time_arr)
    print('check 2')
    state = jacobian_solution[-1, :state_dim]
    jacobian = jacobian_solution[-1, state_dim:].reshape((state_dim, state_dim))

    return state, jacobian
