import numpy as np
from custom.integrator import integrate_with_jacobian


def floquet(flow, state_init, dt, n, periods=1):
    """
    Calculate Floquet multipliers and Floquet vectors associated
    with the full periodic orbit in the full state space.

    :param flow: instance of class, representing system
    :return:
    """
    state, jacobian = integrate_with_jacobian(flow, state_init, dt, n * periods)
    floq_mult, floq_vec = np.linalg.eig(jacobian)

    return floq_mult, floq_vec


if __name__ == "__main__":
    pass
