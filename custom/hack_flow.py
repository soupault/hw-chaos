import numpy as np


class HackFlow(object):
    """
    Class represents following system:
    q' = p + q * (1 - q^2 - p^2)
    p' = -q + p * (1 - q^2 - p^2)
    in polar coordinates:
    r' = r * (1 - r**2)
    th' = -1
    """

    @staticmethod
    def velocity(state_vec, t):
        """
        Returns the velocity field.
        state_vec : the state vector in the full space. [rad, theta]
        t : time is used since odeint() requires it.
        """
        rad, theta, = state_vec[0:2]
        vrad = rad - rad ** 3
        vtheta = -1
        return np.array([vrad, vtheta])

    @staticmethod
    def stability_matrix(state_vec):
        """
        Return the stability matrix at a state point.
        state_vec : the state vector in the full space. [rad, theta]
        """
        rad, theta, = state_vec[0:2]
        stab = np.array([[1 - 3 * rad ** 2, 0],
                         [0, 0]])
        return stab
