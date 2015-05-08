import numpy as np


class BakersMap(object):
    """
    if y <= 0.5:
        x_new, y_new = x/3, 2*y
    else:
        x_new, y_new = x/3 + 0.5, 2*y - 1
    """

    @staticmethod
    def velocity(state_vec, t):
        """
        Return the velocity field of Baker's map.
        state_vec : the state vector in the full space. [x, y]
        t : time is used since odeint() requires it.
        """
        x, y = state_vec[0:2]
        if y <= 0.5:
            vx = -2/3 * x
            vy = y
        else:
            vx = -2/3 * x + 0.5
            vy = y - 1
        return np.array([vx, vy])
