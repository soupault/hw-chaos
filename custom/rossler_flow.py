import numpy as np


class RosslerFlow(object):
    k_a = 0.2
    k_b = 0.2
    k_c = 5.7

    def velocity(self, state_vec, t):
        """
        Velocity function for the Rossler flow.
    
        state_vec: State space vector. dx1 NumPy array: state_vec=[x, y, z]
        t: Time. Has no effect on the function, we have it as an input so that
           ODE would be compatible for use with generic integrators from
           scipy.integrate
    
        :return: velocity at state_vec. dx1 NumPy array: vel = [dx/dt, dy/dt, dz/dt]
        """
        x, y, z = state_vec
        vx = - y - z
        vy = x + self.k_a * y
        vz = self.k_b + z * (x - self.k_c)
        vel = np.array([vx, vy, vz], float)
        return vel

    def stability_matrix(self, state_vec):
        """
        state_vec: State space vector. dx1 NumPy array: state_vec = [x, y, z]
        :return: Stability matrix evaluated at state_vec. dxd NumPy array
                 A[i, j] = del Velocity[i] / del state_vec[j]
        """
        x, y, z = state_vec
        stab = np.array([[0, -1, -1],
                         [1, self.k_a, 0],
                         [z, 0, x - self.k_c]], float)
        return stab

    @staticmethod
    def periodic_orbit(self):
        state_init = np.array([9.2690828474963798,
                               0.0,
                               2.5815927750254137], float)
        period = 5.8810885346818402
        return state_init, period
