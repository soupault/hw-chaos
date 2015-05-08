import numpy as np


class LorenzFlow(object):
    k_sigma = 10.0
    k_rho = 28.0
    k_b = 8.0/3.0

    # C^{1/2} operation matrix
    C2 = np.array([[-1, 0, 0],
                   [0, -1, 0],
                   [0, 0, 1]])
    
    def velocity(self, state_vec, t):
        """
        Return the velocity field of Lorentz system.
        state_vec : the state vector in the full space. [x, y, z]
        t : time is used since odeint() requires it. 
        """
        x, y, z = state_vec[0:3]
        vx = self.k_sigma * (y - x)
        vy = self.k_rho * x - y - x * z
        vz = x * y - self.k_b * z
        return np.array([vx, vy, vz])
    
    def stability_matrix(self, state_vec):
        """
        Return the stability matrix at a state point.
        state_vec: the state vector in the full space. [x, y, z]
        """
        x, y, z = state_vec[0:3]
        stab = np.array([[-self.k_sigma, self.k_sigma, 0],
                         [self.k_rho - z, -1, -x],
                         [y, x, -self.k_b]])
        return stab
    
    @staticmethod
    def reduce_symmetry(self, state_arr):
        """
        Reduce C^{1/2} symmetry of Lorenz system by invariant polynomials.
        (x, y, z) -> (u, v, z) = (x^2 - y^2, 2xy, z)
        
        state_arr: trajectory in the full state space. dimension [nstp x 3]
        return: states in the invariant polynomial basis. dimension [nstp x 3]
        """
        state_arr_red = np.zeros_like(state_arr)
        state_arr_red[:, 0] = state_arr[:, 0] ** 2 - state_arr[:, 1] ** 2
        state_arr_red[:, 1] = 2 * state_arr[:, 0] * state_arr[:, 1]
        state_arr_red[:, 2] = state_arr[:, 2]
        return state_arr_red

    @staticmethod
    def periodic_orbit(self):
        state_vec = np.array([-0.78844208, -1.84888176, 18.75036186])
        dt = 0.0050279107820829149
        n = 156
        return state_vec, dt, n
