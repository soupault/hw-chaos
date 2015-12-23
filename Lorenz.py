############################################################
# This file contains related functions for integrating and reducing
# Lorenz system.
############################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.random import rand
from scipy.integrate import odeint

# G_ means global
G_sigma = 10.0
G_rho = 28.0
G_b = 8.0/3.0

# complete the definition of C^{1/2} operation matrix for Lorenz
# system. 
C2 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
        ])


def velocity(stateVec, t):
    """
    return the velocity field of Lorentz system.
    stateVec : the state vector in the full space. [x, y, z]
    t : time is used since odeint() requires it. 
    """
    x, y, z = stateVec[0:3]
    vx = G_sigma * (y - x)
    vy = G_rho * x - y - x * z
    vz = x * y - G_b * z
    return np.array([vx, vy, vz])


def stability_matrix(stateVec):
    """
    return the stability matrix at a state point.
    stateVec: the state vector in the full space. [x, y, z]
    """
    x, y, z = stateVec[0:3]
    stab = np.array([
        [-G_sigma, G_sigma, 0],
        [G_rho - z, -1, -x],
        [y, x, -G_b]
        ])
    return stab


def jacobian_velocity(stateVec, t):
    """
    Velocity function for the Jacobian integration

    Inputs:
    stateVec: (d+d^2)x1 dimensional state space vector including both the
              state space itself and the tangent space
    t: Time. Has no effect on the function, we have it as an input so that our
       ODE would be compatible for use with generic integrators from
       scipy.integrate

    Outputs:
    velJ = (d+d^2)x1 dimensional velocity vector
    """
    ssp = stateVec[0:3]
    J = stateVec[3:].reshape((3, 3))

    velJ = np.zeros(np.size(stateVec))
    velJ[0:3] = velocity(ssp, t)

    # Last dxd elements of the velJ are determined by the action of
    # stability matrix on the current value of the Jacobian
    velTangent = np.dot(stability_matrix(ssp), J)
    velJ[3:] = np.reshape(velTangent, 9)
    return velJ


def integrator(init_x, dt, nstp):
    """
    The integator of the Lorentz system.
    init_x: the intial condition
    dt : time step
    nstp: number of integration steps.
    
    return : a [ nstp x 3 ] vector 
    """
    state = odeint(velocity, init_x, np.arange(0, dt*nstp, dt))
    return state


def integrator_with_jacob(init_x, dt, nstp):
    """
    integrate the orbit and the Jacobian as well. The meaning 
    of input parameters are the same as 'integrator()'.
    
    return : 
            state: a [ nstp x 3 ] state vector 
            Jacob: [ 3 x 3 ] Jacobian matrix
    """

    # Please fill out the implementation of this function.
    # You can go back to the previous homework to see how to
    # integrate state and Jacobian at the same time.

    sspJacobian0 = np.zeros(3 + 3 ** 2)
    sspJacobian0[0:3] = init_x
    sspJacobian0[3:] = np.reshape(np.identity(3), 9)

    tArray = np.linspace(0, nstp * dt, nstp)
    sspJacobianSolution = odeint(jacobian_velocity, sspJacobian0, tArray)
    
    state = sspJacobianSolution[-1, :3]
    Jacob = sspJacobianSolution[-1, 3:].reshape((3, 3))
    
    return state, Jacob


def reduceSymmetry(states):
    """
    reduce C^{1/2} symmetry of Lorenz system by invariant polynomials.
    (x, y, z) -> (u, v, z) = (x^2 - y^2, 2xy, z)
    
    states: trajectory in the full state space. dimension [nstp x 3]
    return: states in the invariant polynomial basis. dimension [nstp x 3]
    """
    
    m, n = states.shape

    # please fill out the transformation from states to reducedStates.

    reducedStates = np.zeros_like(states)
    reducedStates[:, 0] = states[:, 0] ** 2 - states[:, 1] ** 2
    reducedStates[:, 1] = 2 * states[:, 0] * states[:, 1]
    reducedStates[:, 2] = states[:, 2]
    
    return reducedStates


def plotFig(orbit):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2])
    plt.show()


if __name__ == "__main__":
    
    case = 2
   
    # case 1: try a random initial condition
    if case == 1:
        x0 = rand(3)
        dt = 0.005
        nstp = 50.0/dt
        orbit = integrator(x0, dt, nstp)
        reduced_orbit = reduceSymmetry(orbit)
    
        plotFig(orbit)
        plotFig(reduced_orbit)

    # case 2: periodic orbit
    if case == 2:
        x0 = np.array([-0.78844208, -1.84888176, 18.75036186])
        dt = 0.0050279107820829149
        nstp = 156
        orbit_doulbe = integrator(x0, dt, nstp*2)
        orbit = orbit_doulbe[:nstp, :] # one prime period
        reduced_orbit = reduceSymmetry(orbit)

        plotFig(orbit_doulbe)
        plotFig(reduced_orbit)

    # case 3 : calculate Floquet multipliers and Floquet vectors associated
    # with the full periodic orbit in the full state space.
    # Please check that one of Floquet vectors is in the same/opposite
    # direction with velocity field at x0.
    if case == 3:    
        x0 = np.array([-0.78844208, -1.84888176, 18.75036186])
        dt = 0.0050279107820829149 # integration time step
        # nstp = 156 # number of integration steps => T = nstp * dt
        nstp = 2 * 156 # number of integration steps => T = nstp * dt

        state, jacobian = integrator_with_jacob(x0, dt, nstp)
        floq_mult, floq_vec = np.linalg.eig(jacobian)
        print(floq_mult)
        print(floq_vec)


    # case 4: calculate Floquet multipliers and Flqouet vectors associated
    # with the prime period. 
    if case == 4:
        x0 = np.array([-0.78844208, -1.84888176, 18.75036186])
        dt = 0.0050279107820829149
        nstp = 156

        # please fill out the part to calculate Floquet multipliers and
        # vectors.
        state, jacobian = integrator_with_jacob(x0, dt, nstp)
        print(jacobian)
        floq_mult, floq_vec = np.linalg.eig(np.dot(C2, jacobian))
        print(floq_mult)
        print(floq_vec)
