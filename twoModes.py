
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.random import rand
from scipy.integrate import odeint
from scipy.optimize import fsolve

# Coefficients
G_mu1 = -2.8
G_c1 = -7.75
G_a2 = -2.66

# Chosen template
group_generator = np.array([[0, -1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, -2],
                            [0, 0, 2, 0]])
template_point = np.array([1, 0, 0, 0])
group_tangent = np.dot(group_generator, template_point)


def vel_ss_full(state_full, t):
    """
    Velocity in the full state space.
    
    sv: state vector [x1, y1, x2, y2]
    t: just for convention of odeint, not used.
    return: velocity at state_full. Dimension [1 x 4]
    """
    x1, y1, x2, y2 = state_full
    r2 = x1**2 + y1**2
    
    result = np.array([(G_mu1 - r2) * x1 + G_c1 * (x1 * x2 + y1 * y2),
                       (G_mu1 - r2) * y1 + G_c1 * (x1 * y2 - x2 * y1),
                       x2 + y2 + x1 ** 2 - y1 ** 2 + G_a2 * x2 * r2,
                       -x2 + y2 + 2 * x1 * y1 + G_a2 * y2 * r2])
    return result


def vel_ss_reduced(state_reduced, t):
    """
    Velocity in the slice after reducing the continuous symmetry

    state_reduced: state vector in slice [\hat{x}_1, \hat{x}_2, \hat{y}_2]
    t: not used
    return: velocity at state_reduced. dimension [1 x 3]
    """
    state_red = (state_reduced[0], 0, state_reduced[1],
                 state_reduced[2])
    velocity_at_state = vel_ss_full(state_red, 1)
    tangent_at_state = np.dot(group_generator, state_red)

    velocity = velocity_at_state - np.inner(vel_phase(state_reduced),
                                            tangent_at_state)
    result = np.array([velocity[0], velocity[2], velocity[3]]) 
    return result


def vel_phase(state_reduced):
    """
    phase velocity. 

    state_reduced: state vector in slice [\hat{x}_1, \hat{x}_2, \hat{y}_2]
    Note: phase velocity only depends on the state vector
    """
    state_red_full = (state_reduced[0], 0, state_reduced[1], state_reduced[2])
    velocity_at_state = vel_ss_full(state_red_full, 1)
    tangent_at_state = np.dot(group_generator, state_red_full)

    result = np.dot(velocity_at_state.T, group_tangent) /\
             np.inner(tangent_at_state.T, group_tangent)
    return result


def integrator(init_state, dt, nstp):
    """
    integrate two modes system in the full state space.

    init_state: initial state [x1, y1, x2, y2]
    dt: time step 
    nstp: number of time step
    """
    states = odeint(vel_ss_full, init_state, np.arange(0, dt*nstp, dt))
    return states


def integrator_reduced(init_state, dt, nstp):    
    """
    integrate two modes system in the slice

    init_state: initial state [\hat{x}_1, \hat{x}_2, \hat{y}_2]
    dt: time step 
    nstp: number of time step
    """
    states = odeint(vel_ss_reduced, init_state, np.arange(0, dt*nstp, dt))
    return states


def stability_matrix(state_full):
    """
    Calculate the stability matrix in the full state space

    state: state vector in slice [{x}_1, {x}_2, {y}_1, {y}_2]
    return: stability matrix. Dimension [3 x 3]
    """
    x1, y1, x2, y2 = state_full

    result = np.array(
        [[G_mu1-3*x1**2+G_c1*x2-y1**2, G_c1*y2-2*x1*y1, G_c1*x1, G_c1*y1],
         [G_c1*y2-2*x1*y1, G_mu1-x1**2-G_c1*x1-3*y1**2, -G_c1*y1, G_c1*x1],
         [2*x1+2*G_a2*x1*x2, 2*G_a2*x2*y1-2*y1, 1+G_a2*(x1**2+y1**2), 1],
         [2*y1+2*G_a2*x1*y2, 2*x1+2*G_a2*y1*y2, -1, 1+G_a2*(x1**2+y1**2)]])
    return result


def stability_matrix_reduced(state_reduced):
    """
    calculate the stability matrix on the slice

    stateVec_reduced: state vector in slice [\hat{x}_1, \hat{x}_2, \hat{y}_2]
    return: stability matrix. Dimension [3 x 3]
    """
    x1 = state_reduced[0]
    y1 = 0
    x2 = state_reduced[1]
    y2 = state_reduced[2]
    state_ext = (x1, y1, x2, y2)

    stab_full = stability_matrix(state_ext)
    tangent_at_state = np.dot(group_generator, state_ext)

    sqr_braket = np.dot(np.inner(tangent_at_state, group_tangent),
                        stab_full.T) - \
                 np.dot(np.inner(vel_ss_full(state_ext, 0), group_tangent),
                        group_generator.T)

    stab = np.zeros_like(stab_full)
    for i in range(4):
        for j in range(4):
            stab[i, j] = stab_full[i, j] - \
                         np.dot(tangent_at_state[i],
                                np.dot(sqr_braket, group_tangent)[j]) / \
                         np.inner(tangent_at_state, group_tangent) ** 2 - \
                         np.inner(vel_ss_full(state_ext, 0), group_tangent) / \
                         np.inner(tangent_at_state, group_tangent) * \
                         group_generator[i, j]
    stab = np.delete(stab, 1, axis=0)
    stab = np.delete(stab, 1, axis=1)
    return stab


def group_transform(state_full, phi):
    """
    perform group transform on a particular state. Symmetry group is 'g(phi)'
    and state is 'x'. the transformed state is ' xp = g(phi) * x '
 
    state: state in the full state space. Dimension [1 x 4]
    phi: group angle. in range [0, 2*pi]
    return: the transformed state. Dimension [1 x 4]
    """
    g = np.array([
        [np.cos(phi), -np.sin(phi), 0, 0],
        [np.sin(phi), np.cos(phi), 0, 0],
        [0, 0, np.cos(2*phi), -np.sin(2*phi)],
        [0, 0, np.sin(2*phi), np.cos(2*phi)]
    ])
    state_transformed = np.transpose(np.dot(g, np.transpose(state_full)))
    return state_transformed


def reduce_symmetry(states):
    """
    tranform states in the full state space into the slice.
    Hint: use numpy.arctan2(y,x)
    Note: this function should be able to reduce the symmetry
    of a single state and that of a sequence of states. 

    states: states in the full state space. dimension [m x 4] 
    return: the corresponding states on the slice dimension [m x 3]
    """
    if states.ndim == 1:  # if the state is one point
        x1, y1 = states[:2]
        if x1 == 0:
            phi = np.arccos(0)
        else:
            phi = np.arctan2(-y1, x1)
        x1s, _, x2s, y2s = group_transform(states, phi)
        reduced_states = (x1s, x2s, y2s)

    elif states.ndim == 2: # if they are a sequence of state points
        reduced_states = np.zeros((states.shape[0], states.shape[1]-1))
        for i in range(states.shape[0]):

            x1, y1 = states[i, :2]
            if x1 == 0:
                phi = np.arccos(0)
            else:
                phi = np.arctan2(-y1, x1)
            x1s, _, x2s, y2s = group_transform(states[i, :], phi)
            reduced_states[i, 0] = x1s
            reduced_states[i, 1] = x2s
            reduced_states[i, 2] = y2s

    else:
        reduced_states = (None, None, None)

    return reduced_states


def plot_figure(orbit):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2])
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")


if __name__ == '__main__':
    case = 3

    if case == 1:
        """
        validate your implementation.
        We generate an ergodic trajectory, and then use two
        different methods to obtain the corresponding trajectory in slice.
        The first method is post-processing. The second method utilizes
        the dynamics in the slice directly.
        """
        x0 = 0.1 * rand(4)  # random initial state
        x0_reduced = reduce_symmetry(x0)  # initial state transformed into slice
        dt = 0.005
        nstp = 500.0 / dt
        # trajectory in the full state space
        orbit = integrator(x0, dt, nstp)
        # trajectory in the slice by reducing the symmetry
        reduced_orbit = reduce_symmetry(orbit)
        # trajectory in the slice by integration in slice
        reduced_orbit2 = integrator_reduced(x0_reduced, dt, nstp)
        
        plot_figure(orbit[:, 0:3])
        plot_figure(reduced_orbit[:, 0:3])
        plot_figure(reduced_orbit2[:, 0:3])
        plt.show()

        print(stability_matrix_reduced(np.array([0.1, 0.2, 0.3])))

    elif case == 2:
        """
        Try reasonable guess to find relative equilibrium.
        One possible way: numpy.fsolve
        """
        guess = np.array([0.5, -0.5, 0.1])  # a relative good guess

        req = fsolve(vel_ss_reduced, guess, args=0)
        print('Relative equilibrium: ', req)
        print('Phase velocity at equilibrium: ', vel_phase(req))

        # see how relative equilibrium drifts in the full state space
        req_full = np.array([req[0], 0, req[1], req[2]])
        dt = 0.005
        T = np.abs(2 * np.pi / vel_phase(req))
        nstp = np.round(T / dt)
        orbit = integrator(req_full, dt, nstp)
        plot_figure(orbit[:, 0:3])
        plt.show()

    elif case == 3:
        """
        return map in the Poincare section. This case is similar to hw3.

        We start with the relative equilibrium, and construct a Poincare
        section by its real and imaginary part of its expanding
        eigen-vector (real part and z-axis is in the Poincare section).

        Then we record the Poincare intersection points by an ergodic
        trajectory. Sort them by their distance to the relative equilibrium,
        and calculate the arc length r_n from the relative equilibrium to
        each of them. After that we construct map r_n -> r_{n+1},
        r_n -> r_{n+2}, r_n -> r_{n+3}, etc. The fixed points of these map
        give us good initial guesses for periodic orbits. If you like, you
        can use Newton method to refine these initial conditions. For HW5,
        you are only required to obtain the return map.
        """
        # Relative equilibrium from case 2 [rx1, rx2, ry2]
        req = np.array([0.43996558, -0.38626706, 0.0702044])

        # Find the real and imaginary parts of the expanding eigen-vector at req
        stability_req = stability_matrix_reduced(req)
        w, v = np.linalg.eig(stability_req)
        Vr = np.array([e.real for e in v[:, 1]])
        Vi = np.array([e.imag for e in v[:, 1]])
        assert(np.allclose(Vi, np.array([-0., 0.58062392, -0.00172256])))

        # For simplicity, let's work in new coordinates, with origin at req.
        # Construct an orthogonal basis from Vr, Vi and z-axis (\hat{y}_2 axis).
        q, r = np.linalg.qr(np.column_stack((Vr, Vi, np.array([0, 0, 1]))))

        Px = q[:, 0]  # in the direction of Vr
        Py = q[:, 1]  # in the plan spanned by (Vr, Vi), and orthogonal to Px
        Pz = q[:, 2]  # orthogonal to Px and Py
        assert(np.allclose(Py, np.array([-0.12715969, -0.9918583, 0.00689345])))

        # Produce an ergodic trajectory started from relative equilibrium
        # x0_reduced = req + 0.0001*Vr
        x0_reduced = req + 0.0001*Px
        dt = 0.005
        nstp = int(800.0 / dt)
        time_array = np.linspace(0, dt * nstp, nstp)

        orbit = integrator_reduced(x0_reduced, dt, nstp)

        # Project orbit to the new basis [Px, Py, Pz], with req being origin
        def projection(vec, basis, origin):
            offsets = np.tile(origin, (vec.shape[0], 1))
            return np.dot(vec - offsets, basis)

        basis_eigen = np.array([Px, Py, Pz]).T
        orbit_proj = projection(orbit, basis_eigen, req)

        assert(np.allclose(projection(req, basis_eigen, req), [0, 0, 0]))
        assert(np.allclose(projection(Px + req, basis_eigen, req), [1, 0, 0]))

        # Choose Poincare section be Px = 0 (y-z plane), find all the
        # intersection points by orbit_prj.
        # Note: choose the right direction of this Poincare section, otherwise,
        # you will get two branches of intersection points.
        # Hint: you can find adjacent points who are at the opposite region of
        # this poincare section and then use simple linear interpolation to
        # get the intersection point.

        PoincarePoints = np.array([], dtype=np.float)
        for i in range(len(orbit_proj) - 1):
            if orbit_proj[i][0] > 0 >= orbit_proj[i+1][0]:
                pt_interp = (orbit_proj[i] - orbit_proj[i+1]) / 2 + \
                            orbit_proj[i+1]
                PoincarePoints = np.append(PoincarePoints, [pt_interp])

        PoincarePoints = np.reshape(PoincarePoints, (-1, 3))
        # basis_poincare = np.array([np.zeros(Px.shape), Py, Pz]).T
        # PoincarePoints = projection(PoincarePoints, basis_poincare, [0, 0, 0])

        # The Euclidean distance of intersection points to the origin.
        distance = [np.linalg.norm(e) for e in PoincarePoints]

        # Now reorder the distance from small to large. Also keep note which
        # distance correspond to which intersection point.
        order = np.argsort(distance)

        # Suppose the Euclidean distance is [d1, d2,..., dm]
        # (sorted from small to large), the corresponding intersection points
        # are [p_{k_1}, p_{k_2}, ..., p_{k_m}], then the arch length of
        # p_{k_i} from relative equilibrium is
        # r_{k_i} = \sum_{j = 1}^{j = i} \sqrt( (p_{k_j} - p_{k_{j-1}})^2 )
        # here p_{k_0} refers to the relative eq. itself, which is the origin.
        # Example: r_{k_2} = |p_{k_2} - p_{k_1}| + |p_{k_1} - p_{k_0}|
        # In this way, we have the arch length of each Poincare intersection
        # point. The return map r_n -> r_{n+1} indicates how intersection
        # points stretch and fold on the Poincare section.

        PoincarePoints = PoincarePoints[order]

        PoincarePoints = np.vstack((np.zeros((1, 3)), PoincarePoints))
        length = np.zeros((PoincarePoints.shape[0], 1))

        for n in range(1, len(length)):
            dist = np.linalg.norm(np.array(PoincarePoints[n]) -
                                  np.array(PoincarePoints[n-1]))
            length[n] = length[n-1] + dist
        length = length[1:]

        rev_order = np.argsort(order)
        length = length[rev_order]

        # Plot the return map with different order. Try to locate the fixed
        # points in each of these return map. Each of them corresponds to
        # the initial condition of a periodic orbit. Use the same skill in
        # HW3 to get the initial conditions for these fixed points, and have a
        # look at the structure of the corresponding periodic orbits.

        plt.figure()
        plt.plot(length, length, linestyle='-', color='red')
        plt.plot(length[:-1], length[1:], linestyle='none', marker='.')

        plt.figure()
        plt.plot(length, length, linestyle='-', color='red')
        plt.plot(length[:-2], length[2:], linestyle='none', marker='.')

        plt.figure()
        plt.plot(length, length, linestyle='-', color='red')
        plt.plot(length[:-4], length[4:], linestyle='none', marker='.')
        plt.show()
