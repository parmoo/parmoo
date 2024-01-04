
""" Implementations of the SurrogateOptimizer class.

This module contains implementations of the SurrogateOptimizer ABC, which
are based on the GPS polling strategy for direct search.

Note that these strategies are all gradient-free, and therefore does not
require objective, constraint, or surrogate gradients methods to be defined.

The classes include:
 * ``LocalGPS`` -- Generalized Pattern Search (GPS) algorithm
 * ``GlobalGPS`` -- global random search, followed by GPS

"""

import numpy as np
from scipy.stats.qmc import LatinHypercube
from parmoo.structs import SurrogateOptimizer, AcquisitionFunction
from parmoo.util import xerror


class LocalGPS(SurrogateOptimizer):
    """ Use Generalized Pattern Search (GPS) to identify local solutions.

    Applies GPS to the surrogate problem, in order to identify design
    points that are locally Pareto optimal, with respect to the surrogate
    problem. Sorts poll directions by most recently used and attempts to
    step in promising directions in late iterations.

    """

    # Slots for the LocalGPS class
    __slots__ = ['n', 'lb', 'ub', 'acquisitions', 'budget', 'constraints',
                 'objectives', 'simulations', 'gradients', 'resetObjectives',
                 'penalty_func', 'sim_sd', 'restarts', 'momentum', 'q_ind']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the LocalGPS class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for the
                optimization procedure. It may contain the following:
                 * opt_budget (int): The GPS iteration limit (default: 1000).
                 * opt_restarts (int): Number of multisolve restarts per
                   scalarization (default: n+1).

        Returns:
            SurrogateOptimizer: A new SurrogateOptimizer object.

        """

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
        self.n = lb.size
        self.lb = lb
        self.ub = ub
        self.q_ind = 0
        # Check that the contents of hyperparams are legal
        if 'opt_restarts' in hyperparams:
            if isinstance(hyperparams['opt_restarts'], int):
                if hyperparams['opt_restarts'] < 1:
                    raise ValueError("hyperparams['opt_restarts'] "
                                     "must be positive")
                else:
                    self.restarts = hyperparams['opt_restarts']
            else:
                raise TypeError("hyperparams['opt_restarts'] "
                                 "must be an integer")
        else:
            self.restarts = self.n + 1
        # Check that the contents of hyperparams are legal
        if 'opt_budget' in hyperparams:
            if isinstance(hyperparams['opt_budget'], int):
                if hyperparams['opt_budget'] < 1:
                    raise ValueError("hyperparams['opt_budget'] "
                                     "must be positive")
                else:
                    self.budget = hyperparams['opt_budget']
            else:
                raise TypeError("hyperparams['opt_budget'] "
                                 "must be an integer")
        else:
            self.budget = 1000
        # Check that the contents of hyperparams are legal
        if 'opt_momentum' in hyperparams:
            if isinstance(hyperparams['opt_momentum'], float):
                if 0 <= hyperparams['opt_momentum'] < 1:
                    self.momentum = hyperparams['opt_momentum']
                else:
                    raise ValueError("hyperparams['opt_momentum'] "
                                     "must be in [0, 1)")
            else:
                raise TypeError("hyperparams['opt_momentum'] "
                                 "must be a float")
        else:
            self.momentum = 9e-1
        self.acquisitions = []
        return

    def __obj_func__(self, x_in):
        """ A wrapper for the objective function and acquisition.

        Args:
            x_in (np.ndarray): A 1d array with the design point to evaluate.

        Returns:
            float: The result of acquisition.scalarize(f(x_in, sim(x_in))).

        """

        sx_in = np.asarray(self.simulations(x_in))
        if self.acquisitions[self.q_ind].useSD():
            sdx_in = np.asarray(self.sim_sd(x_in))
        else:
            sdx_in = np.zeros(sx_in.size)
        fx_in = np.asarray(self.penalty_func(x_in, sx_in)).flatten()
        ax = self.acquisitions[self.q_ind].scalarize(fx_in, x_in,
                                                     sx_in, sdx_in)
        return ax

    def solve(self, x):
        """ Solve the surrogate problem using a generalized pattern search.

        Args:
            x (np.ndarray): A 2d array containing a list of feasible
                design points used to warm start the search.

        Returns:
            np.ndarray: A 2d numpy.ndarray of potentially efficient design
            points that were found by the GPS optimizer.

        """

        # Check that x is legal
        if isinstance(x, np.ndarray):
            if self.n != x.shape[1]:
                raise ValueError("The columns of x must match n")
            elif len(self.acquisitions) != x.shape[0]:
                raise ValueError("The rows of x must match the number " +
                                 "of acquisition functions")
        else:
            raise TypeError("x must be a numpy array")
        # Initialize an empty list of results
        result = []
        lb_tmp = np.zeros(self.n)
        ub_tmp = np.ones(self.n)
        # For each acqusisition function
        for j, acquisition in enumerate(self.acquisitions):
            # Create a new trust region
            rad = self.resetObjectives(x[j, :])
            lb_tmp[:] = np.maximum(self.lb[:], x[j, :] - rad)
            ub_tmp[:] = np.minimum(self.ub[:], x[j, :] + rad)
            # Get a candidate
            self.q_ind = j
            mesh_tol = max(1.0e-8, np.min((ub_tmp - lb_tmp) * 1.0e-4))
            xj, fj = __accelerated_pattern_search__(self.n, lb_tmp,
                                                    ub_tmp, x[j],
                                                    self.__obj_func__,
                                                    ibudget=self.budget,
                                                    mesh_tol=mesh_tol,
                                                    momentum=self.momentum,
                                                    istarts=self.restarts)
            result.append(xj)
        return np.asarray(result)


class GlobalGPS(SurrogateOptimizer):
    """ Use randomized search globally followed by GPS locally.

    Use ``RandomSearch`` to globally search the design space (search phase)
    followed by ``LocalGPS`` to refine the potentially efficient solutions
    (poll phase).

    """

    # Slots for the GlobalGPS class
    __slots__ = ['n', 'o', 'lb', 'ub', 'acquisitions', 'constraints',
                 'objectives', 'simulations', 'gradients', 'resetObjectives',
                 'penalty_func', 'opt_budget', 'gps_budget', 'sim_sd',
                 'momentum']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the GlobalGPS class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for the
                optimization procedure. It may contain the following:
                 * opt_budget (int): The function evaluation budget
                   (default: 1500)
                 * gps_budget (int): The number of the total opt_budget
                   evaluations that will be used by GPS (default: 2/3
                   of opt_budget).

        Returns:
            SurrogateOptimizer: A new SurrogateOptimizer object.

        """

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
        self.n = lb.size
        self.o = o
        self.lb = lb
        self.ub = ub
        # Check that the contents of hyperparams are legal
        if 'opt_budget' in hyperparams:
            if isinstance(hyperparams['opt_budget'], int):
                if hyperparams['opt_budget'] < 1:
                    raise ValueError("hyperparams['opt_budget'] "
                                     "must be positive")
                else:
                    self.opt_budget = hyperparams['opt_budget']
            else:
                raise TypeError("hyperparams['opt_budget'] "
                                 "must be an integer")
        else:
            self.opt_budget = 1500
        # Check that the contents of hyperparams are legal
        if 'gps_budget' in hyperparams:
            if isinstance(hyperparams['gps_budget'], int):
                if hyperparams['gps_budget'] < 1 or \
                   hyperparams['gps_budget'] >= self.opt_budget:
                    raise ValueError("hyperparams['gps_budget'] "
                                     "must be between 1 and "
                                     "hyperparams['opt_budget']")
                else:
                    self.gps_budget = hyperparams['gps_budget']
            else:
                raise TypeError("hyperparams['opt_budget'] "
                                 "must be an integer")
        else:
            self.gps_budget = int(2 * self.opt_budget / 3)
        # Check that the contents of hyperparams are legal
        if 'opt_momentum' in hyperparams:
            if isinstance(hyperparams['opt_momentum'], float):
                if 0 <= hyperparams['opt_momentum'] < 1:
                    self.momentum = hyperparams['opt_momentum']
                else:
                    raise ValueError("hyperparams['opt_momentum'] "
                                     "must be in [0, 1)")
            else:
                raise TypeError("hyperparams['opt_momentum'] "
                                 "must be a float")
        else:
            self.momentum = 9e-1
        self.acquisitions = []
        return

    def __obj_func__(self, x_in):
        """ A wrapper for the objective function and acquisition.

        Args:
            x_in (np.ndarray): A 1d array with the design point to evaluate.

        Returns:
            float: The result of acquisition.scalarize(f(x_in, sim(x_in))).

        """

        sx_in = np.asarray(self.simulations(x_in))
        if self.acquisitions[self.q_ind].useSD():
            sdx_in = np.asarray(self.sim_sd(x_in))
        else:
            sdx_in = np.zeros(sx_in.size)
        fx_in = np.asarray(self.penalty_func(x_in, sx_in)).flatten()
        ax = self.acquisitions[self.q_ind].scalarize(fx_in, x_in,
                                                     sx_in, sdx_in)
        return ax

    def solve(self, x):
        """ Solve the surrogate problem by using random search followed by GPS.

        Args:
            x (np.ndarray): A 2d array containing a list of feasible
                design points used to warm start the search.

        Returns:
            np.ndarray: A 2d numpy.ndarray containing a list of potentially
            efficient design points that were found by the optimizers.

        """

        from parmoo.util import updatePF

        # Check that x is legal
        if isinstance(x, np.ndarray):
            if self.n != x.shape[1]:
                raise ValueError("The columns of x must match n")
            elif len(self.acquisitions) != x.shape[0]:
                raise ValueError("The rows of x must match the number " +
                                 "of acquisition functions")
        else:
            raise TypeError("x must be a numpy array")
        # Set the batch size
        batch_size = 1000
        # Initialize lists/arrays
        result = []
        lb_tmp = np.zeros(self.n)
        ub_tmp = np.ones(self.n)
        # For each acquisition function
        for j, acq in enumerate(self.acquisitions):
            # Create a new trust region
            rad = self.resetObjectives(x[j, :])
            lb_old = lb_tmp
            ub_old = ub_tmp
            lb_tmp[:] = np.maximum(self.lb[:], x[j, :] - rad)
            ub_tmp[:] = np.minimum(self.ub[:], x[j, :] + rad)
            # Check if TR has changed
            if j == 0 or np.any(np.abs(lb_old - lb_tmp) +
                                np.abs(ub_old - ub_tmp) > 1.0e-8):
                # Initialize the database
                data = {'x_vals': np.zeros((batch_size, self.n)),
                        'f_vals': np.zeros((batch_size, self.o)),
                        'c_vals': np.zeros((batch_size, 0))}
                # Loop over batch size until k == budget
                k = 0
                nondom = {}
                search_budget = self.opt_budget = self.gps_budget
                while (k < search_budget):
                    # Check how many new points to generate
                    k_new = min(search_budget, k + batch_size) - k
                    if k_new < batch_size:
                        data['x_vals'] = np.zeros((k_new, self.n))
                        data['f_vals'] = np.zeros((k_new, self.o))
                        data['c_vals'] = np.zeros((k_new, 0))
                    # Randomly generate k_new new points
                    for i in range(k_new):
                        if i == 0:
                            xi = x[j, :]
                        else:
                            xi = (np.random.sample(self.n) *
                                  (ub_tmp[:] - lb_tmp[:]) + lb_tmp[:])
                        data['x_vals'][i, :] = xi[:]
                        data['f_vals'][i, :] = self.penalty_func(xi)
                    # Update the PF
                    nondom = updatePF(data, nondom)
                    k += k_new
            f_vals = []
            if acq.useSD():
                f_vals = [acq.scalarize(fi, xi, self.simulations(xi),
                                        self.sim_sd(xi))
                          for fi, xi in zip(nondom['f_vals'],
                                            nondom['x_vals'])]
            else:
                m = self.simulations(nondom['x_vals'][0]).size
                f_vals = [acq.scalarize(fi, xi, self.simulations(xi),
                                        np.zeros(m))
                          for fi, xi in zip(nondom['f_vals'],
                                            nondom['x_vals'])]
            imin = np.argmin(np.asarray([f_vals]))
            x0 = nondom['x_vals'][imin, :].copy()
            # Get a candidate
            self.q_ind = j
            mesh_tol = max(1.0e-8, np.min((ub_tmp - lb_tmp) * 1.0e-4))
            xj, fj = __accelerated_pattern_search__(self.n, lb_tmp,
                                                    ub_tmp, x0,
                                                    self.__obj_func__,
                                                    ibudget=self.gps_budget,
                                                    mesh_start=0.1,
                                                    mesh_tol=mesh_tol,
                                                    momentum=self.momentum,
                                                    istarts=1)
            result.append(xj)
        return np.asarray(result)


def __accelerated_pattern_search__(n, lb, ub, x0, obj_func, ibudget,
                                   mesh_start=None, mesh_tol=1.0e-8,
                                   momentum=0.9, istarts=1):
    """ Solve the optimization problem min obj_func(x) over x in [lb, ub].

    Uses pattern search with 1 additional poll direction inspired by
    Nesterov's momentum.

    Args:
        n (int): The dimension of the search space.

        lb (np.ndarray): A 1D array of length n specifying lower bounds on
            the search space.

        ub (np.ndarray): A 1D array of length n specifying upper bounds on
            the search space.

        x0 (np.ndarray): A 1D array of length n specifying the initial
            location to sample when warm-starting search.

        obj_func (function): A function that takes a 1D numpy.ndarray of
            size n as input (x) and returns the value of f(x).

        ibudget (int): The iteration limit for pattern search. The total
            number of calls to obj_func could be up to 2n * ibudget + 1,
            but typically will be much less.

        mesh_start (float or numpy.ndarray, optional): The initial mesh
            spacing. If an array is given, must be 1D of size n. Defaults
            to (ub - lb) / (istarts - 1).

        mesh_tol (float or numpy.ndarray, optional): The tolerance for the
            mesh. If an array is given, must be 1D of size n. Defaults to
            1.0e-8.

        momentum (float, optional): The decay rate on the momentum term in
            the accelerated poll direction. Defaults to 0.9.

        istarts (int, optional): Number of times to (re)start if
            using the multistart option. Defaults to 1 (no restarts).

    Returns:
        numpy.ndarray(n): The best x observed so far (minimizer of obj_func).

    """

    # Initialize O(n)-sized work arrays and constants
    f_min = np.zeros(istarts)
    m_tmp = np.zeros(n)
    mesh_size = np.zeros(n)
    x_center = np.zeros(n)
    x_min = np.zeros((istarts, n))
    x_tmp = np.zeros(n)
    mesh_start_array = np.zeros(n)
    if mesh_start is None:
        mesh_start_array[:] = 1 / istarts
    elif isinstance(mesh_start_array, float):
        mesh_start_array[:] = mesh_start * (ub[:] - lb[:])
    else:
        mesh_start_array[:] = mesh_start
    if istarts > 1:
        lhs = LatinHypercube(n).random(istarts - 1)
    # Loop over all starts
    for kk in range(istarts):
        # Reset the mesh dimensions
        mesh_size[:] = mesh_start_array[:]
        mesh = np.vstack((np.zeros((1, n)),
                          np.diag(ub[:] - lb[:]),
                          -np.diag(ub[:] - lb[:])))
        # Evaluate the starting point or random point
        if kk == 0:
            x_min[kk, :] = x0[:]
        else:
            x_min[kk, :] = lhs[kk - 1] * (ub - lb) + lb
        f0 = obj_func(x_min[kk])
        f_tol = max(min(abs(f0)**2, 1.0e-8), 1.0e-16)
        f_min[kk] = f0
        # Take n+1 iterations to get "momentum" started
        k_start = 0
        for k in range(ibudget):
            improve = False
            x_center[:] = x_min[kk, :]
            for i, mi in enumerate(mesh[1:, :]):
                # Evaluate x + mi
                x_tmp[:] = x_center[:] + mi[:] * mesh_size[:]
                if np.any((x_tmp > ub) + (x_tmp < lb)):
                    f_tmp = np.inf
                else:
                    f_tmp = obj_func(x_tmp)
                # Check for improvement
                if f_min[kk] - f_tmp > f_tol:
                    f_min[kk] = f_tmp
                    x_min[kk, :] = x_tmp[:]
                    m_min = i + 1
                    improve = True
            # If there was improvement, shuffle the directions
            if improve:
                iswitch = np.where(np.abs(mesh[m_min, :]) > 1.0e-8)[0]
                mesh[0, :] *= momentum
                mesh[0, iswitch] = mesh[m_min, iswitch]
                m_tmp[:] = mesh[m_min, :]
                mesh[2:m_min+1, :] = mesh[1:m_min, :]
                mesh[1, :] = m_tmp[:]
                # Update k_start and break the "warm-up" loop
                k_start += 1
                if k_start >= n+1:
                    break
            # If no improvement, decay the mesh down to the tolerance
            else:
                if np.any(mesh_size[:] < mesh_tol):
                    break
                else:
                    mesh_size[:] *= 0.5
        # Take the remaining iterations
        for k in range(k_start, ibudget):
            improve = False
            x_center[:] = x_min[kk, :]
            for i, mi in enumerate(mesh[:, :]):
                # Evaluate x + mi
                x_tmp[:] = x_center[:] + np.rint(mi[:]) * mesh_size[:]
                if np.any((x_tmp > ub) + (x_tmp < lb)):
                    f_tmp = np.inf
                else:
                    f_tmp = obj_func(x_tmp)
                # Check for improvement
                if f_min[kk] - f_tmp > f_tol:
                    f_min[kk] = f_tmp
                    x_min[kk, :] = x_tmp[:]
                    m_min = i
                    improve = True
                    break
            # If there was improvement, update moment and shuffle directions
            if improve:
                if m_min > 0:
                    iswitch = np.where(np.abs(mesh[m_min, :]) > 1.0e-8)[0]
                    mesh[0, :] *= momentum
                    mesh[0, iswitch] = mesh[m_min, iswitch]
                    m_tmp[:] = mesh[m_min, :]
                    mesh[2:m_min+1, :] = mesh[1:m_min, :]
                    mesh[1, :] = m_tmp[:]
            # If no improvement, decay the mesh down to the tolerance
            else:
                if np.any(mesh_size[:] < mesh_tol):
                    break
                else:
                    mesh_size[:] *= 0.5
    imin = np.argmin(f_min)
    return x_min[imin].copy(), f_min[imin]
