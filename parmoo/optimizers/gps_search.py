
""" Implementations of the SurrogateOptimizer class.

This module contains implementations of the SurrogateOptimizer ABC, which
are based on the GPS polling strategy for direct search.

Note that these strategies are all gradient-free, and therefore does not
require objective, constraint, or surrogate gradients methods to be defined.

The classes include:
 * ``LocalGPS`` -- Generalized Pattern Search (GPS) algorithm
 * ``QuickGPS`` -- GPS plus some smoothing of polls to take fewer f evals
 * ``GlobalGPS`` -- global random search, followed by GPS

"""

import numpy as np
from parmoo.structs import SurrogateOptimizer, AcquisitionFunction
from parmoo.util import xerror


class LocalGPS(SurrogateOptimizer):
    """ Use Generalized Pattern Search (GPS) to identify local solutions.

    Applies GPS to the surrogate problem, in order to identify design
    points that are locally Pareto optimal, with respect to the surrogate
    problem.

    """

    # Slots for the LocalGPS class
    __slots__ = ['n', 'lb', 'ub', 'acquisitions', 'budget', 'constraints',
                 'objectives', 'simulations', 'gradients', 'resetObjectives',
                 'penalty_func', 'sim_sd', 'restarts']

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
        # Check that the contents of hyperparams is legal
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
        # Check that the contents of hyperparams is legal
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
        self.acquisitions = []
        return

    def solve(self, x):
        """ Solve the surrogate problem using generalized pattern search (GPS).

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
        # For each acqusisition function
        for j, acquisition in enumerate(self.acquisitions):
            # Create a new trust region
            rad = self.resetObjectives(x[j, :])
            lb_tmp = np.zeros(self.n)
            ub_tmp = np.ones(self.n)
            for i in range(self.n):
                lb_tmp[i] = max(self.lb[i], x[j, i] - rad)
                ub_tmp[i] = min(self.ub[i], x[j, i] + rad)
            # Initialize temp arrays
            f_min = np.zeros(self.restarts)
            x_big = np.zeros(self.n)
            x_center = np.zeros(self.n)
            x_min = np.zeros((self.restarts, self.n))
            x_tmp = np.zeros(self.n)
            # Loop over restarts
            for kk in range(self.restarts):
                # Reset the mesh dimensions
                mesh = np.diag(ub_tmp[:] - lb_tmp[:] * 0.25)
                # Evaluate the starting point
                sx = np.asarray(self.simulations(x[j, :]))
                if acquisition.useSD():
                    sdx = np.asarray(self.sim_sd(x[j, :]))
                else:
                    sdx = np.zeros(sx.size)
                if kk == 0:
                    x_min[kk, :] = x[j, :]
                else:
                    x_min[kk, :] = (np.random.random_sample(self.n) *
                                    (ub_tmp - lb_tmp) + lb_tmp)
                fx = np.asarray(self.penalty_func(x_min[kk, :], sx))
                f_min[kk] = acquisition.scalarize(fx.flatten(), x_min[kk, :],
                                                  sx, sdx)
                # Loop over the budget
                for k in range(self.budget):
                    # Track whether or not there is improvement
                    improve = False
                    # Build an extra accelerating descent step
                    x_big[:] = 0.0
                    f_center = f_min[kk]
                    x_center[:] = x_min[kk, :]
                    for i in range(self.n):
                        # Evaluate x + mesh[:, i]
                        x_tmp = x_center[:] + mesh[:, i]
                        if any(x_tmp > ub_tmp):
                            f_plus = np.inf
                        else:
                            sx = np.asarray(self.simulations(x_tmp))
                            if acquisition.useSD():
                                sdx = self.sim_sd(x_tmp)
                            else:
                                sdx = 0.0
                            fx = self.penalty_func(x_tmp, sx)
                            f_plus = acquisition.scalarize(fx, x_tmp, sx, sdx)
                        # Check for improvement
                        if f_plus + 1.0e-8 < f_center:
                            if f_plus / f_center < 0.999:
                                x_big[i] = mesh[i, i]
                            if f_plus + 1.0e-8 < f_min[kk]:
                                f_min[kk] = f_plus
                                x_min[kk, :] = x_tmp
                                improve = True
                        # Evaluate x - mesh[:, i]
                        x_tmp = x_center[:] - mesh[:, i]
                        if any(x_tmp < lb_tmp):
                            f_minus = np.inf
                        else:
                            sx = np.asarray(self.simulations(x_tmp))
                            if acquisition.useSD():
                                sdx = self.sim_sd(x_tmp)
                            else:
                                sdx = 0.0
                            fx = self.penalty_func(x_tmp, sx)
                            f_minus = acquisition.scalarize(fx, x_tmp, sx, sdx)
                        # Check for improvement
                        if f_minus + 1.0e-8 < f_center:
                            if f_minus / f_center < 0.999 and f_minus < f_plus:
                                x_big[i] = -mesh[i, i]
                            if f_minus + 1.0e-8 < f_min[kk]:
                                f_min[kk] = f_minus
                                x_min[kk, :] = x_tmp
                                improve = True
                    # If improvement found, try taking a big step
                    if improve:
                        if np.count_nonzero(x_big) > 1:
                            x_tmp = x_center[:] + x_big[:]
                            if any(x_tmp > ub_tmp) or any(x_tmp < lb_tmp):
                                f_tmp = np.inf
                            else:
                                sx = np.asarray(self.simulations(x_tmp))
                                if acquisition.useSD():
                                    sdx = self.sim_sd(x_tmp)
                                else:
                                    sdx = 0.0
                                fx = self.penalty_func(x_tmp, sx)
                                f_tmp = acquisition.scalarize(fx, x_tmp,
                                                              sx, sdx)
                            if f_tmp + 1.0e-8 < f_min[kk]:
                                f_min[kk] = f_tmp
                                x_min[kk, :] = x_tmp
                    # If no improvement, decay the mesh down to the tolerance
                    else:
                        if any([mesh[i, i] < 1.0e-4 for i in range(self.n)]):
                            break
                        else:
                            mesh = mesh * 0.5
            # Append the found minima to the results list
            x_cand_ind = np.argmin(f_min)
            result.append(x_min[x_cand_ind, :].copy())
        return np.asarray(result)


class QuickGPS(SurrogateOptimizer):
    """ Use Generalized Pattern Search (GPS) to identify local solutions.

    Applies GPS to the surrogate problem, in order to identify design
    points that are locally Pareto optimal, with respect to the surrogate
    problem. Sorts poll directions by most recently used and attempts to
    step in promising directions in late iterations.

    """

    # Slots for the QuickGPS class
    __slots__ = ['n', 'lb', 'ub', 'acquisitions', 'budget', 'constraints',
                 'objectives', 'simulations', 'gradients', 'resetObjectives',
                 'penalty_func', 'sim_sd', 'restarts', 'momentum']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the QuickGPS class.

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
        # Check that the contents of hyperparams is legal
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
        # Check that the contents of hyperparams is legal
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
        # Check that the contents of hyperparams is legal
        if 'opt_momentum' in hyperparams:
            if isinstance(hyperparams['opt_momentum'], float):
                if 0 <= hyperparams['opt_budget'] < 1:
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

    @profile
    def solve(self, x):
        """ Solve the surrogate problem using generalized pattern search (GPS).

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
        # For each acqusisition function
        for j, acquisition in enumerate(self.acquisitions):
            # Create a new trust region
            rad = self.resetObjectives(x[j, :])
            mesh_size = np.zeros(self.n)
            m_tmp = np.zeros(self.n)
            lb_tmp = np.zeros(self.n)
            ub_tmp = np.ones(self.n)
            for i in range(self.n):
                lb_tmp[i] = max(self.lb[i], x[j, i] - rad)
                ub_tmp[i] = min(self.ub[i], x[j, i] + rad)
            # Initialize temp arrays
            f_min = np.zeros(self.restarts)
            x_center = np.zeros(self.n)
            x_min = np.zeros((self.restarts, self.n))
            x_tmp = np.zeros(self.n)
            # Loop over restarts
            for kk in range(self.restarts):
                # Reset the mesh dimensions
                mesh_size[:] = 0.25 * (ub_tmp[:] - lb_tmp[:])
                mesh = np.vstack((np.zeros((1,self.n)),
                                  np.diag(ub_tmp[:] - lb_tmp[:]),
                                  -np.diag(ub_tmp[:] - lb_tmp[:])))
                mesh_tol = max(1.0e-8, np.min((ub_tmp - lb_tmp) * 1.0e-4))
                # Evaluate the starting point
                sx = np.asarray(self.simulations(x[j, :]))
                if acquisition.useSD():
                    sdx = np.asarray(self.sim_sd(x[j, :]))
                else:
                    sdx = np.zeros(sx.size)
                if kk == 0:
                    x_min[kk, :] = x[j, :]
                else:
                    x_min[kk, :] = (np.random.random_sample(self.n) *
                                    (ub_tmp - lb_tmp) + lb_tmp)
                fx = np.asarray(self.penalty_func(x_min[kk, :], sx))
                f_min[kk] = acquisition.scalarize(fx.flatten(), x_min[kk, :],
                                                  sx, sdx)
                # Take n+1 iterations to start
                for k in range(self.n+1):
                    # Track whether or not there is improvement
                    improve = False
                    x_center[:] = x_min[kk, :]
                    for i, mi in enumerate(mesh[1:, :]):
                        # Evaluate x + mi
                        x_tmp[:] = x_center[:] + mi[:] * mesh_size[:]
                        if np.any((x_tmp > ub_tmp) + (x_tmp < lb_tmp)):
                            f_tmp = np.inf
                        else:
                            sx = np.asarray(self.simulations(x_tmp))
                            if acquisition.useSD():
                                sdx = self.sim_sd(x_tmp)
                            else:
                                sdx = 0.0
                            fx = self.penalty_func(x_tmp, sx)
                            f_tmp = acquisition.scalarize(fx, x_tmp, sx, sdx)
                        # Check for improvement
                        if f_tmp + 1.0e-8 < f_min[kk]:
                            f_min[kk] = f_tmp
                            x_min[kk, :] = x_tmp[:]
                            m_min = i + 1
                            improve = True
                    # If there was improvement, shuffle the directions
                    if improve:
                        m_tmp[:] = mesh[m_min, :]
                        mesh[2:m_min, :] = mesh[1:m_min-1, :]
                        mesh[1, :] = m_tmp[:]
                    # If no improvement, decay the mesh down to the tolerance
                    else:
                        if np.any(mesh_size[:] < mesh_tol):
                            break
                        else:
                            mesh_size[:] = mesh_size[:] * 0.5
                # Take the remaining iterations
                for k in range(self.n+1, self.budget):
                    # Track whether or not there is improvement
                    improve = False
                    x_center[:] = x_min[kk, :]
                    for i, mi in enumerate(mesh[:, :]):
                        # Evaluate x + mi
                        x_tmp[:] = x_center[:] + np.rint(mi[:]) * mesh_size[:]
                        if np.any((x_tmp > ub_tmp) + (x_tmp < lb_tmp)):
                            f_tmp = np.inf
                        else:
                            sx = np.asarray(self.simulations(x_tmp))
                            if acquisition.useSD():
                                sdx = self.sim_sd(x_tmp)
                            else:
                                sdx = 0.0
                            fx = self.penalty_func(x_tmp, sx)
                            f_tmp = acquisition.scalarize(fx, x_tmp, sx, sdx)
                        # Check for improvement
                        if f_tmp + 1.0e-8 < f_min[kk]:
                            f_min[kk] = f_tmp
                            x_min[kk, :] = x_tmp[:]
                            m_min = i
                            improve = True
                            break
                    # If there was improvement, shuffle the directions
                    if improve:
                        mesh[0, :] = (mesh[0, :] * (1 - self.momentum) +
                                      mesh[m_min, :] * self.momentum)
                        if m_min > 0:
                            m_tmp[:] = mesh[m_min, :]
                            mesh[2:m_min, :] = mesh[1:m_min-1, :]
                            mesh[1, :] = m_tmp[:]
                    # If no improvement, decay the mesh down to the tolerance
                    else:
                        if np.any(mesh_size[:] < mesh_tol):
                            break
                        else:
                            mesh_size[:] = mesh_size[:] * 0.5
            # Append the found minima to the results list
            x_cand_ind = np.argmin(f_min)
            result.append(x_min[x_cand_ind, :].copy())
        return np.asarray(result)


class GlobalGPS(SurrogateOptimizer):
    """ Use randomized search globally followed by GPS locally.

    Use ``RandomSearch`` to globally search the design space (search phase)
    followed by ``QuickGPS`` to refine the potentially efficient solutions
    (poll phase).

    """

    # Slots for the GlobalGPS class
    __slots__ = ['n', 'lb', 'ub', 'acquisitions', 'constraints', 'objectives',
                 'simulations', 'gradients', 'resetObjectives', 'penalty_func',
                 'search_budget', 'gps_budget', 'sim_sd']

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
                   (default: 10,000)
                 * gps_budget (int): The number of the total opt_budget
                   evaluations that will be used by GPS (default: half
                   of opt_budget).

        Returns:
            SurrogateOptimizer: A new SurrogateOptimizer object.

        """

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
        self.n = lb.size
        self.lb = lb
        self.ub = ub
        # Check that the contents of hyperparams are legal
        if 'opt_budget' in hyperparams:
            if isinstance(hyperparams['opt_budget'], int):
                if hyperparams['opt_budget'] < 1:
                    raise ValueError("hyperparams['opt_budget'] "
                                     "must be positive")
                else:
                    budget = hyperparams['opt_budget']
            else:
                raise TypeError("hyperparams['opt_budget'] "
                                 "must be an integer")
        else:
            budget = 1000
        # Check the GPS budget
        if 'gps_budget' in hyperparams:
            if isinstance(hyperparams['gps_budget'], int):
                if hyperparams['gps_budget'] < 1:
                    raise ValueError("hyperparams['gps_budget'] "
                                     "must be positive")
                elif hyperparams['gps_budget'] > budget:
                    raise ValueError("hyperparams['gps_budget'] "
                                     "must be less than "
                                     "hyperparams['opt_budget']")
                else:
                    self.gps_budget = hyperparams['gps_budget']
            else:
                raise TypeError("hyperparams['gps_budget'] "
                                 "must be an integer")
        else:
            self.gps_budget = 1000
        self.search_budget = budget
        # Initialize the list of acquisition functions
        self.acquisitions = []
        return

    def solve(self, x):
        """ Solve the surrogate problem by using random search followed by GPS.

        Args:
            x (np.ndarray): A 2d array containing a list of feasible
                design points used to warm start the search.

        Returns:
            np.ndarray: A 2d numpy.ndarray containing a list of potentially
            efficient design points that were found by the optimizers.

        """

        from .random_search import RandomSearch

        # Check that x is legal
        if isinstance(x, np.ndarray):
            if self.n != x.shape[1]:
                raise ValueError("The columns of x must match n")
            elif len(self.acquisitions) != x.shape[0]:
                raise ValueError("The rows of x must match the number " +
                                 "of acquisition functions")
        else:
            raise TypeError("x must be a numpy array")
        # Do a global search to get global solutions
        gs = RandomSearch(self.n, self.lb, self.ub,
                          {'opt_budget': self.search_budget})
        gs.setObjective(self.objectives)
        gs.setSimulation(self.simulations, self.sim_sd)
        gs.setPenalty(self.penalty_func, self.gradients)
        gs.setConstraints(self.constraints)
        gs.addAcquisition(*self.acquisitions)
        gs_soln = gs.solve(x)
        gps_budget_loc = int(self.gps_budget / gs_soln.shape[0])
        # Do a local search to refine the global solution
        ls = QuickGPS(self.n, self.lb, self.ub,
                      {'opt_budget': gps_budget_loc,
                       'opt_restarts': 1})
        ls.setObjective(self.objectives)
        ls.setSimulation(self.simulations, self.sim_sd)
        ls.setPenalty(self.penalty_func, self.gradients)
        ls.setConstraints(self.constraints)
        ls.addAcquisition(*self.acquisitions)
        ls.setReset(self.resetObjectives)
        ls_soln = ls.solve(gs_soln)
        # Return the list of local solutions
        return ls_soln
