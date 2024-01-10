
""" Optimization methods based on limited-memory BFGS-B (L-BFGS-B).

This module contains implementations of the SurrogateOptimizer ABC, which
are based on the L-BFGS-B quasi-Newton algorithm.

Note that all of these methods are gradient based, and therefore require
objective, constraint, and surrogate gradient methods to be defined.

The classes include:
 * ``GlobalSurrogate_BFGS`` -- Minimize the surrogate globally via L-BFGS-B
 * ``LocalSurrogate_BFGS`` -- Minimize surrogate in trust region via L-BFGS-B

"""

import numpy as np
from parmoo.structs import SurrogateOptimizer, AcquisitionFunction
from parmoo.util import xerror


class GlobalSurrogate_BFGS(SurrogateOptimizer):
    """ Use L-BFGS-B and gradients to identify local solutions.

    Applies L-BFGS-B to the surrogate problem, in order to identify design
    points that are locally Pareto optimal with respect to the surrogate
    problem.

    """

    # Slots for the GlobalSurrogate_BFGS class
    __slots__ = ['n', 'bounds', 'acquisitions', 'budget', 'constraints',
                 'objectives', 'simulations', 'gradients', 'setTR',
                 'penalty_func', 'sim_sd']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the GlobalSurrogate_BFGS class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for the
                optimization procedure. It may contain the following:
                 * opt_budget (int): The evaluation budget per solve
                   (default: 1000).
                 * opt_restarts (int): Number of multisolve restarts per
                   scalarization (default: n+1).

        Returns:
            SurrogateOptimizer: A new SurrogateOptimizer object.

        """

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
        self.n = lb.size
        self.bounds = np.zeros((self.n, 2))
        self.bounds[:, 0] = lb
        self.bounds[:, 1] = ub
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
        """ Solve the surrogate problem using L-BFGS-B.

        Args:
            x (np.ndarray): A 2d array containing a list of design points
                used to warm start the search.

        Returns:
            np.ndarray: A 2d numpy.ndarray of potentially efficient design
            points that were found by L-BFGS-B.

        """

        from scipy import optimize

        # Check that x is legal
        if isinstance(x, np.ndarray):
            if self.n != x.shape[1]:
                raise ValueError("The columns of x must match n")
            elif len(self.acquisitions) != x.shape[0]:
                raise ValueError("The rows of x must match the number " +
                                 "of acquisition functions")
        else:
            raise TypeError("x must be a numpy array")
        # Check that x is feasible.
        for xj in x:
            if np.any(xj[:] < self.bounds[:, 0]) or \
               np.any(xj[:] > self.bounds[:, 1]):
                raise ValueError("some of starting points (x) are infeasible")
        # Create an infinite trust region
        rad = np.ones(self.n) * np.infty
        self.setTR(x[j, :], rad)
        # Loop over and solve acqusisition functions
        result = []
        for j, acquisition in enumerate(self.acquisitions):

            # Define the scalarized wrapper functions
            if acquisition.useSD():

                def scalar_f(x, *args):
                    sx = self.simulations(x)
                    sdx = self.sim_sd(x)
                    fx = self.penalty_func(x, sx)
                    return acquisition.scalarize(fx, x, sx, sdx)

            else:

                def scalar_f(x, *args):
                    sx = self.simulations(x)
                    sdx = np.zeros(sx.size)
                    fx = self.penalty_func(x, sx)
                    return acquisition.scalarize(fx, x, sx, sdx)

            def scalar_g(x, *args):
                return acquisition.scalarizeGrad(self.penalty_func(x),
                                                 self.gradients(x))

            # Get the solution via multistart solve
            soln = x[j, :].copy()
            for i in range(self.restarts):
                if i == 0:
                    # Use center point to warm-start first start
                    x0 = x[j, :].copy()
                elif i == 1:
                    # Use predicted gradient step to warm-start second start
                    x0 = x[j, :].copy()
                    gg = scalar_g(x0)
                    for ii in range(self.n):
                        if gg[ii] < 0:
                            x0[ii] = self.bounds[ii, 1]
                        elif gg[ii] > 0:
                            x0[ii] = self.bounds[ii, 0]
                else:
                    # Random starting point within bounds for all other starts
                    x0 = (np.random.random_sample(self.n) *
                          (self.bounds[:, 1] - self.bounds[:, 0]) +
                          self.bounds[:, 0])

                # Solve the problem globally within bound constraints
                res = optimize.minimize(scalar_f, x0, method='L-BFGS-B',
                                        jac=scalar_g, bounds=self.bounds,
                                        options={'maxiter': self.budget})
                if scalar_f(res['x']) < scalar_f(soln):
                    soln = res['x']
            # Append the found minima to the results list
            result.append(soln)
        return np.asarray(result)


class LocalSurrogate_BFGS(SurrogateOptimizer):
    """ Use L-BFGS-B and gradients to identify solutions within a trust region.

    Applies L-BFGS-B to the surrogate problem, in order to identify design
    points that are locally Pareto optimal with respect to the surrogate
    problem.

    """

    # Slots for the LBFGSB class
    __slots__ = ['n', 'bounds', 'acquisitions', 'budget', 'constraints',
                 'objectives', 'gradients', 'penalty_func', 'setTR',
                 'restarts', 'simulations', 'sim_sd']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the LocalSurrogate_BFGS class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for the
                optimization procedure. It may contain the following:
                 * opt_budget (int): The evaluation budget per solve
                   (default: 1000).
                 * opt_restarts (int): Number of multisolve restarts per
                   scalarization (default: 2).

        Returns:
            SurrogateOptimizer: A new SurrogateOptimizer object.

        """

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
        self.n = lb.size
        self.bounds = np.zeros((self.n, 2))
        self.bounds[:, 0] = lb
        self.bounds[:, 1] = ub
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
            self.restarts = 2
        self.acquisitions = []
        return

    def __checkTR(self, center):
        """ Check the recommended trust region for a new center. """

        # Search the history for the given radius
        rad = np.zeros(self.n)
        for (ci, ri) in self.prev_centers:
            if np.all(center - ci < self.des_tol):
                rad[:] = ri[:]
                break
        # If found in the history, decay the radius
        if np.any(rad > 0):
            rad = rad * 0.5
            rad = np.maximum(rad, self.des_tol)
        else:
            rad = np.minimum(rad, (self.ub - self.lb) * 0.05)
            rad = np.maximum(rad, self.des_tol)
        # Update the history
        self.prev_centers.append((center, rad))
        return rad

    def solve(self, x):
        """ Solve the surrogate problem using L-BFGS-B.

        Args:
            x (np.ndarray): A 2d array containing a list of design points
                used to warm start the search.

        Returns:
            np.ndarray: A 2d numpy.ndarray of potentially efficient design
            points that were found by L-BFGS-B.

        """

        from scipy import optimize

        # Check that x is legal
        if isinstance(x, np.ndarray):
            if self.n != x.shape[1]:
                raise ValueError("The columns of x must match n")
            elif len(self.acquisitions) != x.shape[0]:
                raise ValueError("The rows of x must match the number " +
                                 "of acquisition functions")
        else:
            raise TypeError("x must be a numpy array")
        # Check that x is feasible.
        for xj in x:
            if np.any(xj[:] < self.bounds[:, 0]) or \
               np.any(xj[:] > self.bounds[:, 1]):
                raise ValueError("some of starting points (x) are infeasible")
        # Initialize an empty list of results
        result = []
        # For each acqusisition function
        for j, acquisition in enumerate(self.acquisitions):

            # Define the scalarized wrapper functions
            if acquisition.useSD():

                def scalar_f(x, *args):
                    sx = self.simulations(x)
                    sdx = self.sim_sd(x)
                    fx = self.penalty_func(x, sx)
                    return acquisition.scalarize(fx, x, sx, sdx)

            else:

                def scalar_f(x, *args):
                    sx = self.simulations(x)
                    sdx = np.zeros(sx.size)
                    fx = self.penalty_func(x, sx)
                    return acquisition.scalarize(fx, x, sx, sdx)

            def scalar_g(x, *args):
                return acquisition.scalarizeGrad(self.penalty_func(x),
                                                 self.gradients(x))

            # Create a new trust region
            rad = self.__checkTR(x[j, :], rad)
            self.resetTR(x[j, :], rad)
            bounds = np.zeros((self.n, 2))
            bounds[:, 0] = np.maximum(self.bounds[:, 0], x[j, :] - rad)
            bounds[:, 1] = np.minimum(self.bounds[:, 1], x[j, :] + rad)

            # Get the solution via multistart solve
            soln = x[j, :].copy()
            f0 = scalar_f(soln)
            for i in range(self.restarts):
                if i == 0:
                    # Use center point to warm-start first start
                    x0 = x[j, :].copy()
                elif i == 1:
                    # Use predicted gradient step to warm-start second start
                    x0 = x[j, :].copy()
                    gg = scalar_g(x0)
                    for ii in range(self.n):
                        if gg[ii] < 0:
                            x0[ii] = bounds[ii, 1]
                        elif gg[ii] > 0:
                            x0[ii] = bounds[ii, 0]
                else:
                    # Random starting point within bounds for all other starts
                    x0 = (np.random.random_sample(self.n) *
                          (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0])

                # Solve the problem within the local trust region
                res = optimize.minimize(scalar_f, x0, method='L-BFGS-B',
                                        jac=scalar_g, bounds=bounds,
                                        options={'maxiter': self.budget})
                if scalar_f(res['x']) < scalar_f(soln):
                    soln = res['x']
            # Append the found minima to the results list
            result.append(soln)
        return np.asarray(result)
