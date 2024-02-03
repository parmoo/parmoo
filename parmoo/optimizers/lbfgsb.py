
""" Optimization methods based on limited-memory BFGS-B (L-BFGS-B).

This module contains implementations of the SurrogateOptimizer ABC, which
are based on the L-BFGS-B quasi-Newton algorithm.

Note that all of these methods are gradient based, and therefore require
objective, constraint, and surrogate gradient methods to be defined.

The classes include:
 * ``GlobalSurrogate_BFGS`` -- Minimize the surrogate globally via L-BFGS-B
 * ``LocalSurrogate_BFGS`` -- Minimize surrogate in trust region via L-BFGS-B

"""

from jax import jacrev
from jax import numpy as jnp
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
                 'objectives', 'simulations', 'setTR', 'penalty_func',
                 'sim_sd']

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
        if self.n != x.shape[1]:
            raise ValueError("The columns of x must match n")
        elif len(self.acquisitions) != x.shape[0]:
            raise ValueError("The rows of x must match the number " +
                             "of acquisition functions")
        # Check that x is feasible.
        for xj in x:
            if np.any(xj[:] < self.bounds[:, 0]) or \
               np.any(xj[:] > self.bounds[:, 1]):
                raise ValueError("some of starting points (x) are infeasible")
        # Create an infinite trust region
        rad = np.ones(self.n) * np.infty
        self.setTR(np.zeros(self.n), rad)
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

            scalar_g1 = jacrev(scalar_f)
            def scalar_g(x, *args): return np.asarray(scalar_g1(x, *args)).flatten()

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
                 'objectives', 'penalty_func', 'setTR',
                 'restarts', 'simulations', 'sim_sd', 'prev_centers',
                 'des_tols', 'targets']

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
        if 'des_tols' in hyperparams:
            if isinstance(hyperparams['des_tols'], np.ndarray):
                if hyperparams['des_tols'].size != self.n:
                    raise ValueError("the length of hyperparpams['des_tols']"
                                     " must match the length of lb and ub")
                if not np.all(hyperparams['des_tols']):
                    raise ValueError("all entries in hyperparams['des_tols']"
                                     " must be greater than 0")
            else:
                raise TypeError("hyperparams['des_tols'] must be an array.")
            self.des_tols = np.asarray(hyperparams['des_tols'])
        else:
            self.des_tols = (np.ones(self.n) *
                             float(jnp.sqrt(jnp.finfo(jnp.ones(1)).eps)))
        self.acquisitions = []
        self.prev_centers = []
        self.targets = []
        return

    def __checkTR(self, center):
        """ Check the recommended trust region for a new center. """

        # Search the history for the given radius
        rad = np.zeros(self.n)
        for (ci, ri) in reversed(self.prev_centers):
            if np.all(np.abs(center - np.asarray(ci)) < self.des_tols):
                rad[:] = np.asarray(ri)
                break
        # If not found in the history initialize
        if np.all(rad == 0):
            rad = (self.bounds[:, 1] - self.bounds[:, 0]) * 0.1
            rad = np.maximum(rad, self.des_tols)
        return rad

    def __checkTargets(self):
        """ Use internal list of targets to check and update the TR radii """

        for ti in self.targets:
            # Decay all "missed" targets' TR radii
            ci, ri, _, _ = ti
            ri = np.maximum(ri * 0.5, self.des_tols)
            # Update the TR history
            found = False
            j = 0
            for (cj, rj) in reversed(self.prev_centers):
                j -= 1
                if np.all(np.abs(cj - ci) < self.des_tols):
                    self.prev_centers[j][-1] = ri
            if not found:
                self.prev_centers.append([ci, ri])
        # Reset the list of targets for next iteration
        self.targets = []
        return

    def returnResults(self, x, fx, sx, sdx):
        """ Collect the results of a function evaluation.

        Args:
            x (np.ndarray): The design point evaluated.

            fx (np.ndarray): The objective function values at x.

            sx (np.ndarray): The simulation function values at x.

            sdx (np.ndarray): The standard deviation in the simulation
                outputs at x.

        """

        for i, ti in enumerate(self.targets):
            j = ti[3]
            fxj = self.acquisitions[j].scalarize(fx, x, sx, sdx)
            # Remove any targets that have been "hit"
            if fxj < ti[2]:
                del self.targets[i]
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
        if self.n != x.shape[1]:
            raise ValueError("The columns of x must match n")
        elif len(self.acquisitions) != x.shape[0]:
            raise ValueError("The rows of x must match the number " +
                             "of acquisition functions")
        # Check that x is feasible.
        for xj in x:
            if np.any(xj[:] < self.bounds[:, 0]) or \
               np.any(xj[:] > self.bounds[:, 1]):
                raise ValueError("some of starting points (x) are infeasible")
        # Reset targets and decay any trust regions from previous iteration
        self.__checkTargets()
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

            scalar_g1 = jacrev(scalar_f)
            def scalar_g(x, *args): return np.asarray(scalar_g1(x, *args)).flatten()

            # Create a new trust region
            rad = self.__checkTR(x[j, :])
            self.setTR(x[j, :], rad)
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
                xj = res['x']
                fj = scalar_f(res['x'])
                if fj < scalar_f(soln):
                    soln = xj
            # Append the found minima to the results list
            result.append(soln)
            # We need to remember this "target" for later
            self.targets.append([x[j, :], rad, fj, j])
        return np.asarray(result)

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        import json

        # Serialize BFGS object in dictionary
        bfgs_state = {'n': self.n,
                      'budget': self.budget,
                      'restarts': self.restarts}
        # Serialize numpy.ndarray objects
        bfgs_state['bounds'] = self.bounds.tolist()
        bfgs_state['des_tols'] = self.des_tols.tolist()
        # Flatten arrays
        bfgs_state['prev_centers'] = []
        for (ci, ri) in self.prev_centers:
            bfgs_state['prev_centers'].append([ci.tolist(), ri.tolist()])
        bfgs_state['targets'] = []
        for ti in self.targets:
            bfgs_state['targets'].append([ti[0].tolist(), ti[1].tolist(), ti[2], ti[3]])
        # Save file
        with open(filename, 'w') as fp:
            json.dump(bfgs_state, fp)
        return

    def load(self, filename):
        """ Reload important data into this class after a previous save.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data has been saved.

        """

        import json

        # Load file
        with open(filename, 'r') as fp:
            bfgs_state = json.load(fp)
        # Deserialize BFGS object from dictionary
        self.n = bfgs_state['n']
        self.budget = bfgs_state['budget']
        self.restarts = bfgs_state['restarts']
        # Deserialize numpy.ndarray objects
        self.bounds = np.array(bfgs_state['bounds'])
        self.des_tols = np.array(bfgs_state['des_tols'])
        # Extract history arrays
        self.prev_centers = []
        for (ci, ri) in bfgs_state['prev_centers']:
            self.prev_centers.append([np.array(ci), np.array(ri)])
        self.targets = []
        for ti in bfgs_state['targets']:
            self.targets.append([np.array(ti[0]), np.array(ti[1]), ti[2], ti[3]])
        return
