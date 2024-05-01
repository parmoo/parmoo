
""" Implementations of Pattern Search (PS) and its variations.

This module contains implementations of the SurrogateOptimizer ABC, which
are based on pattern search.

Note that these strategies are all gradient-free, and therefore do not
require any gradients to be defined. This makes them friendly for
getting started.

The classes include:
 * ``LocalSurrogate_PS`` -- A multi-start pattern search (PS) algorithm
 * ``GlobalSurrogate_PS`` -- global random search, followed by LocalPS

"""

import jax
from jax import numpy as jnp
import numpy as np
from parmoo.structs import SurrogateOptimizer
from parmoo.util import xerror
from scipy.stats.qmc import LatinHypercube


class LocalSurrogate_PS(SurrogateOptimizer):
    """ Use multi-start Pattern Search to solve surrogate problem locally.

    Applies PS to the surrogate problem, in order to identify design
    points that are locally Pareto optimal, with respect to the surrogate
    problem. Sorts poll directions by most recently used and attempts to
    step in promising directions in late iterations.

    """

    # Slots for the LocalSurrogate_PS class
    __slots__ = ['n', 'lb', 'ub', 'acquisitions', 'budget', 'constraints',
                 'objectives', 'simulations', 'setTR',
                 'penalty_func', 'sim_sd', 'restarts', 'momentum', 'q_ind',
                 'prev_centers', 'des_tols', 'targets', 'np_rng', 'eps']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the LocalSurrogate_PS class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for the
                optimization procedure. It may contain the following:
                 * opt_budget (int): The PS iteration limit (default: 1000).
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
        self.eps = np.sqrt(jnp.finfo(jnp.ones(1)).eps)
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
            self.des_tols = (np.ones(self.n) * self.eps)
        # Check the hyperparameter dictionary for random generator
        if 'np_random_gen' in hyperparams:
            if isinstance(hyperparams['np_random_gen'], np.random.Generator):
                self.np_rng = hyperparams['np_random_gen']
            else:
                raise TypeError("When present, hyperparams['np_random_gen'] "
                                "must be an instance of the class "
                                "numpy.random.Generator")
        else:
            self.np_rng = np.random.default_rng()
        self.acquisitions = []
        self.prev_centers = []
        self.targets = []
        return

    def _obj_func(self, x_in):
        """ A wrapper for the objective function and acquisition.

        Args:
            x_in (np.ndarray): A 1d array with the design point to evaluate.

        Returns:
            float: The result of acquisition.scalarize(f(x_in, sim(x_in))).

        """

        sx_in = self.simulations(x_in)
        if self.acquisitions[self.q_ind].useSD():
            sdx_in = self.sim_sd(x_in)
        else:
            sdx_in = jnp.zeros(sx_in.size)
        fx_in = self.penalty_func(x_in, sx_in)
        ax = self.acquisitions[self.q_ind].scalarize(fx_in, x_in,
                                                     sx_in, sdx_in)
        return ax

    def _checkTR(self, center):
        """ Check the recommended trust region for a new center. """

        # Search the history for the given radius
        rad = np.zeros(self.n)
        for (ci, ri) in reversed(self.prev_centers):
            if np.all(np.abs(center - np.asarray(ci)) < self.des_tols):
                rad[:] = np.asarray(ri)
                break
        # If not found in the history initialize
        if np.all(rad == 0):
            rad = (self.ub - self.lb) * 0.1
            rad = np.maximum(rad, self.des_tols)
        return rad

    def _checkTargets(self):
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
        """ Solve the surrogate problem in a trust region via pattern search.

        Args:
            x (np.ndarray): A 2d array containing a list of feasible
                design points used to warm start the search.

        Returns:
            np.ndarray: A 2d numpy.ndarray of potentially efficient design
            points that were found by the PS optimizer.

        """

        # Check that x is legal
        if self.n != x.shape[1]:
            raise ValueError("The columns of x must match n")
        elif len(self.acquisitions) != x.shape[0]:
            raise ValueError("The rows of x must match the number " +
                             "of acquisition functions")
        # Initialize an empty list of results
        result = []
        lb_tmp = np.zeros(self.n)
        ub_tmp = np.ones(self.n)
        self._checkTargets()
        # For each acqusisition function
        for j, acquisition in enumerate(self.acquisitions):
            # Create a new trust region
            rad = self._checkTR(x[j, :])
            self.setTR(x[j, :], rad)
            lb_tmp[:] = np.maximum(self.lb[:], x[j, :] - rad)
            ub_tmp[:] = np.minimum(self.ub[:], x[j, :] + rad)
            # Recompile the objective function
            self.q_ind = j
            mesh_tol = max(self.eps,
                           np.min((ub_tmp - lb_tmp) * np.sqrt(self.eps)))
            try:
                obj_func = jax.jit(self._obj_func)
                _ = obj_func(x[j])
            except BaseException:
                obj_func = self._obj_func
            # Get a candidate
            xj, fj = _accelerated_pattern_search(self.n, lb_tmp,
                                                 ub_tmp, x[j],
                                                 obj_func,
                                                 ibudget=self.budget,
                                                 mesh_tol=mesh_tol,
                                                 momentum=self.momentum,
                                                 istarts=self.restarts,
                                                 np_rng=self.np_rng)
            result.append(xj)
            # We need to remember this "target" for later
            self.targets.append([x[j, :], rad, fj, j])
        self.objectives = None
        self.constraints = None
        self.penalty_func = None
        return np.asarray(result)

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        import json

        # Serialize PS object in dictionary
        ps_state = {'n': self.n,
                    'budget': self.budget,
                    'restarts': self.restarts,
                    'momentum': self.momentum,
                    'q_ind': self.q_ind}
        # Serialize numpy.ndarray objects
        ps_state['lb'] = self.lb.tolist()
        ps_state['ub'] = self.ub.tolist()
        ps_state['des_tols'] = self.des_tols.tolist()
        # Flatten arrays
        ps_state['prev_centers'] = []
        for (ci, ri) in self.prev_centers:
            ps_state['rev_centers'].append([ci.tolist(), ri.tolist()])
        ps_state['targets'] = []
        for ti in self.targets:
            ps_state['targets'].append([ti[0].tolist(), ti[1].tolist(),
                                        ti[2], ti[3]])
        # Save file
        with open(filename, 'w') as fp:
            json.dump(ps_state, fp)
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
            ps_state = json.load(fp)
        # Deserialize PS object from dictionary
        self.n = ps_state['n']
        self.budget = ps_state['budget']
        self.restarts = ps_state['restarts']
        self.momentum = ps_state['momentum']
        self.q_ind = ps_state['q_ind']
        # Deserialize numpy.ndarray objects
        self.lb = np.array(ps_state['lb'])
        self.ub = np.array(ps_state['ub'])
        self.des_tols = np.array(ps_state['des_tols'])
        # Extract history arrays
        self.prev_centers = []
        for (ci, ri) in ps_state['prev_centers']:
            self.prev_centers.append([np.array(ci), np.array(ri)])
        self.targets = []
        for ti in ps_state['targets']:
            self.targets.append([np.array(ti[0]), np.array(ti[1]),
                                 ti[2], ti[3]])
        return


class GlobalSurrogate_PS(SurrogateOptimizer):
    """ Use randomized search globally followed by a local PS.

    Use ``RandomSearch`` to globally search the design space followed
    ``LocalSurrogate_PS`` to refine the potentially efficient solutions.

    """

    # Slots for the GlobalSurrogate_PS class
    __slots__ = ['n', 'o', 'lb', 'ub', 'acquisitions', 'constraints',
                 'objectives', 'simulations', 'setTR',
                 'penalty_func', 'opt_budget', 'gps_budget', 'sim_sd',
                 'momentum', 'np_rng', 'eps']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the GlobalPS class.

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
                 * ps_budget (int): The number of the total opt_budget
                   evaluations that will be used by PS (default: 2/3
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
        self.eps = np.sqrt(jnp.finfo(jnp.ones(1)).eps)
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
        # Check the hyperparameter dictionary for random generator
        if 'np_random_gen' in hyperparams:
            if isinstance(hyperparams['np_random_gen'], np.random.Generator):
                self.np_rng = hyperparams['np_random_gen']
            else:
                raise TypeError("When present, hyperparams['np_random_gen'] "
                                "must be an instance of the class "
                                "numpy.random.Generator")
        else:
            self.np_rng = np.random.default_rng()
        self.acquisitions = []
        return

    def _obj_func(self, x_in):
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
        """ Solve the surrogate problem by using random search followed by PS.

        Args:
            x (np.ndarray): A 2d array containing a list of feasible
                design points used to warm start the search.

        Returns:
            np.ndarray: A 2d numpy.ndarray containing a list of potentially
            efficient design points that were found by the optimizers.

        """

        from parmoo.util import updatePF

        # Check that x is legal
        if self.n != x.shape[1]:
            raise ValueError("The columns of x must match n")
        elif len(self.acquisitions) != x.shape[0]:
            raise ValueError("The rows of x must match the number " +
                             "of acquisition functions")
        # Create an infinite trust region
        rad = np.ones(self.n) * np.infty
        self.setTR(self.lb, rad)
        # Perform a random search globally
        batch_size = 1000
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
                if i < len(self.acquisitions):
                    xi = x[i, :]
                else:
                    xi = (self.np_rng.random(self.n) *
                          (self.ub[:] - self.lb[:]) + self.lb[:])
                sxi = self.simulations(xi)
                data['x_vals'][i, :] = xi[:]
                data['f_vals'][i, :] = self.penalty_func(xi, sxi)
            # Update the PF
            nondom = updatePF(data, nondom)
            k += k_new
        f_vals = []
        # Loop over acquisition functions and perform pattern search
        result = []
        for j, acq in enumerate(self.acquisitions):
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
            # Recompile the objective function
            self.q_ind = j
            mesh_tol = max(self.eps,
                           np.min((self.ub - self.lb) * np.sqrt(self.eps)))
            try:
                obj_func = jax.jit(self._obj_func)
                _ = obj_func(x[j])
            except BaseException:
                obj_func = self._obj_func
            # Get a candidate
            xj, fj = _accelerated_pattern_search(self.n, self.lb,
                                                 self.ub, x0,
                                                 obj_func,
                                                 ibudget=self.gps_budget,
                                                 mesh_start=0.1,
                                                 mesh_tol=mesh_tol,
                                                 momentum=self.momentum,
                                                 istarts=1)
            result.append(xj)
        self.objectives = None
        self.constraints = None
        self.penalty_func = None
        return np.asarray(result)


def _accelerated_pattern_search(n, lb, ub, x0, obj_func, ibudget,
                                mesh_start=None, mesh_tol=1.0e-8,
                                momentum=0.9, istarts=1, np_rng=None):
    """ Solve the optimization problem min obj_func(x) over x in [lb, ub].

    Uses pattern search with an additional search direction inspired by
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

        np_rng (numpy.random.Generator, optional): An instance of a numpy
            generator object that will be used for generating consistent
            restarts. Defaults to None, which results in a new generator
            being created.

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
        lhs = LatinHypercube(n, seed=np_rng).random(istarts - 1)
    if np_rng is None:
        np_rng = np.random.default_rng()
    # Check working tolerance (unit roundoff)
    mu = jnp.finfo(jnp.ones(1)).eps
    root_mu = np.sqrt(mu)
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
        f_tol = max(min(abs(f0)**2, root_mu), mu)
        f_min[kk] = f0
        # Take n+1 iterations to get "momentum" started
        k_start = 0
        for k in range(ibudget):
            improve = False
            x_center[:] = x_min[kk, :]
            for i, mi in enumerate(mesh[1:, :]):
                # Evaluate x + mi
                x_tmp[:] = x_center[:] + mi[:] * mesh_size[:]
                if _check_bounds(x_tmp, lb, ub):
                    f_tmp = np.inf
                else:
                    f_tmp = obj_func(x_tmp)
                # Check for improvement
                if _check_improv(f_min[kk], f_tmp, f_tol):
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
                if _check_bounds(x_tmp, lb, ub):
                    f_tmp = np.inf
                else:
                    f_tmp = obj_func(x_tmp)
                # Check for improvement
                if _check_improv(f_min[kk], f_tmp, f_tol):
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


@jax.jit
def _check_improv(f_min, f_tmp, f_tol):
    return f_min - f_tmp > f_tol


@jax.jit
def _check_bounds(x_tmp, lb, ub):
    return np.any((x_tmp > ub) + (x_tmp < lb))
