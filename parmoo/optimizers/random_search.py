
""" Optimizers based on random search (RS).

This module contains implementations of the SurrogateOptimizer ABC, which
are based on randomized search strategies.

Note that these strategies are all gradient-free, and therefore does not
require objective, constraint, or surrogate gradients methods to be defined.

The classes include:
 * ``GlobalSurrogate_RS`` -- optimize surrogates globally via RS

"""

import jax
from jax import numpy as jnp
import numpy as np
from parmoo.structs import SurrogateOptimizer, AcquisitionFunction
from parmoo.util import xerror


class GlobalSurrogate_RS(SurrogateOptimizer):
    """ Use randomized search to identify potentially efficient designs.

    Randomly search the design space and use the surrogate models to predict
    whether each search point is potentially Pareto optimal.

    """

    # Slots for the GlobalSurrogate_RS class
    __slots__ = ['n', 'o', 'lb', 'ub', 'acquisitions', 'constraints',
                 'objectives', 'budget', 'simulations', 'setTR',
                 'penalty_func', 'sim_sd', 'np_rng']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the GlobalSurrogate_RS class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for the
                optimization procedure. It may contain the following:
                 * opt_budget (int): The sample size (default 10,000)

        Returns:
            SurrogateOptimizer: A new SurrogateOptimizer object.

        """

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
        self.o = o
        self.n = lb.size
        self.lb = lb
        self.ub = ub
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
            self.budget = 10000
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
        # Initialize the list of acquisition functions
        self.acquisitions = []
        return

    def solve(self, x):
        """ Solve the surrogate problem using random search.

        Args:
            x (np.ndarray): A 2d array containing a list of feasible
                design points used to warm start the search.

        Returns:
            np.ndarray: A 2d numpy.ndarray containing a list of potentially
            efficient design points that were found by the random search.

        """

        from parmoo.util import updatePF

        # Check that x is legal
        if self.n != x.shape[1]:
            raise ValueError("The columns of x must match n")
        elif len(self.acquisitions) != x.shape[0]:
            raise ValueError("The rows of x must match the number " +
                             "of acquisition functions")
        # Initialize the surrogates with an infinite trust region
        rad = np.ones(self.n) * np.infty
        self.setTR(x[0, :], rad)
        # Compile the penalty function
        try:
            pen_func = jax.jit(self.penalty_func)
            x0 = x[0, :]
            sx0 = self.simulations(x0)
            f0 = pen_func(x0, sx0)
        except BaseException:
            pen_func = self.penalty_func
        # Set the batch size
        batch_size = 1000
        # Initialize the database
        data = {'x_vals': np.zeros((batch_size, self.n)),
                'f_vals': np.zeros((batch_size, self.o)),
                'c_vals': np.zeros((batch_size, 0))}
        # Loop over batch size until k == budget
        k = 0
        nondom = {}
        while (k < self.budget):
            # Check how many new points to generate
            k_new = min(self.budget, k + batch_size) - k
            if k_new < batch_size:
                data['x_vals'] = np.zeros((k_new, self.n))
                data['f_vals'] = np.zeros((k_new, self.o))
                data['c_vals'] = np.zeros((k_new, 0))
            # Randomly generate k_new new points
            for i in range(k_new):
                xi = (self.np_rng.random(self.n) *
                      (self.ub[:] - self.lb[:]) + self.lb[:])
                data['x_vals'][i, :] = xi[:]
                sxi = self.simulations(xi)
                data['f_vals'][i, :] = pen_func(xi, sxi)
            # Update the PF
            nondom = updatePF(data, nondom)
            k += k_new
        # Extract results for each scalarization via random search
        results = []
        for iq, acq in enumerate(self.acquisitions):
            # Compile the scalarization function
            if acq.useSD():
                _sca_func = lambda fi, xi: acq.scalarize(fi, xi,
                                                         self.simulations(xi),
                                                         self.sim_sd(xi))
            else:
                _sca_func = lambda fi, xi: acq.scalarize(fi, xi,
                                                         self.simulations(xi),
                                                         jnp.zeros(sxi.size))
            try:
                sca_func = jax.jit(_sca_func)
                q0 = sca_func(f0, x0)
            except BaseException:
                sca_func = _sca_func
            # Use acquisition functions to extract array of results
            f_vals = [sca_func(fi, xi) for fi, xi in zip(nondom['f_vals'],
                                                         nondom['x_vals'])]
            imin = np.argmin(np.asarray([f_vals]))
            results.append(nondom['x_vals'][imin, :])
        self.objectives = None
        self.constraints = None
        self.penalty_func = None
        return np.asarray(results)
