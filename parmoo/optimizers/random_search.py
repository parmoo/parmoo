
""" Optimizers based on random search (RS).

This module contains implementations of the SurrogateOptimizer ABC, which
are based on randomized search strategies.

Note that these strategies are all gradient-free, and therefore does not
require objective, constraint, or surrogate gradients methods to be defined.

The classes include:
 * ``GlobalSurrogate_RS`` -- optimize surrogates globally via RS

"""

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
    __slots__ = ['n', 'o', 'lb', 'ub', 'acquisitions', 'constraints', 'objectives',
                 'budget', 'simulations', 'setTR',
                 'penalty_func', 'sim_sd']

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
                xi = (np.random.sample(self.n) *
                      (self.ub[:] - self.lb[:]) + self.lb[:])
                data['x_vals'][i, :] = xi[:]
                sxi = self.simulations(xi)
                data['f_vals'][i, :] = self.penalty_func(xi, sxi)
            # Update the PF
            nondom = updatePF(data, nondom)
            k += k_new
        # Use acquisition functions to extract array of results
        results = []
        for iq, acq in enumerate(self.acquisitions):
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
            results.append(nondom['x_vals'][imin, :])
        return np.asarray(results)
