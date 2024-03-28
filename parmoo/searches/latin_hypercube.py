
""" Implementations of the GlobalSearch class.

This module contains implementations of the GlobalSearch ABC, which are based
on the Latin hypercube design.

The classes include:
 * ``LatinHypercube`` -- Latin hypercube sampling

"""

import numpy as np
from parmoo.structs import GlobalSearch
from parmoo.util import xerror
from scipy.stats import qmc


class LatinHypercube(GlobalSearch):
    """ Implementation of a Latin hypercube search.

    This GlobalSearch strategy uses a Latin hypercube design to sample in the
    design space.

    """

    # Slots for the LatinHypercube class
    __slots__ = ['lb', 'ub', 'budget', 'sampler', 'np_rng']

    def __init__(self, m, lb, ub, hyperparams):
        """ Constructor for the LatinHypercube GlobalSearch class.

        Args:
            m (int): The number of simulation outputs (unused by this class).

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for the
                LatinHypercube design. It may contain:
                 * search_budget (int): The sim eval budget for the search

        Returns:
            LatinHypercube: A new LatinHypercube object.

        """

        # Check inputs
        xerror(o=m, lb=lb, ub=ub, hyperparams=hyperparams)
        self.n = lb.size
        # Assign the bounds
        self.lb = lb
        self.ub = ub
        # Check for a search budget
        if 'search_budget' in hyperparams:
            if isinstance(hyperparams['search_budget'], int):
                if hyperparams['search_budget'] < 0:
                    raise ValueError("hyperparams['search_budget'] must "
                                     "be nonnegative")
                else:
                    self.budget = hyperparams['search_budget']
            else:
                raise ValueError("hyperparams['search_budget'] must "
                                 "be an int")
        else:
            self.budget = 100
        self.sampler = qmc.LatinHypercube(d=self.n)
        if 'np_random_gen' in hyperparams:
            if isinstance(hyperparams['np_random_gen'], np.random.Generator):
                self.np_rng = hyperparams['np_random_gen']
            else:
                raise TypeError("When present, hyperparams['np_random_gen'] "
                                "must be an instance of the class "
                                "numpy.random.Generator")
        return

    def startSearch(self, lb, ub):
        """ Begin a new Latin hypercube sampling.

        Args:
            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The dimension must match n.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match n.

        Returns:
            np.ndarray: A 2d array, containing the list of design points
            to be evaluated.

        """

        # Check inputs
        xerror(lb=lb, ub=ub)
        # Assign the bounds
        self.lb = lb
        self.ub = ub
        # If the budget is 0, just return an empty array
        if self.budget == 0:
            return np.asarray([])
        # Otherwise, return a n-dimensional Latin hypercube design
        else:
            return qmc.scale(self.sampler.random(n=self.budget),
                             self.lb, self.ub)

    def resumeSearch(self):
        """ Resume a previous Latin hypercube sampling.

        Returns:
            np.ndarray: A 2d array, containing the list of design points
            to be evaluated.

        """

        # If the budget is 0, just return an empty array
        if self.budget == 0:
            return np.asarray([])
        # Otherwise, return a n-dimensional Latin hypercube design
        else:
            return qmc.scale(self.sampler.random(n=self.budget),
                             self.lb, self.ub)
