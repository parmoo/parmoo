
""" Implementations of the GlobalSearch class.

This module contains implementations of the GlobalSearch ABC, which are based
on the Latin hypercube design.

The classes include:
 * ``LatinHypercube`` -- Latin hypercube sampling

"""

import numpy as np
from parmoo.searches.global_search import GlobalSearch
from parmoo.utilities.error_checks import get_hp, xerror
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

        xerror(o=m, lb=lb, ub=ub, hyperparams=hyperparams)
        self.n = lb.size
        self.lb = lb
        self.ub = ub
        self.budget = get_hp(
                "search_budget", hyperparams, int, lambda x: x >= 0, 100
        )
        self.np_rng = get_hp(
                "np_random_gen", hyperparams, np.random.Generator,
                lambda x: True, np.random.default_rng()
        )
        self.sampler = qmc.LatinHypercube(d=self.n, seed=self.np_rng)

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
