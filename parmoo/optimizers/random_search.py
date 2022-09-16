
""" Implementations of the SurrogateOptimizer class.

This module contains implementations of the SurrogateOptimizer ABC, which
are based on randomized search strategies.

Note that these strategies are all gradient-free, and therefore does not
require objective, constraint, or surrogate gradients methods to be defined.

The classes include:
 * ``RandomSearch`` -- search globally by generating random samples

"""

import numpy as np
import inspect
from parmoo.structs import SurrogateOptimizer, AcquisitionFunction


class RandomSearch(SurrogateOptimizer):
    """ Use randomized search to identify potentially efficient designs.

    Randomly search the design space and use the surrogate models to predict
    whether each search point is potentially Pareto optimal.

    """

    # Slots for the RandomSearch class
    __slots__ = ['n', 'lb', 'ub', 'acquisitions', 'constraints', 'objectives',
                 'budget']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the RandomSearch class.

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

        from parmoo.util import xerror

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
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

    def setObjective(self, obj_func):
        """ Add a vector-valued objective function that will be solved.

        Args:
            obj_func (function): A vector-valued function that can be evaluated
                to solve the surrogate optimization problem.

        """

        # Check whether obj_func() has an appropriate signature
        if callable(obj_func):
            if len(inspect.signature(obj_func).parameters) != 1:
                raise ValueError("obj_func() must accept exactly one input")
            else:
                # Add obj_func to the problem
                self.objectives = obj_func
        else:
            raise TypeError("obj_func() must be callable")
        return

    def setReset(self, reset):
        """ Add a reset function for resetting surrogate updates.

        This method is not used by this class.

        """

        return

    def setPenalty(self, penalty_func, grad_func):
        """ Add a matrix-valued gradient function for obj_func.

        Args:
            penalty_func (function): A vector-valued penalized objective
                that incorporates a penalty for violating constraints.

            grad_func (function): A matrix-valued function that can be
                evaluated to obtain the Jacobian matrix for obj_func.

        """

        # Do nothing, RandomSearch is gradient free
        return

    def setConstraints(self, constraint_func):
        """ Add a constraint function that will be satisfied.

        Args:
            constraint_func (function): A vector-valued function from the
                design space whose components correspond to constraint
                violations. If the problem is unconstrained, a function
                that returns zeros could be provided.

        """

        # Check whether constraint_func() has an appropriate signature
        if callable(constraint_func):
            if len(inspect.signature(constraint_func).parameters) != 1:
                raise ValueError("constraint_func() must accept exactly one"
                                 + " input")
            else:
                # Add constraint_func to the problem
                self.constraints = constraint_func
        else:
            raise TypeError("constraint_func() must be callable")
        return

    def addAcquisition(self, *args):
        """ Add an acquisition function for the surrogate optimizer.

        Args:
            *args (AcquisitionFunction): Acquisition functions that are used
                to scalarize the list of objectives in order to solve the
                surrogate optimization problem.

        """

        # Check for illegal inputs
        if not all([isinstance(arg, AcquisitionFunction) for arg in args]):
            raise TypeError("Args must be instances of AcquisitionFunction")
        # Append all arguments to the acquisitions list
        for arg in args:
            self.acquisitions.append(arg)
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
            if any(self.constraints(xj) > 0.00000001) or \
               np.any(xj[:] < self.lb[:]) or \
               np.any(xj[:] > self.ub[:]):
                raise ValueError("some of starting points (x) are infeasible")
        # Set the batch size
        batch_size = 1000
        # Initialize the database
        o = self.objectives(x[0, :]).size
        p = self.constraints(x[0, :]).size
        data = {'x_vals': np.zeros((batch_size, self.n)),
                'f_vals': np.zeros((batch_size, o)),
                'c_vals': np.zeros((batch_size, p))}
        # Loop over batch size until k == budget
        k = 0
        nondom = {}
        while (k < self.budget):
            # Check how many new points to generate
            k_new = min(self.budget, k + batch_size) - k
            if k_new < batch_size:
                data['x_vals'] = np.zeros((k_new, self.n))
                data['f_vals'] = np.zeros((k_new, o))
                data['c_vals'] = np.zeros((k_new, p))
            # Randomly generate k_new new points
            for i in range(k_new):
                data['x_vals'][i, :] = np.random.random_sample(self.n) \
                                       * (self.ub[:] - self.lb[:]) + self.lb[:]
                data['f_vals'][i, :] = self.objectives(data['x_vals'][i, :])
                data['c_vals'][i, :] = self.constraints(data['x_vals'][i, :])
            # Update the PF
            nondom = updatePF(data, nondom)
            k += k_new
        # Use acquisition functions to extract array of results
        results = []
        for acq in self.acquisitions:
            imin = np.argmin(np.asarray([acq.scalarize(fi)
                                         for fi in nondom['f_vals']]))
            results.append(nondom['x_vals'][imin, :])
        return np.asarray(results)
