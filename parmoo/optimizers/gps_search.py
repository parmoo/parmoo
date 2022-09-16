
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
import inspect
from parmoo.structs import SurrogateOptimizer, AcquisitionFunction


class LocalGPS(SurrogateOptimizer):
    """ Use Generalized Pattern Search (GPS) to identify local solutions.

    Applies GPS to the surrogate problem, in order to identify design
    points that are locally Pareto optimal, with respect to the surrogate
    problem.

    """

    # Slots for the LocalGPS class
    __slots__ = ['n', 'lb', 'ub', 'acquisitions', 'budget', 'constraints',
                 'objectives']

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
                 * opt_budget (int): The evaluation budget (default: 10,000).

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

        # Do nothing, LocalGPS is gradient free
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
        # Check that x is feasible.
        for xj in x:
            if any(self.constraints(xj) > 0.00000001) or \
               np.any(xj[:] < self.lb[:]) or \
               np.any(xj[:] > self.ub[:]):
                raise ValueError("some of starting points (x) are infeasible")
        # Initialize an empty list of results
        result = []
        # For each acqusisition function
        for j, acquisition in enumerate(self.acquisitions):
            # Reset the mesh dimensions
            mesh = np.diag(self.ub[:] - self.lb[:] * 0.5)
            # Evaluate the starting point
            v = np.asarray(self.objectives(x[j, :]))
            f_min = acquisition.scalarize(v.flatten())
            x_min = x[j, :]
            # Loop over the budget
            for k in range(int(self.budget / (self.n * 2 *
                                              len(self.acquisitions)))):
                # Track whether or not there is improvement
                improve = False
                for i in range(self.n):
                    # Evaluate x + mesh[:, i]
                    x_tmp = x_min + mesh[:, i]
                    if any(x_tmp > self.ub):
                        f_tmp = np.inf
                    elif any(self.constraints(x_tmp) > 0.00000001):
                        f_tmp = np.inf
                    else:
                        v = np.asarray(self.objectives(x_tmp))
                        f_tmp = acquisition.scalarize(v.flatten())
                    # Check for improvement
                    if f_tmp + 10**(-8) < f_min:
                        f_min = f_tmp
                        x_min = x_tmp
                        improve = True
                    # Evaluate x - mesh[:, i]
                    x_tmp = x_min - mesh[:, i]
                    if any(x_tmp < self.lb):
                        f_tmp = np.inf
                    elif any(self.constraints(x_tmp) > 0.00000001):
                        f_tmp = np.inf
                    else:
                        v = np.asarray(self.objectives(x_tmp))
                        f_tmp = acquisition.scalarize(v.flatten())
                    # Check for improvement
                    if f_tmp + 10**(-8) < f_min:
                        f_min = f_tmp
                        x_min = x_tmp
                        improve = True
                # If no improvement, decay the mesh down to the tolerance
                if not improve:
                    if any([mesh[i, i] < 0.0001 for i in range(self.n)]):
                        break
                    else:
                        mesh = mesh * 0.5
            # Append the found minima to the results list
            result.append(x_min)
        return np.asarray(result)


class GlobalGPS(SurrogateOptimizer):
    """ Use randomized search globally followed by GPS locally.

    Use ``RandomSearch`` to globally search the design space (search phase)
    followed by ``LocalGPS`` to refine the potentially efficient solutions
    (poll phase).

    """

    # Slots for the GlobalGPS class
    __slots__ = ['n', 'lb', 'ub', 'acquisitions', 'constraints', 'objectives',
                 'search_budget', 'gps_budget']

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

        from parmoo.util import xerror

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
            budget = 10000
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
            self.gps_budget = int(budget / 2)
        self.search_budget = budget - self.gps_budget
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

        # Do nothing, GlobalGPS is gradient free
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
        # Check that x is feasible.
        for xj in x:
            if any(self.constraints(xj) > 0.00000001) or \
               np.any(xj[:] < self.lb[:]) or \
               np.any(xj[:] > self.ub[:]):
                raise ValueError("some of starting points (x) are infeasible")
        # Do a global search to get global solutions
        gs = RandomSearch(self.n, self.lb, self.ub,
                          {'opt_budget': self.search_budget})
        gs.setObjective(self.objectives)
        gs.setConstraints(self.constraints)
        gs.addAcquisition(*self.acquisitions)
        gs_soln = gs.solve(x)
        gps_budget_loc = int(self.gps_budget / gs_soln.shape[0])
        # Do a local search to refine the global solution
        ls = LocalGPS(self.n, self.lb, self.ub,
                      {'opt_budget': gps_budget_loc})
        ls.setObjective(self.objectives)
        ls.setConstraints(self.constraints)
        ls.addAcquisition(*self.acquisitions)
        ls_soln = ls.solve(gs_soln)
        # Return the list of local solutions
        return ls_soln
