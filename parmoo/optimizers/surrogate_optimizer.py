""" The abstract base class (ABC) for SurrogateOptimizer objects. """

from abc import ABC, abstractmethod
import inspect
from parmoo.acquisitions.acquisition_function import AcquisitionFunction


class SurrogateOptimizer(ABC):
    """ ABC describing surrogate optimization techniques.

    This class contains the following methods.
     * ``setObjective(obj_func)`` (default implementation provided)
     * ``setSimulation(sim_func, sd_func)`` (default implementation provided)
     * ``setConstraints(constraint_func)`` (default implementation provided)
     * ``setPenalty(penaltyFunc, gradFunc)`` (default implementation provided)
     * ``setTrFunc(trFunc)`` (default implementation provided)
     * ``addAcquisition(*args)`` (default implementation provided)
     * ``returnResults(x, fx, sx, sdx)``
     * ``solve(x)``
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the SurrogateOptimizer class.

        Args:
            o (int): The number of objectives.

            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

            hyperparams (dict): A dictionary of hyperparameters for the
                optimization procedure.

        Returns:
            SurrogateOptimizer: A new SurrogateOptimizer object.

        """

    def setObjective(self, obj_func):
        """ Add a vector-valued objective function that will be solved.

        Args:
            obj_func (function): A vector-valued function that can be evaluated
                to solve the surrogate optimization problem.

        """

        # Check whether obj_func() has an appropriate signature
        if callable(obj_func):
            if len(inspect.signature(obj_func).parameters) != 2:
                raise ValueError("obj_func() must accept exactly two inputs")
            else:
                # Add obj_func to the problem
                self.objectives = obj_func
        else:
            raise TypeError("obj_func() must be callable")
        return

    def setSimulation(self, sim_func, sd_func):
        """ Add a vector-valued simulation function, used to calculate objs.

        Args:
            sim_func (function): A vector-valued function that can be evaluated
                to determine the surrogate-predicted simulation outputs.

            sd_func (function): A vector-valued function that can be evaluated
                to determine the standard deviations of the surrogate
                predictions.

        """

        # Check whether sim_func() has an appropriate signature
        if callable(sim_func):
            if len(inspect.signature(sim_func).parameters) != 1:
                raise ValueError("sim_func() must accept exactly one input")
            else:
                # Add sim_func to the problem
                self.simulations = sim_func
        else:
            raise TypeError("sim_func() must be callable")
        # Check whether sd_func() has an appropriate signature
        if callable(sd_func):
            if len(inspect.signature(sd_func).parameters) != 1:
                raise ValueError("sd_func() must accept exactly one input")
            else:
                self.sim_sd = sd_func
        else:
            raise TypeError("sd_func() must be callable")
        return

    def setPenalty(self, penalty_func):
        """ Add a matrix-valued gradient function for obj_func.

        Args:
            penalty_func (function): A vector-valued penalized objective
                that incorporates a penalty for violating constraints.

            grad_func (function): A matrix-valued function that can be
                evaluated to obtain the Jacobian matrix for obj_func.

        """

        # Check whether penalty_func() has an appropriate signature
        if callable(penalty_func):
            if len(inspect.signature(penalty_func).parameters) != 2:
                raise ValueError("penalty_func must accept exactly two inputs")
            else:
                # Add penalty to the problem
                self.penalty_func = penalty_func
        else:
            raise TypeError("penalty_func must be callable")
        return

    def setConstraints(self, constraint_func):
        """ Add a constraint function that will be satisfied.

        Args:
            constraint_func (function): A vector-valued function from the
                design space whose components correspond to constraint
                violations. If the problem has only bound constraints, this
                function returns zeros.

        """

        # Check whether constraint_func() has an appropriate signature
        if callable(constraint_func):
            if len(inspect.signature(constraint_func).parameters) != 2:
                raise ValueError("constraint_func() must accept exactly two"
                                 + " input")
            else:
                # Add constraint_func to the problem
                self.constraints = constraint_func
        else:
            raise TypeError("constraint_func() must be callable")
        return

    def setTrFunc(self, trFunc):
        """ Add a TR setter function for alerting surrogates.

        Args:
            trFunc (function): A function with 2 inputs, which the optimizer
                must call prior to solving each surrogate optimization problem
                in order to set the trust-region center and radius.

        """

        # Check whether trFunc() has an appropriate signature
        if callable(trFunc):
            if len(inspect.signature(trFunc).parameters) != 2:
                raise ValueError("trFunc() must accept exactly 2 inputs")
            else:
                # Add obj_func to the problem
                self.setTR = trFunc
        else:
            raise TypeError("trFunc() must be callable")
        return

    def returnResults(self, x, fx, sx, sdx):
        """ This is a callback function to collect evaluation results.

        Implement this function to receive the results of each
        true simulation evaluation from the MOOP class at runtime.

        Args:
            x (ndarray): A 1D array with the design point evaluated.

            fx (ndarray): A 1D array with the objective function values at x.

            sx (ndarray): The simulation function values at x.

            sdx (ndarray): The standard deviation in the simulation prediction.

        """

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

    @abstractmethod
    def solve(self, x_k):
        """ Solve the surrogate problem.

        You may assume that the following internal attributes are defined
        and contain callable definitions of the objective, constraint,
        penalty, and simulation (surrogate) functions, respectively:
         * ``self.objectives``,
         * ``self.constraints``,
         * ``self.penalty_func``, and
         * ``self.simulations``.

        Additionally, you may assume that:
         * ``self.acquisitions`` contains a list of one or more
           ``AcquisitionFunction`` object instances, each of whose
           ``acq.scalarize(f_vals, x_vals, s_vals_mean, s_vals_sd)``
           is set and ready to call; and
         * ``self.setTR(x, r)`` can be called to set a trust-region
           centered at ``x`` with radius ``r`` (and re-fit the surrogates
           accordingly).

        Note: If implementing your own solver, try to jit (or re-jit) any of
        the objective, constraint, penalty, simulation surrogate, and/or
        acquisition functions after each call to ``self.setTR``.
        Additionally, if provided by the user,
        the objectives, constraints, penalty, and acq.scalarize,
        functions should all be differentiable by importing and
        calling ``jax.jacrev()``.

        Args:
            x_k (ndarray): A 2D array containing a list of current iterates.

        Returns:
            ndarray: A 2D array matching the shape of x_k specifying x_{k+1}.

        """

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the load method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        raise NotImplementedError("This class method has not been implemented")

    def load(self, filename):
        """ Reload important data into this class after a previous save.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the save method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data has been saved.

        """

        raise NotImplementedError("This class method has not been implemented")
