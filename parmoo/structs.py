
""" Abstract base classes (ABCs) for ParMOO project.

This module contains several abstract base classes that can be used
to create a flexible framework for surrogate based multiobjective
optimization.

The classes include:
 * AcquisitionFunction
 * GlobalSearch
 * SurrogateFunction
 * SurrogateOptimizer

"""

from abc import ABC, abstractmethod
import inspect
import numpy as np
from scipy.stats import tstd


class AcquisitionFunction(ABC):
    """ ABC describing acquisition functions.

    This class contains the following methods:
     * ``useSD()``
     * ``setTarget(data, constraint_func, history)``
     * ``scalarize(f_vals)``
     * ``scalarizeGrad(f_vals, g_vals)``
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the AcquisitionFunction class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                space.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                space.

            hyperparams (dict): A dictionary of hyperparameters that are
                passed to the acquisition function.

        Returns:
            AcquisitionFunction: A new AcquisitionFunction object.

        """

    @abstractmethod
    def setTarget(self, data, penalty_func, history):
        """ Set a new target value or region for the AcquisitionFunction.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database. It contains two mandatory fields:
                 * 'x_vals' (numpy.ndarray): A 2d array containing the
                   list of design points.
                 * 'f_vals' (numpy.ndarray): A 2d array containing the
                   corresponding list of objective values.

            If gradients are available, data may contain one additional
                field:
                 * 'g_vals' (numpy.ndarray): A 3d array containing the
                   Jacobian of the objective function at each
                   point in 'x_vals'.

            penalty_func (function): A function of one (x) or two (x, sx)
                inputs that evaluates all (penalized) objective scores.

            history (dict): A persistent dictionary that could be used by
                the implementation of the AcquisitionFunction to pass data
                between iterations.

        Returns:
            numpy.ndarray: A 1d array containing a feasible starting point
            for the scalarized problem.

        """

    def useSD(self):
        """ Query whether this method uses uncertainties.

        When False, allows users to shortcut expensive uncertainty
        computations.

        Default implementation returns True, requiring full uncertainty
        computation for applicable models.

        """

        return True

    @abstractmethod
    def scalarize(self, f_vals, x_vals, s_vals_mean, s_vals_sd):
        """ Scalarize a vector-valued function using the AcquisitionFunction.

        Args:
            f_vals (np.ndarray): A 1D array specifying a vector of function
                values to be scalarized.

            x_vals (np.ndarray): A 1D array specifying a vector the design
                point corresponding to f_vals.

            s_vals_mean (np.ndarray): A 1D array specifying the expected
                simulation outputs for the x value being scalarized.

            s_vals_sd (np.ndarray): A 1D array specifying the standard
                deviation for each of the simulation outputs.

        Returns:
            float: The scalarized value.

        """

    def scalarizeGrad(self, f_vals, g_vals):
        """ Scalarize a Jacobian of gradients using the current weights.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values for the scalarized gradient.

            g_vals (numpy.ndarray): A 2d array specifying the gradient
                values to be scalarized.

        Returns:
            np.ndarray: The 1d array for the scalarized gradient.

        """

        raise NotImplementedError("This class method has not been implemented")

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


class GlobalSearch(ABC):
    """ ABC describing global search techniques.

    This class contains the following methods.
     * ``startSearch(lb, ub)``
     * ``resumeSearch()``
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the GlobalSearch class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                space.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                space.

            hyperparams (dict): A dictionary of hyperparameters for the
                global search. It may contain any inputs specific to the
                search algorithm.

        Returns:
            GlobalSearch: A new GlobalSearch object.

        """

    @abstractmethod
    def startSearch(self, lb, ub):
        """ Begin a new global search.

        Args:
            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The dimension must match n.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match n.

        Returns:
            numpy.ndarray: A 2d array, containing the list of design points
            to be evaluated.

        """

    def resumeSearch(self):
        """ Resume a global search.

        Returns:
            numpy.ndarray: A 2d array, containing the list of design points
            to be evaluated.

        """

        raise NotImplementedError("This class method has not been implemented")

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


class SurrogateFunction(ABC):
    """ ABC describing surrogate functions.

    This class contains the following methods.
     * ``fit(x, f)``
     * ``update(x, f)``
     * ``setTrustRegion(center, radius)`` (default implementation provided)
     * ``evaluate(x)``
     * ``gradient(x)``
     * ``stdDev(x)``
     * ``stdDevGrad(x)``
     * ``improve(x, global_improv)`` (default implementation provided)
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, m, lb, ub, hyperparams):
        """ Constructor for the SurrogateFunction class.

        Args:
            m (int): The number of objectives to fit.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters to be used
                by the surrogate models, including:
                 * des_tols (numpy.ndarray, optional): A 1d array whose length
                   matches lb and ub. Each entry is a number (greater than 0)
                   specifying the design space tolerance for that variable.
                   By default, des_tols = [1.0e-8, ..., 1.0e-8].


        Returns:
            SurrogateFunction: A new SurrogateFunction object.

        """

    @abstractmethod
    def fit(self, x, f):
        """ Fit a new surrogate to the given data.

        Args:
             x (numpy.ndarray): A 2d array containing the list of
                 design points.

             f (numpy.ndarray): A 2d array containing the corresponding list
                 of objective values.

        """

    @abstractmethod
    def update(self, x, f):
        """ Update an existing surrogate model using new data.

        Args:
             x (numpy.ndarray): A 2d array containing the list of
                 new design points, with which to update the surrogate
                 models.

             f (numpy.ndarray): A 2d array containing the corresponding list
                 of objective values.

        """

    def setTrustRegion(self, center, radius):
        """ Alert the surrogate of the trust region center and radius.

        Default implementation does nothing, which would be the case for a
        global surrogate model.

        Args:
            center (numpy.ndarray): A 1d array containing the center for
                this local fit.

            radius (np.ndarray or float): The trust-region radius.

        """

        return

    @abstractmethod
    def evaluate(self, x):
        """ Evaluate the surrogate at a design point.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which to the surrogate should be evaluated.

        Returns:
            numpy.ndarray: A 1d array containing the predicted objective value
            at x.

        """

    def gradient(self, x):
        """ Evaluate the gradient of the surrogate at a design point.

        Note: this method need not be implemented when using a derivative
        free SurrogateOptimization solver.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the gradient of the surrogate should be evaluated.

        Returns:
            numpy.ndarray: A 2d array containing the Jacobian matrix of the
            surrogate at x.

        """

        raise NotImplementedError("This class method has not been implemented")

    def stdDev(self, x):
        """ Evaluate the standard deviation (uncertainty) of the surrogate at x.

        Note: this method need not be implemented when the acquisition
        function does not use the model uncertainty.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the standard deviation should be evaluated.

        Returns:
            numpy.ndarray: A 1d array containing the standard deviation at x.

        """

        raise NotImplementedError("This class method has not been implemented")

    def stdDevGrad(self, x):
        """ Evaluate the gradient of the standard deviation at x.

        Note: this method need not be implemented when the acquisition
        function does not use both the model uncertainty and gradient.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the gradient of standard deviation should be evaluated.

        Returns:
            numpy.ndarray: A 2d array containing the Jacobian matrix of the
            standard deviation at x.

        """

        raise NotImplementedError("This class method has not been implemented")

    def improve(self, x, global_improv):
        """ Suggests a design to evaluate to improve the surrogate near x.

        A default implementation is given based on random sampling.
        Re-implement the improve method to overwright the default
        policy.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the surrogate should be improved.

            global_improv (Boolean): When True, returns a point for global
                improvement, ignoring the value of x.

        Returns:
            numpy.ndarray: A 2d array containing the list of design points
            that should be evaluated to improve the surrogate.

        """

        # Check that the x is legal
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array")
        else:
            if x.size != self.n:
                raise ValueError("x must have length n")
            elif (np.any(x < self.lb - self.eps) or
                  np.any(x > self.ub + self.eps)):
                raise ValueError("x cannot be infeasible")
        # Allocate the output array.
        x_new = np.zeros(self.n)
        if global_improv:
            # If global improvement has been specified, randomly select a
            # point from within the bound constraints.
            x_new[:] = self.lb[:] + (np.random.random(self.n)
                                     * (self.ub[:] - self.lb[:]))
            while any([np.all(np.abs(x_new - xj) < self.eps)
                       for xj in self.x_vals]):
                x_new[:] = self.lb[:] + (np.random.random(self.n)
                                         * (self.ub[:] - self.lb[:]))
        else:
            # Find the n+1 closest points to x in the current database
            diffs = np.asarray([np.abs(x - xj) / self.eps
                                for xj in self.x_vals])
            dists = np.asarray([np.amax(dj) for dj in diffs])
            inds = np.argsort(dists)
            diffs = diffs[inds]
            if dists[inds[self.n]] > 1.5:
                # Calculate the normalized sample standard dev along each axis
                stddev = np.asarray(tstd(diffs[:self.n+1], axis=0))
                stddev[:] = np.maximum(stddev, np.ones(self.n))
                stddev[:] = stddev[:] / np.amin(stddev)
                # Sample within B(x, dists[inds[self.n]] / stddev)
                rad = (dists[inds[self.n]] * self.eps) / stddev
                x_new = np.fmin(np.fmax(2.0 * (np.random.random(self.n) - 0.5)
                                        * rad[:] + x, self.lb), self.ub)
                while any([np.all(np.abs(x_new - xj) < self.eps)
                           for xj in self.x_vals]):
                    x_new = np.fmin(np.fmax(2.0 *
                                            (np.random.random(self.n) - 0.5)
                                            * rad[:] + x, self.lb), self.ub)
            else:
                # If the n+1st nearest point is too close, use global_improv.
                x_new[:] = self.lb[:] + np.random.random(self.n) \
                           * (self.ub[:] - self.lb[:])
                # If the nearest point is too close, resample.
                while any([np.all(np.abs(x_new - xj) < self.eps)
                           for xj in self.x_vals]):
                    x_new[:] = self.lb[:] + (np.random.random(self.n)
                                             * (self.ub[:] - self.lb[:]))
        # Return the point to be sampled in a 2d array.
        return np.asarray([x_new])

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

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                space.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                space.

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
            if len(inspect.signature(obj_func).parameters) != 1:
                raise ValueError("obj_func() must accept exactly one input")
            else:
                # Add obj_func to the problem
                self.objectives = obj_func
        else:
            raise TypeError("obj_func() must be callable")
        return

    def setSimulation(self, sim_func, sd_func=None):
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
        if sd_func is not None and callable(sd_func):
            if len(inspect.signature(sd_func).parameters) not in (1, 2):
                raise ValueError("sd_func() must accept one or 2 inputs")
            else:
                # Add sd_func to the problem
                self.sim_sd = sd_func
        else:
            raise TypeError("sd_func() must be callable")
        return

    def setPenalty(self, penalty_func, grad_func):
        """ Add a matrix-valued gradient function for obj_func.

        Args:
            penalty_func (function): A vector-valued penalized objective
                that incorporates a penalty for violating constraints.

            grad_func (function): A matrix-valued function that can be
                evaluated to obtain the Jacobian matrix for obj_func.

        """

        # Check whether grad_func() has an appropriate signature
        if callable(grad_func):
            if len(inspect.signature(grad_func).parameters) != 1:
                raise ValueError("grad_func() must accept exactly one input")
            else:
                # Add grad_func to the problem
                self.gradients = grad_func
        else:
            raise TypeError("grad_func() must be callable")
        # Check whether penalty_func() has an appropriate signature
        if callable(penalty_func):
            if len(inspect.signature(penalty_func).parameters) not in [1, 2]:
                raise ValueError("penalty_func must accept exactly one input")
            else:
                # Add Lagrangian to the problem
                self.penalty_func = penalty_func
        else:
            raise TypeError("penalty_func must be callable")
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

    def setTrFunc(self, trFunc):
        """ Add a TR setter function for alerting surrogates.

        Args:
            trFunc (function): A function with 2 inputs, which will be
                called prior to solving the surrogate optimization
                problem with each acquisition function in order to set
                the surrogate trust region center and radius.

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
        """ Collect the results of a function evaluation.

        Args:
            x (np.ndarray): The design point evaluated.

            fx (np.ndarray): The objective function values at x.

            sx (np.ndarray): The simulation function values at x.

            sdx (np.ndarray): The standard deviation in the simulation
                outputs at x.

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
    def solve(self, x):
        """ Solve the surrogate problem.

        Args:
            x (numpy.ndarray): A 2d array containing a list of feasible
                design points used to warm start the search.

        Returns:
            float: A 2d numpy.ndarray of potentially efficient design points
            that were found by the surrogate optimizer.

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
