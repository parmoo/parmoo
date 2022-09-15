
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


class AcquisitionFunction(ABC):
    """ ABC describing acquisition functions.

    This class contains two methods:
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

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def setTarget(self, data, constraint_func, history):
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

            constraint_func (function): A function whose components evaluate
                to zero if an only if no constraint is violated. If a
                constraint is violated, then constraint_func returns the
                magnitude of the violation.

            history (dict): A persistent dictionary that could be used by
                the implementation of the AcquisitionFunction to pass data
                between iterations.

        Returns:
            numpy.ndarray: A 1d array containing a feasible starting point
            for the scalarized problem.

        """

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def scalarize(self, f_vals):
        """ Scalarize a vector-valued function using the AcquisitionFunction.

        Args:
            f_vals (np.ndarray): A 1D array specifying a vector of function
                values to be scalarized.

        Returns:
            float: The scalarized value.

        """

        raise NotImplementedError("This class method has not been implemented")

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

    This class contains two methods.
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

        raise NotImplementedError("This class method has not been implemented")

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

        raise NotImplementedError("This class method has not been implemented")

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

    This class contains three methods.
     * ``fit(x, f)``
     * ``update(x, f)``
     * ``setCenter(x)``
     * ``evaluate(x)``
     * ``gradient(x)``
     * ``improve(x, global_improv)``
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

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def fit(self, x, f):
        """ Fit a new surrogate to the given data.

        Args:
             x (numpy.ndarray): A 2d array containing the list of
                 design points.

             f (numpy.ndarray): A 2d array containing the corresponding list
                 of objective values.

        """

        raise NotImplementedError("This class method has not been implemented")

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

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def setCenter(self, center):
        """ Set the center for the fit, if this is a local method.

        Args:
            center (numpy.ndarray): A 1d array containing the center for
                this local fit.

        """

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def evaluate(self, x):
        """ Evaluate the surrogate at a design point.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which to the Gaussian RBF should be evaluated.

        Returns:
            numpy.ndarray: A 1d array containing the predicted objective value
            at x.

        """

        raise NotImplementedError("This class method has not been implemented")

    def gradient(self, x):
        """ Evaluate the gradient of the surrogate at a design point.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the gradient of the RBF should be evaluated.

        Returns:
            numpy.ndarray: A 2d array containing the Jacobian matrix of the
            RBF interpolants at x.

        """

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def improve(self, x, global_imrpov):
        """ Suggests a design to evaluate to improve the surrogate near x.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the surrogate should be improved.

            global_improv (Boolean): When True, returns a point for global
                improvement, ignoring the value of x.

        Returns:
            numpy.ndarray: A 2d array containing the list of design points
            that should be evaluated to improve the surrogate.

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


class SurrogateOptimizer(ABC):
    """ ABC describing surrogate optimization techniques.

    This class contains three methods.
     * ``setObjective(obj_func)``
     * ``setGradient(grad_func)``
     * ``setConstraints(constraint_func)``
     * ``addAcquisition(*args)``
     * ``setReset(reset)``
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

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def setObjective(self, obj_func):
        """ Add a vector-valued objective function that will be solved.

        Args:
            obj_func (function): A vector-valued function that can be evaluated
                to solve the surrogate optimization problem.

        """

        raise NotImplementedError("This class method has not been implemented")

    def setGradient(self, grad_func):
        """ Add a matrix-valued gradient function for obj_func.

        Args:
            grad_func (function): A matrix-valued function that can be
                evaluated to obtain the Jacobian matrix for obj_func.

        """

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def setConstraints(self, constraint_func):
        """ Add a constraint function that will be satisfied.

        Args:
            constraint_func (function): A vector-valued function from the
                design space whose components correspond to constraint
                violations. If the problem is unconstrained, a function
                that returns zeros could be provided.

        """

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def addAcquisition(self, *args):
        """ Add an acquisition function for the surrogate optimizer.

        Args:
            args (AcquisitionFunction): Acquisition functions that are used
                to scalarize the list of surrogates in order to solve the
                surrogate optimization problem.

        """

        raise NotImplementedError("This class method has not been implemented")

    @abstractmethod
    def setReset(self, reset):
        """ Add a reset function for resetting surrogate updates.

        Args:
            reset (function): A function with one input, which will be
                called prior to solving the surrogate optimization
                problem with each acquisition function.

        """

        raise NotImplementedError("This class method has not been implemented")

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
