""" The abstract base class (ABC) for AcquisitionFunction objects. """

from abc import ABC, abstractmethod


class AcquisitionFunction(ABC):
    """ ABC describing acquisition functions.

    This class contains the following methods:
     * ``setTarget(data, penalty_func)``
     * ``scalarize(f_vals, x_vals, s_vals_mean, s_vals_sd)``
     * ``useSD()``
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the AcquisitionFunction class.

        Args:
            o (int): The number of objectives.

            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

            hyperparams (dict): A dictionary of hyperparameters that are
                passed to the acquisition function.

        Returns:
            AcquisitionFunction: A new AcquisitionFunction object.

        """

    @abstractmethod
    def setTarget(self, data, penalty_func):
        """ Set a new target value or region for the AcquisitionFunction.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database. It contains two mandatory fields:
                 * 'x_vals' (ndarray): A 2D array of design points.
                 * 'f_vals' (ndarray): A 2D array of corresponding objectives.

            penalty_func (function): A function of one (x) or two (x, sx)
                inputs that evaluates all (penalized) objective scores.

        Returns:
            ndarray: A 1D array containing a feasible starting point
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

        Note: For best performance, make sure that jax can jit this method.

        Additionally, for compatibility with gradient-based solvers,
        this method must be implemented in jax and be differentiable
        via the jax.jacrev() tool.

        Args:
            f_vals (ndarray): A 1D array specifying a vector of function
                values to be scalarized.

            x_vals (ndarray): A 1D array specifying a vector the design
                point corresponding to f_vals.

            s_vals_mean (ndarray): A 1D array specifying the expected value
                of the simulation outputs for the x value being scalarized.

            s_vals_sd (ndarray): A 1D array specifying the standard deviation
                for each of the simulation outputs.

        Returns:
            float: The scalarized value.

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
