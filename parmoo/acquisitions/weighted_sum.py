
""" Implementations of the weighted-sum scalarization technique.

This module contains implementations of the AcquisitionFunction ABC, which
use the weighted-sum technique.

The classes include:
 * ``UniformWeights`` (sample convex weights from a uniform distribution)
 * ``FixedWeights`` (uses a fixed scalarization, which can be set upon init)

"""

import numpy as np
import inspect
from parmoo.structs import AcquisitionFunction
from parmoo.util import xerror


class UniformWeights(AcquisitionFunction):
    """ Randomly generate scalarizing weights.

    Generates uniformly distributed scalarization weights, by randomly
    sampling the probability simplex.

    """

    # Slots for the UniformWeights class
    __slots__ = ['n', 'o', 'lb', 'ub', 'weights']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the UniformWeights class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for tuning
                the acquisition function.

        Returns:
            UniformWeights: A new UniformWeights generator.

        """

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
        # Set the objective count
        self.o = o
        # Set the design variable count
        self.n = np.size(lb)
        # Set the bound constraints
        self.lb = lb
        self.ub = ub
        # Initialize the weights array
        self.weights = np.zeros(o)
        return

    def setTarget(self, data, lagrange_func, history):
        """ Randomly generate a new vector of scalarizing weights.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database.

            lagrange_func (function): A function whose components correspond
                to constraint violation amounts.

            history (dict): Another unused argument for this function.

        Returns:
            numpy.ndarray: A 1d array containing the 'best' feasible starting
            point for the scalarized problem (if any previous evaluations
            were feasible) or the point in the existing database that is
            most nearly feasible.

        """

        from parmoo.util import updatePF

        # Check whether any data was given
        no_data = False
        # Check for illegal input from data
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        else:
            if ('x_vals' in data) != ('f_vals' in data):
                raise AttributeError("if x_vals is a key in data, then " +
                                     "f_vals must also appear")
            elif 'x_vals' in data:
                if data['x_vals'] is not None and data['f_vals'] is not None:
                    if data['x_vals'].shape[0] != data['f_vals'].shape[0]:
                        raise ValueError("x_vals and f_vals must be equal " +
                                         "length")
                    if data['x_vals'].shape[1] != self.n:
                        raise ValueError("The rows of x_vals must have " +
                                         "length n")
                    if data['f_vals'].shape[1] != self.o:
                        raise ValueError("The rows of f_vals must have " +
                                         "length o")
                else:
                    no_data = True
            else:
                no_data = True
        # Check whether lagrange_func() has an appropriate signature
        if callable(lagrange_func):
            if len(inspect.signature(lagrange_func).parameters) != 1:
                raise ValueError("lagrange_func() must accept exactly one"
                                 + " input")
        else:
            raise TypeError("lagrange_func() must be callable")
        if no_data:
            # If data is empty, then the Pareto front is empty
            pf = {'x_vals': np.zeros((0, self.n)),
                  'f_vals': np.zeros((0, self.o)),
                  'c_vals': np.zeros((0, 1))}
        else:
            # Get the Pareto front
            pf = updatePF(data, {})
        # Sample the weights uniformly from the unit simplex
        self.weights = -np.log(1.0 - np.random.random_sample(self.o))
        self.weights = self.weights[:] / sum(self.weights[:])
        # If pf is empty, randomly select the starting point
        if pf['x_vals'].shape[0] == 0:
            # Randomly search for a good starting point
            x_min = np.random.random_sample(self.n) * (self.ub - self.lb) \
                    + self.lb
            for count in range(1000):
                x = np.random.random_sample(self.n) * (self.ub - self.lb) \
                    + self.lb
                if np.dot(self.weights, lagrange_func(x)) \
                   < np.dot(self.weights, lagrange_func(x_min)):
                    x_min[:] = x[:]
            return x_min
        else:
            i = np.argmin(np.asarray([np.dot(self.weights, fi)
                                      for fi in pf['f_vals']]))
            x = pf['x_vals'][i, :]
            return x

    def scalarize(self, f_vals):
        """ Scalarize a vector of function values using the current weights.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values to be scalarized.

        Returns:
            float: The scalarized value.

        """

        # Check that the function values are legal
        if isinstance(f_vals, np.ndarray):
            if self.o != np.size(f_vals):
                raise ValueError("f_vals must have length o")
        else:
            raise TypeError("f_vals must be a numpy array")
        # Compute the dot product between the weights and function values
        return np.dot(f_vals, self.weights)

    def scalarizeGrad(self, f_vals, g_vals):
        """ Scalarize a Jacobian of gradients using the current weights.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values for the scalarized gradient (not used here).

            g_vals (numpy.ndarray): A 2d array specifying the gradient
                values to be scalarized.

        Returns:
            np.ndarray: The 1d array for the scalarized gradient.

        """

        # Check that the function values are legal
        if isinstance(f_vals, np.ndarray):
            if self.o != np.size(f_vals):
                raise ValueError("f_vals must have length o")
        else:
            raise TypeError("f_vals must be a numpy array")
        # Check that the gradient values are legal
        if isinstance(g_vals, np.ndarray):
            if self.o != g_vals.shape[0] or self.n != g_vals.shape[1]:
                raise ValueError("g_vals must have shape o-by-n")
        else:
            raise TypeError("g_vals must be a numpy array")
        # Compute the dot product between the weights and the gradient values
        return np.dot(np.transpose(g_vals), self.weights)


class FixedWeights(AcquisitionFunction):
    """ Use fixed scalarizing weights.

    Use a fixed scalarization scheme, based on a fixed weighted sum.

    """

    # Slots for the FixedWeights class
    __slots__ = ['n', 'o', 'lb', 'ub', 'weights']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the FixedWeights class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for tuning
                the acquisition function. May contain the following key:
                 * 'weights' (numpy.ndarray): A 1d array of length o that,
                   when present, specifies the scalarization weights to use.
                   When absent, the default weights are w = [1/o, ..., 1/o].

        Returns:
            FixedWeights: A new FixedWeights generator.

        """

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
        # Set the objective count
        self.o = o
        # Set the design variable count
        self.n = np.size(lb)
        # Set the bound constraints
        self.lb = lb
        self.ub = ub
        # Check the hyperparams dictionary for weights
        if 'weights' in hyperparams:
            # If weights are provided, check that they are legal
            if not isinstance(hyperparams['weights'], np.ndarray):
                raise TypeError("when present, 'weights' must be a " +
                                 "numpy array")
            else:
                if hyperparams['weights'].size != self.o:
                    raise ValueError("when present, 'weights' must " +
                                     "have length o")
                else:
                    # Assign the weights
                    self.weights = hyperparams['weights'].flatten()
        else:
            # If no weights were provided, use an even weighting
            self.weights = np.ones(self.o) / float(self.o)
        return

    def setTarget(self, data, lagrange_func, history):
        """ Randomly generate a feasible starting point.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database.

            lagrange_func (function): A function whose components correspond
                to constraint violation amounts.

            history (dict): Another unused argument for this function.

        Returns:
            numpy.ndarray: A 1d array containing the 'best' feasible starting
            point for the scalarized problem (if any previous evaluations
            were feasible) or the point in the existing database that is
            most nearly feasible.

        """

        from parmoo.util import updatePF

        # Check whether any data was given
        no_data = False
        # Check for illegal input from data
        if not isinstance(data, dict):
            raise TypeError("data must be a dict")
        else:
            if ('x_vals' in data) != ('f_vals' in data):
                raise AttributeError("if x_vals is a key in data, then " +
                                     "f_vals must also appear")
            elif 'x_vals' in data:
                if data['x_vals'] is not None and data['f_vals'] is not None:
                    if data['x_vals'].shape[0] != data['f_vals'].shape[0]:
                        raise ValueError("x_vals and f_vals must be equal " +
                                         "length")
                    if data['x_vals'].shape[1] != self.n:
                        raise ValueError("The rows of x_vals must have " +
                                         "length n")
                    if data['f_vals'].shape[1] != self.o:
                        raise ValueError("The rows of f_vals must have " +
                                         "length o")
                else:
                    no_data = True
            else:
                no_data = True
        # Check whether lagrange_func() has an appropriate signature
        if callable(lagrange_func):
            if len(inspect.signature(lagrange_func).parameters) != 1:
                raise ValueError("lagrange_func() must accept exactly one"
                                 + " input")
        else:
            raise TypeError("lagrange_func() must be callable")
        if no_data:
            # If data is empty, then the Pareto front is empty
            pf = {'x_vals': np.zeros((0, self.n)),
                  'f_vals': np.zeros((0, self.o)),
                  'c_vals': np.zeros((0, 1))}
        else:
            # Get the Pareto front
            pf = updatePF(data, {})
        # If pf is empty, randomly select the starting point
        if pf['x_vals'].shape[0] == 0:
            # Randomly search for a good starting point
            x_min = np.random.random_sample(self.n) * (self.ub - self.lb) \
                    + self.lb
            for count in range(1000):
                x = np.random.random_sample(self.n) * (self.ub - self.lb) \
                    + self.lb
                if np.dot(self.weights, lagrange_func(x)) \
                   < np.dot(self.weights, lagrange_func(x_min)):
                    x_min[:] = x[:]
            return x_min
        else:
            i = np.argmin(np.asarray([np.dot(self.weights, fi)
                                      for fi in pf['f_vals']]))
            x = pf['x_vals'][i, :]
            return x

    def scalarize(self, f_vals):
        """ Scalarize a vector of function values using the current weights.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values to be scalarized.

        Returns:
            float: The scalarized value.

        """

        # Check that the function values are legal
        if isinstance(f_vals, np.ndarray):
            if self.o != np.size(f_vals):
                raise ValueError("f_vals must have length o")
        else:
            raise TypeError("f_vals must be a numpy array")
        # Compute the dot product between the weights and function values
        return np.dot(f_vals, self.weights)

    def scalarizeGrad(self, f_vals, g_vals):
        """ Scalarize a Jacobian of gradients using the current weights.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values for the scalarized gradient (not used here).

            g_vals (numpy.ndarray): A 2d array specifying the gradient
                values to be scalarized.

        Returns:
            np.ndarray: The 1d array for the scalarized gradient.

        """

        # Check that the function values are legal
        if isinstance(f_vals, np.ndarray):
            if self.o != np.size(f_vals):
                raise ValueError("f_vals must have length o")
        else:
            raise TypeError("f_vals must be a numpy array")
        # Check that the gradient values are legal
        if isinstance(g_vals, np.ndarray):
            if self.o != g_vals.shape[0] or self.n != g_vals.shape[1]:
                raise ValueError("g_vals must have shape o-by-n")
        else:
            raise TypeError("g_vals must be a numpy array")
        # Compute the dot product between the weights and the gradient values
        return np.dot(np.transpose(g_vals), self.weights)
