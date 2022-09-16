
""" Implementations of the epsilon-constraint-style scalarizations.

This module contains implementations of the AcquisitionFunction ABC, which
use the epsilon constraint method.

The classes include:
 * ``RandomConstraint`` (randomly set a ub for all but 1 objective)

"""

import numpy as np
import inspect
from parmoo.structs import AcquisitionFunction


class RandomConstraint(AcquisitionFunction):
    """ Improve upon a randomly set target point.

    Randomly sets a target point inside the current Pareto front.
    Attempts to improve one of the objective values by reformulating
    all other objectives as constraints, upper bounded by their target
    value.

    """

    # Slots for the RandomConstraint class
    __slots__ = ['n', 'o', 'lb', 'ub', 'f_ub', 'weights']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the RandomConstraint class.

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
            RandomConstraint: A new RandomConstraint scalarizer.

        """

        from parmoo.util import xerror

        # Check inputs
        xerror(o=o, lb=lb, ub=ub, hyperparams=hyperparams)
        self.o = o
        # Set the design variable count
        self.n = np.size(lb)
        # Initialize the objective/design bounds
        self.f_ub = np.zeros(self.o)
        self.weights = np.zeros(self.o)
        self.ub = ub
        self.lb = lb
        return

    def setTarget(self, data, lagrange_func, history):
        """ Randomly generate a target based on current nondominated points.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database. It contains two mandatory fields:
                 * 'x_vals' (numpy.ndarray): A 2d array containing the
                   list of design points.
                 * 'f_vals' (numpy.ndarray): A 2d array containing the
                   corresponding list of objective values.

            lagrange_func (function): A function whose components correspond
                to constraint violation amounts.

            history (dict): A persistent dictionary that could be used by
                the implementation of the AcquisitionFunction to pass data
                between iterations; also unused by this scheme.

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
        # If pf is empty, randomly select weights and starting point
        if pf['x_vals'].shape[0] == 0:
            self.f_ub[:] = np.inf
            self.weights = -np.log(1.0 - np.random.random_sample(self.o))
            self.weights = self.weights[:] / sum(self.weights[:])
            # Randomly select a feasible starting point
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
            # Randomly select pts in the convex hull of the nondominate pts
            ipts = np.random.randint(0, pf['f_vals'].shape[0], size=self.o)
            self.weights = -np.log(1.0 - np.random.random_sample(self.o))
            self.weights[:] = self.weights[:] / np.linalg.norm(self.weights)
            target = np.dot(self.weights, pf['f_vals'][ipts, :])
            fi = pf['f_vals'][ipts[0], :]
            # Set the bounds
            self.f_ub[:] = np.inf
            self.weights[:] = 0.0001
            for j in range(self.o):
                # If fi[j] is less than target[j], this is a bound
                if fi[j] + 0.00000001 < target[j]:
                    self.f_ub[j] = target[j]
                # Otherwise, it is an objective
                else:
                    self.weights[j] = 1.0
            # Normalize the weights
            self.weights[:] = self.weights[:] / np.linalg.norm(self.weights)
            # The corresponding x_val is feasible by construction
            return pf['x_vals'][ipts[0], :]

    def scalarize(self, f_vals):
        """ Scalarize a vector of function values using the current bounds.

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
        # Return the weighted sum of objectives, if the bounds are satisfied
        result = np.dot(f_vals, self.weights)
        for i in range(self.o):
            if f_vals[i] > self.f_ub[i]:
                result = result + 10.0 * (f_vals[i] - self.f_ub[i])
        return result

    def scalarizeGrad(self, f_vals, g_vals):
        """ Scalarize a Jacobian of gradients using the current bounds.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values for the scalarized gradient, which are used to
                penalize exceeding the bounds.

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
        result = np.dot(np.transpose(g_vals), self.weights)
        # Add the gradient of the penalty for any bound violations
        for i in range(self.o):
            if f_vals[i] > self.f_ub[i]:
                result = result + 10.0 * g_vals[i]
        return result
