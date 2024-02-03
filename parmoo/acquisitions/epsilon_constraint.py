
""" Implementations of the epsilon-constraint-style scalarizations.

This module contains implementations of the AcquisitionFunction ABC, which
use the epsilon constraint method.

The classes include:
 * ``RandomConstraint`` (randomly set a ub for all but 1 objective)

"""

import numpy as np
from scipy import stats, integrate
import inspect
from parmoo.structs import AcquisitionFunction
from parmoo.util import xerror


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

    def useSD(self):
        """ Querry whether this method uses uncertainties.

        When False, allows users to shortcut expensive uncertainty
        computations.

        """

        return False

    def setTarget(self, data, penalty_func):
        """ Randomly generate a target based on current nondominated points.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database. It contains two mandatory fields:
                 * 'x_vals' (numpy.ndarray): A 2d array containing the
                   list of design points.
                 * 'f_vals' (numpy.ndarray): A 2d array containing the
                   corresponding list of objective values.

            penalty_func (function): A function of one (x) or two (x, sx)
                inputs that evaluates the (penalized) objectives.

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
        # Check whether penalty_func() has an appropriate signature
        if callable(penalty_func):
            if len(inspect.signature(penalty_func).parameters) not in [1, 2]:
                raise ValueError("penalty_func() must accept 1 or 2"
                                 + " inputs")
        else:
            raise TypeError("penalty_func() must be callable")
        if no_data:
            # If data is empty, then the Pareto front is empty
            pf = {'x_vals': np.zeros((0, self.n)),
                  'f_vals': np.zeros((0, self.o)),
                  'c_vals': np.zeros((0, 1))}
        else:
            # Get the Pareto front
            pf = updatePF(data, {})
        # If data is empty, randomly select weights and starting point
        if no_data:
            self.weights = -np.log(1.0 - np.random.random_sample(self.o))
            self.weights[:] = self.weights[:] / np.linalg.norm(self.weights)
            self.f_ub[:] = np.inf
            # Randomly select a starting point
            x_start = (np.random.random_sample(self.n) * (self.ub - self.lb)
                       + self.lb)
            return x_start
        # If data is nonempty but pf is empty, use a penalty to select
        elif pf is None or pf['x_vals'].shape[0] == 0:
            self.weights = -np.log(1.0 - np.random.random_sample(self.o))
            self.weights[:] = self.weights[:] / np.linalg.norm(self.weights)
            self.f_ub[:] = np.inf
            # Check for "most feasible" starting x
            x_best = np.zeros(data['x_vals'].shape[1])
            p_best = np.infty
            for xi, fi, ci in zip(data['x_vals'], data['f_vals'],
                                  data['c_vals']):
                p_temp = np.sum(fi) / 1.0e-8 + np.sum(ci)
                if p_temp < p_best:
                    x_best = xi
                    p_best = p_temp
            return x_best
        else:
            # Randomly select pts in the convex hull of the nondominate pts
            ipts = np.random.randint(0, pf['f_vals'].shape[0], size=self.o)
            self.weights = -np.log(1.0 - np.random.random_sample(self.o))
            self.weights[:] = self.weights[:] / np.linalg.norm(self.weights)
            target = np.dot(self.weights, pf['f_vals'][ipts, :])
            fi = pf['f_vals'][ipts[0], :]
            # Set the bounds
            self.f_ub[:] = np.inf
            self.weights[:] = 1.0e-4
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

    def scalarize(self, f_vals, x_vals, s_vals_mean, s_vals_sd):
        """ Scalarize a vector of function values using the current bounds.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values to be scalarized.

            x_vals (np.ndarray): A 1D array specifying a vector the design
                point corresponding to f_vals (unused by this method).

            s_vals_mean (np.ndarray): A 1D array specifying the expected
                simulation outputs for the x value being scalarized
                (unused by this method).

            s_vals_sd (np.ndarray): A 1D array specifying the standard
                deviation for each of the simulation outputs (unused by
                this method).

        Returns:
            float: The scalarized value.

        """

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

        # Compute the dot product between the weights and the gradient values
        result = np.dot(np.transpose(g_vals), self.weights)
        # Add the gradient of the penalty for any bound violations
        for i in range(self.o):
            if f_vals[i] > self.f_ub[i]:
                result = result + 10.0 * g_vals[i]
        return result


class EI_RandomConstraint(AcquisitionFunction):
    """ Expected improvement of a randomly set target point.

    Randomly sets a target point inside the current Pareto front.
    Attempts to improve one of the objective values by reformulating
    all other objectives as constraints, upper bounded by their target
    value. Uses surrogate uncertainties to maximize expected improvement
    in the target objective subject to constraints.

    """

    # Slots for the RandomConstraint class
    __slots__ = ['n', 'o', 'lb', 'ub', 'f_ub', 'weights', 'best', 'f']

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
                the acquisition function. Including
                 * mc_sample_size (int): The number of samples to use for
                   monte carlo integration (defaults to 10 * m ** 2).

        Returns:
            RandomConstraint: A new RandomConstraint scalarizer.

        """

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
        if 'mc_sample_size' in hyperparams.keys():
            self.sample_size = hyperparams['mc_sample_size']
        else:
            self.sample_size = None
        return

    def useSD(self):
        """ Querry whether this method uses uncertainties.

        When False, allows users to shortcut expensive uncertainty
        computations.

        """

        return True

    def setTarget(self, data, penalty_func):
        """ Randomly generate a target based on current nondominated points.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database. It contains two mandatory fields:
                 * 'x_vals' (numpy.ndarray): A 2d array containing the
                   list of design points.
                 * 'f_vals' (numpy.ndarray): A 2d array containing the
                   corresponding list of objective values.

            penalty_func (function): A function of one (x) or two (x, sx)
                inputs that evaluates the (penalized) objectives.

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
        # Check whether penalty_func() has an appropriate signature
        if callable(penalty_func):
            if len(inspect.signature(penalty_func).parameters) not in [1, 2]:
                raise ValueError("penalty_func() must accept 1 or 2"
                                 + " inputs")
        else:
            raise TypeError("penalty_func() must be callable")
        # Save the penalty function for later
        self.f = penalty_func
        if no_data:
            # If data is empty, then the Pareto front is empty
            pf = {'x_vals': np.zeros((0, self.n)),
                  'f_vals': np.zeros((0, self.o)),
                  'c_vals': np.zeros((0, 1))}
        else:
            # Get the Pareto front
            pf = updatePF(data, {})
        # If data is empty, randomly select weights and starting point
        if no_data:
            self.weights = -np.log(1.0 - np.random.random_sample(self.o))
            self.weights[:] = self.weights[:] / np.linalg.norm(self.weights)
            self.f_ub[:] = np.inf
            # Randomly select a starting point
            x_start = (np.random.random_sample(self.n) * (self.ub - self.lb)
                       + self.lb)
            return x_start
        # If data is nonempty but pf is empty, use a penalty to select
        elif pf is None or pf['x_vals'].shape[0] == 0:
            self.weights = -np.log(1.0 - np.random.random_sample(self.o))
            self.weights[:] = self.weights[:] / np.linalg.norm(self.weights)
            self.f_ub[:] = np.inf
            # Check for "most feasible" starting x
            x_best = np.zeros(data['x_vals'].shape[1])
            p_best = np.infty
            for xi, fi, ci in zip(data['x_vals'], data['f_vals'],
                                  data['c_vals']):
                p_temp = np.sum(fi) / 1.0e-8 + np.sum(ci)
                if p_temp < p_best:
                    x_best = xi
                    p_best = p_temp
            return x_best
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
            self.best = np.dot(pf['f_vals'][ipts[0], :], self.weights)
            return pf['x_vals'][ipts[0], :]

    def scalarize(self, f_vals, x_vals, s_vals_mean, s_vals_sd):
        """ Scalarize a vector of function values using the current bounds.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values to be scalarized.

            x_vals (np.ndarray): A 1D array specifying a vector the design
                point corresponding to f_vals (unused by this method).

            s_vals_mean (np.ndarray): A 1D array specifying the expected
                simulation outputs for the x value being scalarized
                (unused by this method).

            s_vals_sd (np.ndarray): A 1D array specifying the standard
                deviation for each of the simulation outputs (unused by
                this method).

        Returns:
            float: The scalarized value.

        """

        # If the feasible set was empty, just use the given fi
        if self.best is None:
            result = np.dot(f_vals, self.weights)
            for i in range(self.o):
                if f_vals[i] > self.f_ub[i]:
                    result = result + 10.0 * (f_vals[i] - self.f_ub[i])
            return result
        # If the feasible set was nonempty and the number of sim outs is 1,
        # then calculate the EI over the best seen feasible fi for the
        # current penalty function
        elif s_vals_mean.size == 1:
            # Construct the distribution for sampling
            s_cov = stats.Covariance.from_diagonal(s_vals_sd**2)
            s_dist = stats.multivariate_normal(mean=s_vals_mean, cov=s_cov)

            def weighted_f(sx):
                """ Calculates the pdf-weighted value of f at sx """

                fx = self.f(x_vals, np.array([sx]))
                # Add penalty
                for j in range(self.o):
                    if fx[j] > self.f_ub[j]:
                        fx[:] = fx[:] + 10.0 * (fx[j] - self.f_ub[j])
                result = min(np.dot(fx, self.weights) - self.best, 0.0)
                return result * s_dist.pdf(np.array([sx]))

            y = integrate.quad(weighted_f, -np.infty, np.infty)
            return y[0]
        # If there is at least one feasible point and the number of sim outs
        # is greater than 1, then evaluate the EI over fi* via MC integration
        else:
            if self.sample_size is None:
                self.sample_size = int(10 * s_vals_mean.size ** 2)
            # Construct the distribution for sampling
            s_cov = stats.Covariance.from_diagonal(s_vals_sd)
            s_dist = stats.multivariate_normal(mean=s_vals_mean, cov=s_cov)
            result = 0.0
            # Loop over sample size
            for i in range(self.sample_size):
                s_vals = s_dist.rvs().flatten()
                fi = self.f(x_vals, s_vals)
                # Add penalty
                for j in range(self.o):
                    if fi[j] > self.f_ub[j]:
                        fi[:] = fi[:] + 10.0 * (fi[j] - self.f_ub[j])
                result = min(np.dot(fi, self.weights) - self.best, 0.0)
            result /= self.sample_size
            return result

    def scalarizeGrad(self, f_vals, g_vals):
        """ Not implemented for this acquisition function, do not use
        gradient-based methods.

        """

        raise NotImplementedError("The EI-based acquisition does not " +
                                  "support the usage of gradients")
