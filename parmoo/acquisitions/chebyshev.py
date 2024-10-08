
""" Implementations of the (augmented) Chebyshev scalarization technique.

This module contains implementations of the AcquisitionFunction ABC, which
use the weighted-sum technique.

The classes include:
 * ``UniformAugChebyshev`` (sample Chebyshev weights uniformly)
 * ``FixedAugChebyshev`` (uses a fixed weights, which can be set upon init)

"""

from jax import numpy as jnp
import numpy as np
import inspect
from parmoo.structs import AcquisitionFunction
from parmoo.util import xerror


class UniformAugChebyshev(AcquisitionFunction):
    """ Randomly generate weights and scalarize via augmented Chebyshev.

    Generates uniformly distributed scalarization weights, by randomly
    sampling the probability simplex.

    """

    # Slots for the UniformAugChebyshev class
    __slots__ = ['n', 'o', 'lb', 'ub', 'weights', 'alpha', 'np_rng', 'eps']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the UniformAugChebyshev class.

        Args:
            o (int): The number of objectives.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for tuning
                the acquisition function. May include:
                 * 'alpha' (float): The weight to place on the linear
                   term. When not present, defaults to
                   1e-4 / number of objectives.

        Returns:
            UniformAugChebyshev: A new UniformAugChebyshev generator.

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
        # Check hyperparameters
        self.eps = jnp.finfo(jnp.ones(1)).eps
        self.alpha = self.eps ** 0.25 / self.o
        if 'alpha' in hyperparams.keys():
            if isinstance(hyperparams['alpha'], float):
                if hyperparams['alpha'] >= 0 and hyperparams['alpha'] <= 1:
                    self.alpha = hyperparams['alpha']
                else:
                    raise ValueError("When present, hyperparams['alpha'] " +
                                     "must be in the range [0, 1]")
            else:
                raise TypeError("When present, hyperparams['alpha'] " +
                                "must be a float type")
        # Check the hyperparameter dictionary for random generator
        if 'np_random_gen' in hyperparams:
            if isinstance(hyperparams['np_random_gen'], np.random.Generator):
                self.np_rng = hyperparams['np_random_gen']
            else:
                raise TypeError("When present, hyperparams['np_random_gen'] "
                                "must be an instance of the class "
                                "numpy.random.Generator")
        else:
            self.np_rng = np.random.default_rng()
        return

    def useSD(self):
        """ Query whether this method uses uncertainties.

        When False, allows users to shortcut expensive uncertainty
        computations.

        """

        return False

    def setTarget(self, data, penalty_func):
        """ Randomly generate a new vector of scalarizing weights.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database.

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
                raise ValueError("penalty_func() must accept exactly one"
                                 + " input")
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
        # Sample the weights uniformly from the unit simplex
        self.weights = -np.log(1.0 - self.np_rng.random(self.o))
        self.weights = self.weights[:] / sum(self.weights[:])
        # If data is empty, randomly select weights and starting point
        if no_data:
            # Randomly select a starting point
            x_start = (self.np_rng.random(self.n) * (self.ub - self.lb)
                       + self.lb)
            return x_start
        # If data is nonempty but pf is empty, use a penalty to select
        elif pf is None or pf['x_vals'].shape[0] == 0:
            x_best = np.zeros(data['x_vals'].shape[1])
            p_best = np.inf
            for xi, fi, ci in zip(data['x_vals'], data['f_vals'],
                                  data['c_vals']):
                p_temp = np.sum(fi) / np.sqrt(self.eps) + np.sum(ci)
                if p_temp < p_best:
                    x_best = xi
                    p_best = p_temp
            return x_best
        else:
            xx = np.zeros(1)
            sx = np.zeros(1)
            sdx = np.zeros(1)
            i = np.argmin(np.asarray([self.scalarize(fi, xx, sx, sdx)
                                      for fi in pf['f_vals']]))
            x = pf['x_vals'][i, :]
            return x

    def scalarize(self, f_vals, x_vals, s_vals_mean, s_vals_sd, manifold=None):
        """ Scalarize a vector of function values using the current weights.

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

        if not isinstance(manifold, int):
            return (jnp.max(f_vals * self.weights)
                    + self.alpha * jnp.sum(f_vals))
        else:
            return ((f_vals * self.weights)[manifold]
                    + self.alpha * jnp.sum(f_vals))

    def getManifold(self, f_vals):
        """ Check which manifold is active for a given function value.

        Each component of f_vals is its own smooth manifold, but once
        the max function is applied for the augmented Chebyshev scalarization,
        only 1 component is typically active in the surrogate
        minimization problem.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values to be scalarized.

        Returns:
            numpy.ndarray, dtype=int: A 1d array of integers, matching the
            length of f_vals, where
             - 0 indicates that this component of the manifold is inactive and
             - 1 indicates that this component of the manifold is active.

        """

        return np.isclose(f_vals * self.weights,
                          (f_vals * self.weights).max(),
                          atol=np.sqrt(self.eps)).astype(int)


class FixedAugChebyshev(AcquisitionFunction):
    """ Use fixed weights with the augmented Chebyshev scalarization.

    Use a fixed scalarization scheme, based on a fixed weighted sum.

    """

    # Slots for the FixedAugChebyshev class
    __slots__ = ['n', 'o', 'lb', 'ub', 'weights', 'alpha', 'np_rng', 'eps']

    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the FixedAugChebyshev class.

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
                 * 'alpha' (float): The weight to place on the linear
                   term. When not present, defaults to
                   1e-4 / number of objectives.

        Returns:
            FixedAugChebyshev: A new FixedAugChebyshev generator.

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
        # Check the hyperparameter dictionary for random generator
        if 'np_random_gen' in hyperparams:
            if isinstance(hyperparams['np_random_gen'], np.random.Generator):
                self.np_rng = hyperparams['np_random_gen']
            else:
                raise TypeError("When present, hyperparams['np_random_gen'] "
                                "must be an instance of the class "
                                "numpy.random.Generator")
        else:
            self.np_rng = np.random.default_rng()
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
            # If no weights provided, sample from the unit simplex
            self.weights = -np.log(1.0 - self.np_rng.random(self.o))
            self.weights = self.weights[:] / sum(self.weights[:])
        # Check hyperparameters dictionary for alpha
        self.eps = jnp.finfo(jnp.ones(1)).eps
        self.alpha = self.eps ** 0.25 / self.o
        if 'alpha' in hyperparams.keys():
            if isinstance(hyperparams['alpha'], float):
                if hyperparams['alpha'] >= 0 and hyperparams['alpha'] <= 1:
                    self.alpha = hyperparams['alpha']
                else:
                    raise ValueError("When present, hyperparams['alpha'] " +
                                     "must be in the range [0, 1]")
            else:
                raise TypeError("When present, hyperparams['alpha'] " +
                                "must be a float type")
        return

    def useSD(self):
        """ Query whether this method uses uncertainties.

        When False, allows users to shortcut expensive uncertainty
        computations.

        """

        return False

    def setTarget(self, data, penalty_func):
        """ Randomly generate a feasible starting point.

        Args:
            data (dict): A dictionary specifying the current function
                evaluation database.

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
                raise ValueError("penalty_func() must accept exactly one"
                                 + " input")
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
            # Randomly select a starting point
            x_start = (self.np_rng.random(self.n) * (self.ub - self.lb)
                       + self.lb)
            return x_start
        # If data is nonempty but pf is empty, use a penalty to select
        elif pf is None or pf['x_vals'].shape[0] == 0:
            x_best = np.zeros(data['x_vals'].shape[1])
            p_best = np.inf
            for xi, fi, ci in zip(data['x_vals'], data['f_vals'],
                                  data['c_vals']):
                p_temp = np.sum(fi) / np.sqrt(self.eps) + np.sum(ci)
                if p_temp < p_best:
                    x_best = xi
                    p_best = p_temp
            return x_best
        else:
            xx = np.zeros(1)
            sx = np.zeros(1)
            sdx = np.zeros(1)
            i = np.argmin(np.asarray([self.scalarize(fi, xx, sx, sdx)
                                      for fi in pf['f_vals']]))
            x = pf['x_vals'][i, :]
            return x

    def scalarize(self, f_vals, x_vals, s_vals_mean, s_vals_sd, manifold=None):
        """ Scalarize a vector of function values using the current weights.

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

        if not isinstance(manifold, int):
            return (jnp.max(f_vals * self.weights)
                    + self.alpha * jnp.sum(f_vals))
        else:
            return ((f_vals * self.weights)[manifold]
                    + self.alpha * jnp.sum(f_vals))

    def getManifold(self, f_vals):
        """ Check which manifold is active for a given function value.

        Each component of f_vals is its own smooth manifold, but once
        the max function is applied for the augmented Chebyshev scalarization,
        only 1 component is typically active in the surrogate
        minimization problem.

        Args:
            f_vals (numpy.ndarray): A 1d array specifying the function
                values to be scalarized.

        Returns:
            numpy.ndarray, dtype=int: A 1d array of integers, matching the
            length of f_vals, where
             - 0 indicates that this component of the manifold is inactive and
             - 1 indicates that this component of the manifold is active.

        """

        return np.isclose(f_vals * self.weights,
                          (f_vals * self.weights).max(),
                          atol=np.sqrt(self.eps)).astype(int)
