""" The abstract base class (ABC) for SurrogateFunction objects. """

from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import tstd


class SurrogateFunction(ABC):
    """ ABC describing surrogate functions.

    This class contains the following methods.
     * ``fit(x, f)``
     * ``update(x, f)``
     * ``setTrustRegion(center, radius)`` (default implementation provided)
     * ``evaluate(x)``
     * ``stdDev(x)``
     * ``improve(x, global_improv)`` (default implementation provided)
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, m, lb, ub, hyperparams):
        """ Constructor for the SurrogateFunction class.

        Args:
            m (int): The number of objectives to fit.

            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

            hyperparams (dict): A dictionary of hyperparameters to be used
                by the surrogate models, including:
                 * des_tols (ndarray, optional): A 1D array whose length
                   matches lb and ub. Each entry is a number (greater than 0)
                   specifying the design space tolerance for that variable.

        Returns:
            SurrogateFunction: A new SurrogateFunction object.

        """

    @abstractmethod
    def fit(self, x, f):
        """ Fit a new surrogate to the given data.

        Args:
             x (ndarray): A 2D array containing the design points to fit.

             f (ndarray): A 2D array of the corresponding objectives values.

        """

    @abstractmethod
    def update(self, x, f):
        """ Update an existing surrogate model using new data.

        Args:
             x (ndarray): A 2D array containing new design points to fit.

             f (ndarray): A 2D array of the corresponding objectives values.

        """

    def setTrustRegion(self, center, radius):
        """ Alert the surrogate of the trust-region center and radius.

        Default implementation does nothing, which would be the case for a
        global surrogate model.

        Args:
            center (ndarray): A 1D array containing the center for a local fit.

            radius (ndarray or float): The radius for the local fit.

        """

        return

    @abstractmethod
    def evaluate(self, x):
        """ Evaluate the surrogate at a design point.

        Note: For best performance, make sure that jax can jit this method.

        Additionally, for compatibility with gradient-based solvers,
        this method must be implemented in jax and be differentiable
        via the jax.jacrev() tool.

        Args:
            x (ndarray): A 1D array containing the design point to evaluate.

        Returns:
            ndarray: A 1D array containing the predicted outputs at x.

        """

    def stdDev(self, x):
        """ Evaluate the standard deviation of the surrogate at x.

        Note: this method need not be implemented when the acquisition
        function does not use the model uncertainty.

        Additionally, for compatibility with gradient-based solvers,
        this method must be implemented in jax and be differentiable
        via the jax.jacrev() tool.

        Args:
            x (ndarray): A 1D array containing the design point to evaluate.

        Returns:
            ndarray: A 1D array containing the output standard deviation at x.

        """

        raise NotImplementedError("This class method has not been implemented")

    def improve(self, x, global_improv):
        """ Suggests a design to evaluate to improve the surrogate near x.

        A default implementation is given based on random sampling.
        Re-implement the improve method to overwrite the default
        policy.

        Args:
            x (ndarray): A 1D array containing a design point where greater
                accuracy is needed.

            global_improv (Boolean): When True, ignore the value of x and
                seek global model improvement.

        Returns:
            ndarray: A 2D array containing a list of (at least 1) design points
            that could be evaluated to improve the surrogate model's accuracy.

        """

        # Check that the x is legal
        try:
            if x.size != self.n:
                raise ValueError("x must have length n")
            elif (np.any(x < self.lb - self.eps) or
                  np.any(x > self.ub + self.eps)):
                raise ValueError("x cannot be infeasible")
        except AttributeError:
            raise TypeError("x must be a numpy array-like object")
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
