
""" Implementations of the SurrogateFunction class.

This module contains implementations of the SurrogateFunction ABC, which rely
on Gaussian basis functions (i.e., Gaussian processes).

The classes include:
 * ``GaussRBF`` -- fits Gaussian radial basis functions (RBFs)

"""

import numpy as np
from parmoo.structs import SurrogateFunction
from scipy.spatial.distance import cdist
from scipy.stats import tstd
from parmoo.util import xerror


class GaussRBF(SurrogateFunction):
    """ A RBF surrogate model, using a Gaussian basis.

    This class implements a local RBF surrogate with a Gaussian basis,
    using the SurrogateFunction ABC.

    """

    # Slots for the UniformRandom class
    __slots__ = ['m', 'n', 'lb', 'ub', 'x_vals', 'f_vals', 'eps', 'std_dev',
                 'nugget', 'loc_inds', 'tr_center', 'tr_radius',
                 'weights', 'prior', 'v', 'w', 'order', 'y_std_dev']

    def __init__(self, m, lb, ub, hyperparams):
        """ Constructor for the GaussRBF class.

        Args:
            m (int): The number of objectives to fit.

            lb (numpy.ndarray): A 1d array of lower bounds for the design
                region. The number of design variables is inferred from the
                dimension of lb.

            ub (numpy.ndarray): A 1d array of upper bounds for the design
                region. The dimension must match ub.

            hyperparams (dict): A dictionary of hyperparameters for the
                RBF models, including:
                 * des_tols (numpy.ndarray, optional): A 1d array whose length
                   matches lb and ub. Each entry is a number (greater than 0)
                   specifying the design space tolerance for that variable.
                   By default, des_tols = [1.0e-8, ..., 1.0e-8].
                 * tail_order (int, optional): Order of the polynomial tail.
                   Can be 0 or 1, defaults to 0.

        Returns:
            GaussRBF: A new GaussRBF object.

        """

        # Check inputs
        xerror(o=m, lb=lb, ub=ub, hyperparams=hyperparams)
        # Initialize problem dimensions
        self.m = m
        self.lb = lb
        self.ub = ub
        self.n = self.lb.size
        self.std_dev = 0.0
        # Create empty database
        self.x_vals = np.zeros((0, self.n))
        self.f_vals = np.zeros((0, self.m))
        self.weights = np.zeros((0, 0))
        self.prior = np.zeros((self.n+1, self.m))
        self.v = np.zeros((0, 0))
        self.w = np.zeros((0, 0))
        self.y_std_dev = np.ones(self.m)
        # Initialize trust region settings
        self.tr_center = np.zeros(0)
        self.tr_radius = np.zeros(0)
        self.loc_inds = []
        # Check for the 'nugget' optional value in hyperparams
        if 'nugget' in hyperparams:
            if isinstance(hyperparams['nugget'], float):
                self.nugget = hyperparams['nugget']
                if self.nugget < 0.0:
                    raise ValueError("hyperparams['nugget'] cannot be a"
                                     + " negative number")
            else:
                raise ValueError("hyperparams['nugget'] contained an illegal"
                                 + " value")
        else:
            self.nugget = 0.0
        # Check for 'des_tols' optional key in hyperparams
        if 'des_tols' in hyperparams:
            if isinstance(hyperparams['des_tols'], np.ndarray):
                if hyperparams['des_tols'].size == self.n:
                    if np.all(hyperparams['des_tols'] > 0.0):
                        self.eps = hyperparams['des_tols']
                    else:
                        raise ValueError("hyperparams['des_tols'] must all be"
                                         + " greater than 0")
                else:
                    raise ValueError("hyperparams['des_tols'] must have length"
                                     + " n")
            else:
                raise ValueError("hyperparams['des_tols'] contained an illegal"
                                 + " value")
        else:
            self.eps = np.zeros(self.n)
            self.eps[:] = 1.0e-8
        # Check for 'tail_order' optional key in hyperparams
        if 'tail_order' in hyperparams:
            if isinstance(hyperparams['tail_order'], int):
                if hyperparams['tail_order'] in [0, 1]:
                    self.order = hyperparams['tail_order']
                else:
                    raise ValueError("hyperparams['tail_order'] must be "
                                     + "0 or 1")
            else:
                raise ValueError("hyperparams['tail_order'] contained an "
                                 + "illegal value")
        else:
            self.order = 0
        return

    def __gaussian(self, r):
        """ Gaussian bump function """
        return np.exp(-(1.0 / self.std_dev * r) ** 2.0)

    def fit(self, x, f):
        """ Fit a new Gaussian RBF to the given data.

        Args:
             x (numpy.ndarray): A 2d array containing the list of
                 design points.

             f (numpy.ndarray): A 2d array containing the corresponding list
                 of objective values.

        """

        # Check that the x and f values are legal
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array")
        if isinstance(f, np.ndarray):
            if f.shape[0] == 0:
                raise ValueError("no data provided")
            if self.m != np.size(f[0, :]):
                raise ValueError("each row of f must have length m")
        else:
            raise TypeError("f must be a numpy array")
        if x.shape[0] != f.shape[0]:
            raise ValueError("x and f must have equal lengths")
        # Initialize the internal database with x and f
        self.x_vals = x
        self.f_vals = f
        # Initialize the local indices for future usage
        self.tr_center = self.lb[:] - np.ones(self.n)
        self.tr_radius = np.zeros(self.n)
        return

    def update(self, x, f):
        """ Update an existing Gaussian RBF using new data.

        Args:
             x (numpy.ndarray): A 2d array containing the list of
                 new design points, with which to update the surrogate
                 models.

             f (numpy.ndarray): A 2d array containing the corresponding list
                 of objective values.

        """

        # Check that the x and f values are legal
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array")
        if isinstance(f, np.ndarray):
            if f.shape[0] == 0:
                return    # No new data, do nothing
            if self.m != np.size(f[0, :]):
                raise ValueError("each row of f must have length m")
        else:
            raise TypeError("f must be a numpy array")
        if x.shape[0] != f.shape[0]:
            raise ValueError("x and f must have equal lengths")
        # Update the internal database with x and f
        self.x_vals = np.concatenate((self.x_vals, x), axis=0)
        self.f_vals = np.concatenate((self.f_vals, f), axis=0)
        # Reinitialize the local indices for future usage
        self.tr_center = -np.ones(self.n)
        self.tr_radius = np.zeros(self.n)
        return

    def setTrustRegion(self, center, radius):
        """ Set the new trust region center and refit the local RBF.

        Args:
            center (numpy.ndarray): A 1d array containing the new trust
                region center.

            radius (numpy.ndarray or float): The trust region radius.

        """

        # Check that the center is legal
        if not isinstance(center, np.ndarray):
            raise TypeError("center must be a numpy array")
        else:
            if center.size != self.n:
                raise ValueError("center must have length n")
            elif (np.any(center < self.lb - self.eps) or
                  np.any(center > self.ub + self.eps)):
                raise ValueError("center cannot be infeasible")
        # Check that the radius is legal
        if isinstance(radius, np.ndarray):
            if radius.size != self.n:
                raise ValueError("radius must have length n")
            elif np.any(radius <= 0):
                raise ValueError("radius must be positive")
        elif isinstance(radius, float):
            if radius <= 0:
                raise ValueError("radius must be positive")
        else:
            raise TypeError("radius must be a numpy array or float")
        # Some helper variables used below
        rad_tmp = np.zeros(self.n)
        rad_tmp[:] = radius
        refit = False
        # If the radius is infinite, there is no need to fit more than once
        if np.all(rad_tmp == np.infty) and np.all(self.tr_radius == 0):
            # Set the trust region center and radius to large values
            self.tr_center = (self.ub + self.lb) / 2.0
            self.tr_radius = (self.ub - self.lb) / 2.0
            # Compute the standard deviation for the Gaussian bubbles
            self.std_dev = np.power(np.prod(self.tr_radius * 2.0) /
                                    float(self.x_vals.shape[0]),
                                    1.0 / float(self.n))
            # Use all points in the current database
            self.loc_inds = [i for i in range(self.x_vals.shape[0])]
            refit = True
        # Otherwise, if the nearest neighbor has changed, refit the RBF
        elif np.any(self.tr_center != center) or np.any(self.tr_radius
                                                        != rad_tmp):
            # Update the trust region center and radius
            self.tr_center = center
            self.tr_radius = rad_tmp
            # Update the standard deviation for the Gaussian bubbles
            self.std_dev = np.linalg.norm(self.tr_radius * 2)
            # Get points in the new trust region
            self.loc_inds = []
            rdists = np.asarray([np.linalg.norm((xj - center))
                                 for xj in self.x_vals])
            for i, ri in enumerate(rdists):
                if ri < 3.0 * self.std_dev:
                    self.loc_inds.append(i)
            refit = True
        # Only do the following if we are re-fitting the models
        if refit:
            # Build the Gaussian covariance matrix
            if len(self.loc_inds) < self.x_vals.shape[0]:
                cov = self.__gaussian(cdist(self.x_vals[self.loc_inds, :],
                                            self.x_vals[self.loc_inds, :],
                                            'euclidean'))
            else:
                cov = self.__gaussian(cdist(self.x_vals, self.x_vals,
                                      'euclidean'))
            # Add the nugget, if present
            if self.nugget > 0:
                for i in range(len(self.loc_inds)):
                    cov[i, i] = cov[i, i] + self.nugget
            # Get eigenvalue decomp to solve the SPD system with multiple RHS
            self.w = np.zeros(cov.shape[0])
            self.v = np.zeros(cov.shape)
            self.w, self.v = np.linalg.eigh(cov)
            # Check the smallest singular value for a bad solution
            sigma_n = np.min(self.w)
            if sigma_n < 1.0e-8:
                for i in range(len(self.loc_inds)):
                    cov[i, i] = cov[i, i] + 1.0e-8 - sigma_n
                self.w, self.v = np.linalg.eigh(cov)
            # Fit prior weights and remove tail effects from RHS
            rhs = self.f_vals[self.loc_inds, :].copy()
            if self.order >= 0:
                self.prior[-1, :] = np.sum(rhs, axis=0) / rhs.shape[0]
                rhs[:, :] = rhs[:, :] - self.prior[-1, :]
                if self.order >= 1:
                    x_inds = [int(i) for i in self.loc_inds
                              if np.linalg.norm(center - self.x_vals[i])
                              <= self.std_dev + 1.0e-8]
                    A = self.x_vals[x_inds, :].copy()
                    b = self.f_vals[x_inds, :].copy()
                    b -= self.prior[-1, :]
                    self.prior[:-1, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                    rhs[:, :] = rhs[:, :] - np.dot(self.x_vals[self.loc_inds],
                                                   self.prior[:-1])
            self.y_std_dev = np.maximum(tstd(rhs, axis=0), 0.01)
            # Finish the solve
            self.weights = np.zeros((self.m, rhs.shape[0]))
            for i in range(self.m):
                tmp = np.dot(self.v.transpose(), rhs[:, i]) / self.w[:]
                self.weights[i, :] = np.dot(self.v, tmp)
        return

    def evaluate(self, x):
        """ Evaluate the Gaussian RBF at a design point.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which to the Gaussian RBF should be evaluated.

        Returns:
            numpy.ndarray: A 1d array containing the predicted objective value
            at x.

        """

        dists = self.__gaussian(cdist(self.x_vals[self.loc_inds],
                                [x])).flatten()
        mean = self.prior[-1, :].copy()
        if self.order >= 1:
            mean += np.dot(x, self.prior[:-1, :])
        return np.dot(self.weights, dists) + mean[:]

    def gradient(self, x):
        """ Evaluate the gradients of the Gaussian RBF at a design point.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the gradient of the RBF should be evaluated.

        Returns:
            numpy.ndarray: A 2d array containing the Jacobian matrix of the
            RBF interpolants at x.

        """

        outs = np.zeros((len(self.loc_inds), self.n))
        dists = self.__gaussian(cdist(self.x_vals[self.loc_inds],
                                      [x])).flatten()
        for i, xi in enumerate(self.x_vals[self.loc_inds]):
            outs[i, :] = 2.0 * (xi - x) * dists[i] / (self.std_dev ** 2.0)
        if self.order == 1:
            return np.dot(self.weights, outs) + self.prior[:-1, :].T
        else:
            return np.dot(self.weights, outs)

    def stdDev(self, x):
        """ Evaluate the standard deviation (uncertainty) of the Gaussian RBF at x.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the standard deviation should be evaluated.

        Returns:
            numpy.ndarray: A 1d array containing the standard deviation at x.

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
        # Get vector of Gaussian-transformed distances
        dists = self.__gaussian(cdist(self.x_vals[self.loc_inds],
                                [x])).flatten()
        # Solve using previously factored Kernel matrix
        stdd_weights = np.zeros(self.v.shape[0])
        tmp = np.dot(self.v.transpose(), dists) / self.w[:]
        stdd_weights[:] = np.dot(self.v, tmp)
        # Evaluate standard deviation of all m surrogates at x
        return (np.sqrt(max(self.__gaussian(0.0) -
                            np.dot(stdd_weights, dists), 0)
                * np.ones(self.m))) * self.y_std_dev[:]

    def stdDevGrad(self, x):
        """ Evaluate the gradient of the standard deviation of the GaussRBF at x.
 
        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the gradient of standard deviation should be evaluated.

        Returns:
            numpy.ndarray: A 2d array containing the Jacobian matrix of the
            standard deviation at x.

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
        # Get vector of Gaussian-transformed distances
        dists = self.__gaussian(cdist(self.x_vals[self.loc_inds],
                                      [x])).flatten()
        # Solve using previously factored Kernel matrix
        stdd_weights = np.zeros(self.v.shape[0])
        tmp = np.dot(self.v.transpose(), dists) / self.w[:]
        stdd_weights[:] = np.dot(self.v, tmp)
        # Evaluate standard deviation of all m surrogates at x
        stdd = np.sqrt(max(self.__gaussian(0.0) - np.dot(stdd_weights, dists),
                           0))
        # Evaluate all m gradients at x
        kgrad = np.zeros((self.v.shape[0], self.n))
        for i, xi in enumerate(self.x_vals[self.loc_inds]):
            kgrad[i, :] = 2.0 * (xi - x) * dists[i] / (self.std_dev ** 2.0)
        # Just return 0 when derivative is undefined (at training points)
        if stdd < 1.0e-4:
            result = np.zeros(self.n)
        else:
            result = -np.dot(kgrad.T, stdd_weights).flatten() / stdd
        # Build Jacobian (all m rows identical)
        jac = np.zeros((self.m, self.n))
        for i in range(self.m):
            jac[i, :] = result[:] * self.y_std_dev[i]
        return jac

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        import json

        # Serialize RBF object in dictionary
        gp_state = {'m': self.m,
                    'n': self.n,
                    'std_dev': self.std_dev,
                    'loc_inds': self.loc_inds,
                    'nugget': self.nugget,
                    'order': self.order}
        # Serialize numpy.ndarray objects
        gp_state['lb'] = self.lb.tolist()
        gp_state['ub'] = self.ub.tolist()
        gp_state['x_vals'] = self.x_vals.tolist()
        gp_state['f_vals'] = self.f_vals.tolist()
        gp_state['y_std_dev'] = self.y_std_dev.tolist()
        gp_state['eps'] = self.eps.tolist()
        gp_state['tr_center'] = self.tr_center.tolist()
        gp_state['weights'] = self.weights.tolist()
        gp_state['prior'] = self.prior.tolist()
        gp_state['v'] = self.v.tolist()
        gp_state['w'] = self.w.tolist()
        # Save file
        with open(filename, 'w') as fp:
            json.dump(gp_state, fp)
        return

    def load(self, filename):
        """ Reload important data into this class after a previous save.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data has been saved.

        """

        import json

        # Load file
        with open(filename, 'r') as fp:
            gp_state = json.load(fp)
        # Deserialize RBF object from dictionary
        self.m = gp_state['m']
        self.n = gp_state['n']
        self.std_dev = gp_state['std_dev']
        self.nugget = gp_state['nugget']
        self.loc_inds = gp_state['loc_inds']
        self.order = gp_state['order']
        # Deserialize numpy.ndarray objects
        self.lb = np.array(gp_state['lb'])
        self.ub = np.array(gp_state['ub'])
        self.x_vals = np.array(gp_state['x_vals'])
        self.f_vals = np.array(gp_state['f_vals'])
        self.y_std_dev = np.array(gp_state['y_std_dev'])
        self.eps = np.array(gp_state['eps'])
        self.tr_center = np.array(gp_state['tr_center'])
        self.weights = np.array(gp_state['weights'])
        self.prior = np.array(gp_state['prior'])
        self.v = np.array(gp_state['v'])
        self.w = np.array(gp_state['w'])
        return
