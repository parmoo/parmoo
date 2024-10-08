
""" Implementations of the SurrogateFunction class.

This module contains implementations of the SurrogateFunction ABC, which rely
on local linear models.

The classes include:
 * ``Linear`` -- fits a local linear surrogate

"""

from jax import numpy as jnp
import numpy as np
from parmoo.structs import SurrogateFunction
from parmoo.util import xerror


class Linear(SurrogateFunction):
    """ A local linear surrogate model.

    This class implements a local (TR constrained) linear surrogate,
    using the SurrogateFunction ABC.

    """

    # Slots for the UniformRandom class
    __slots__ = ['m', 'n', 'lb', 'ub', 'x_vals', 'f_vals', 'eps',
                 'loc_inds', 'tr_center', 'weights']

    def __init__(self, m, lb, ub, hyperparams):
        """ Constructor for the Linear class.

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

        Returns:
            Linear: A new Linear surrogate object.

        """

        # Check inputs
        xerror(o=m, lb=lb, ub=ub, hyperparams=hyperparams)
        # Initialize problem dimensions
        self.m = m
        self.lb = lb
        self.ub = ub
        self.n = self.lb.size
        # Create empty database
        self.x_vals = np.zeros((0, self.n))
        self.f_vals = np.zeros((0, self.m))
        self.weights = np.zeros(self.n + 1)
        # Initialize trust-region settings
        self.tr_center = np.zeros(0)
        self.loc_inds = []
        # Check for 'des_tols' optional key in hyperparams
        mu = np.sqrt(jnp.finfo(jnp.ones(1)).eps)
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
            self.eps[:] = mu
        return

    def fit(self, x, f):
        """ Fit a new linear model to the given data.

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
        # Reset the center to trigger a re-fit of the model
        self.tr_center = self.lb[:] - np.ones(self.n)
        return

    def update(self, x, f):
        """ Update an existing linear model using new data.

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
        # Reset the center to trigger a re-fit of the model
        self.tr_center = self.lb - np.ones(self.n)
        return

    def setTrustRegion(self, center, radius):
        """ Set the new trust-region center and refit the local linear model.

        Args:
            center (numpy.ndarray): A 1d array containing the new trust-region
                center.

            radius (numpy.ndarray or float): The trust-region radius.

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
        # If the center has changed, refit the model
        if np.any(np.abs(self.tr_center - center) > self.eps):
            # Update the center and sort the nearest neighbors
            self.tr_center = center
            idists = np.argsort(np.asarray([np.linalg.norm(xj - center)
                                            for xj in self.x_vals]))
            # Check the n nearest neighbors
            xn = self.x_vals[idists[self.n]]
            r_tmp = np.linalg.norm(center - xn)
            # Get all points within the radius
            self.loc_inds = [int(i) for i in idists
                             if np.linalg.norm(np.maximum(np.abs(center -
                                               self.x_vals[i]) - self.eps, 0))
                             <= r_tmp]
            # Get (min norm) LS fit
            A = np.hstack((self.x_vals[self.loc_inds],
                           np.ones((len(self.loc_inds), 1))))
            self.weights = np.linalg.lstsq(A, self.f_vals[self.loc_inds],
                                           rcond=None)[0]
        return

    def evaluate(self, x):
        """ Evaluate the linear model at a design point.

        Args:
            x (numpy.ndarray): A 1d array containing the design point at
                which the linear model should be evaluated.

        Returns:
            numpy.ndarray: A 1d array containing the predicted objective value
            at x.

        """

        return jnp.dot(self.weights[:-1].T, x).flatten() + self.weights[-1]

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        import json

        # Serialize RBF object in dictionary
        ls_state = {'m': self.m,
                    'n': self.n,
                    'loc_inds': self.loc_inds}
        # Serialize numpy.ndarray objects
        ls_state['lb'] = self.lb.tolist()
        ls_state['ub'] = self.ub.tolist()
        ls_state['x_vals'] = self.x_vals.tolist()
        ls_state['f_vals'] = self.f_vals.tolist()
        ls_state['eps'] = self.eps.tolist()
        ls_state['tr_center'] = self.tr_center.tolist()
        ls_state['weights'] = self.weights.tolist()
        # Save file
        with open(filename, 'w') as fp:
            json.dump(ls_state, fp)
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
            ls_state = json.load(fp)
        # Deserialize RBF object from dictionary
        self.m = ls_state['m']
        self.n = ls_state['n']
        self.loc_inds = ls_state['loc_inds']
        # Deserialize numpy.ndarray objects
        self.lb = np.array(ls_state['lb'])
        self.ub = np.array(ls_state['ub'])
        self.x_vals = np.array(ls_state['x_vals'])
        self.f_vals = np.array(ls_state['f_vals'])
        self.eps = np.array(ls_state['eps'])
        self.tr_center = np.array(ls_state['tr_center'])
        self.weights = np.array(ls_state['weights'])
        return
