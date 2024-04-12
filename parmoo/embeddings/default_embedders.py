
""" Default embeddings for ParMOO.

This module contains implementations of the Embedder ABC for all of ParMOO's
default embeddings.

This includes the following classes:
 * ``ContinuousEmbedder`` for real/continuous variables,
 * ``IntegerEmbedder`` for integer variables,
 * ``CategoricalEmbedder`` for categorical variables, and
 * ``IdentityEmbedder`` for raw/unscaled variables.

"""

from jax import jit
from jax import numpy as jnp
import numpy as np
from parmoo.structs import Embedder


class ContinuousEmbedder(Embedder):
    """ An Embedder class for continuous variables that simply rescales and
    shifts all inputs to the box [0, 1]. """

    __slots__ = ['scale', 'scaled_des_tol', 'shift']

    def __init__(self, settings):
        """ Generate the scaling matrices for a continuous variable.

        Args:
            settings (dict): Contains the following keys
                 * 'lb' (float or int, required): This specifies the lower
                   bound for the design variable. This value must be strictly
                   less than 'ub' (below) up to the tolerance (below).
                 * 'ub' (float or int, required): This specifies the upper
                   bound for the design variable. This value must be strictly
                   greater than 'lb' (above) up to the tolerance (below).
                 * 'des_tol' (float, optional): This specifies the design
                   tolerance for this variable, i.e., the minimum spacing
                   before two design values are considered equivalent up to
                   measurement error. If not specified, the
                   default value is 1.0e-8 * max(ub - lb, 1.0e-4).

        """

        # Error handling and extracting input bounds
        if isinstance(settings, dict):
            if 'lb' in settings:
                try:
                    lb = float(settings['lb'])
                except BaseException:
                    raise TypeError("lower bound must contain a float or int")
            else:
                raise KeyError("'lb' is a required key for continuous "
                               "design variables")
            if 'ub' in settings:
                try:
                    ub = float(settings['ub'])
                except BaseException:
                    raise TypeError("upper bound must contain a float or int")
            else:
                raise KeyError("'ub' is a required key for continuous "
                               "design variables")
            if 'des_tol' in settings:
                des_tol = settings['des_tol']
                if not isinstance(des_tol, float):
                    raise TypeError("design tolerance must be a float type")
                if des_tol <= 0:
                    raise ValueError("design tolerance must be strictly "
                                     "greater than 0")
            else:
                des_tol = 1.0e-8 * max(ub - lb, 1.0e-4)
            if lb + des_tol > ub:
                raise ValueError("lower bound must be strictly less than "
                                 "upper bound for all design variables "
                                 "up to the design tolerance")
        else:
            raise TypeError("settings must be a dictionary")
        # Calculate the embedding as a sequence of matrix operations
        self.scale = jnp.ones(1) * (ub - lb)
        self.shift = jnp.ones(1) * lb
        self.scaled_des_tol = np.ones(1) * des_tol / self.scale
        # Jit the embedder and extractor
        self.embed = jit(self._embed)
        self.extract = jit(self._extract)
        return

    def getLatentDesTols(self):
        """ Get the design tolerances along each dimension of the embedding.

        Returns:
            numpy.ndarray: array of design space tolerances after embedding

        """

        return self.scaled_des_tol

    def getFeatureDesTols(self):
        """ Get the design tolerances in the feature space (pre-embedding).

        Returns:
            float: the design tolerance in the feature space -- a value of
            0 indicates a discrete variable

        """

        return float(self.scaled_des_tol[0] * self.scale[0])

    def getEmbeddingSize(self):
        """ Get dimension of embedded space.

        Returns:
            int: the dimension of the embedded space.

        """
        
        return 1

    def getInputType(self):
        """ Get the input type for this embedder.

        Returns:
            str: A numpy string representation of the input type from the
            feature space.

        """
        
        return 'f8'

    def getLowerBounds(self):
        """ Get a vector of lower bounds for the embedded space.

        Returns:
            numpy.ndarray: array of lower bounds in embedded space

        """

        return np.zeros(1)

    def getUpperBounds(self):
        """ Get a vector of upper bounds for the embedded space.

        Returns:
            numpy.ndarray: array of upper bounds in the embedded space

        """

        return np.ones(1)

    def _embed(self, x):
        """ Embed a design input as n-dimensional vector for ParMOO.

        Args:
            x (float): The value of the design variable to embed.

        Returns:
            numpy.ndarray: A numpy array of length 1 containing a
            rescaling of x

        """

        return jnp.minimum(jnp.maximum((x - self.shift) / self.scale, 0), 1)

    def _extract(self, x):
        """ Extract a design variable from an n-dimensional vector.

        Args:
            x (numpy.ndarray): A numpy array of length 1 containing the
                value to extract.

        Returns:
            float: The de-scaled value of x (from the original input space)

        """

        return (x * self.scale + self.shift)[0]


class IntegerEmbedder(Embedder):
    """ An Embedder class for integer variables that simply rescales and
    shifts all inputs to the box [0, 1], then de-scales and bins to the
    nearest integer upon extraction. """

    __slots__ = ['scale', 'scaled_des_tol', 'shift']

    def __init__(self, settings):
        """ Generate the scaling matrices for an integer variable.

        Args:
            settings (dict): Contains the following keys
                 * 'lb' (float or int, required): This specifies the lower
                   bound for the design variable. This value must be strictly
                   less than 'ub' (below) up to the tolerance (below).
                 * 'ub' (float or int, required): This specifies the upper
                   bound for the design variable. This value must be strictly
                   greater than 'lb' (above) up to the tolerance (below).
                 * 'des_tol' (float, optional): This specifies the design
                   tolerance for this variable, i.e., the minimum spacing
                   before two design values are considered equivalent up to
                   measurement error. If not specified, the
                   default value is 1.0e-8 * max(ub - lb, 1.0e-4).

        """

        # Error handling and extracting input bounds
        if isinstance(settings, dict):
            if 'lb' in settings:
                try:
                    lb = float(settings['lb'])
                except BaseException:
                    raise TypeError("lower bound must contain a float or int")
            else:
                raise KeyError("'lb' is a required key for continuous "
                               "design variables")
            if 'ub' in settings:
                try:
                    ub = float(settings['ub'])
                except BaseException:
                    raise TypeError("upper bound must contain a float or int")
            else:
                raise KeyError("'ub' is a required key for continuous "
                               "design variables")
            if lb >= ub:
                raise ValueError("lower bound must be strictly less than "
                                 "upper bound for all design variables ")
        else:
            raise TypeError("settings must be a dictionary")
        # Calculate the embedding as a sequence of matrix operations
        self.scale = jnp.ones(1) * (ub - lb)
        self.shift = jnp.ones(1) * lb
        self.scaled_des_tol = np.ones(1) * 0.5 / self.scale
        # Jit the embedder and extractor
        self.embed = jit(self._embed)
        self.extract = jit(self._extract)
        return

    def getLatentDesTols(self):
        """ Get the design tolerances along each dimension of the embedding.

        Returns:
            numpy.ndarray: array of design space tolerances after embedding

        """

        return self.scaled_des_tol

    def getFeatureDesTols(self):
        """ Get the design tolerances in the feature space (pre-embedding).

        Returns:
            float: the design tolerance in the feature space -- a value of
            0 indicates a discrete variable

        """

        return 0.0

    def getEmbeddingSize(self):
        """ Get dimension of embedded space.

        Returns:
            int: the dimension of the embedded space.

        """
        
        return 1

    def getInputType(self):
        """ Get the input type for this embedder.

        Returns:
            str: A numpy string representation of the input type from the
            feature space.

        """
        
        return 'i4'

    def getLowerBounds(self):
        """ Get a vector of lower bounds for the embedded space.

        Returns:
            numpy.ndarray: array of lower bounds in embedded space

        """

        return np.zeros(1)

    def getUpperBounds(self):
        """ Get a vector of upper bounds for the embedded space.

        Returns:
            numpy.ndarray: array of upper bounds in the embedded space

        """

        return np.ones(1)

    def _embed(self, x):
        """ Embed a design input as n-dimensional vector for ParMOO.

        Args:
            x (float or int): The value of the design variable to embed.

        Returns:
            numpy.ndarray: A numpy array of length 1 containing a
            rescaling of x

        """

        return jnp.minimum(jnp.maximum((x - self.shift) / self.scale, 0), 1)

    def _extract(self, x):
        """ Extract a design variable from an n-dimensional vector.

        Args:
            x (numpy.ndarray): A numpy array of length 1 containing the
                value to extract.

        Returns:
            float: The de-scaled value of x rounded to the nearest integer

        """

        return jnp.rint(x * self.scale + self.shift)[0]


class CategoricalEmbedder(Embedder):
    """ An Embedder class for categorical variables that uses matrix
    operations to embed a one-hot-encoding into a lower-dimensional
    space then round the result back to the nearest category. """

    __slots__ = ['cent', 'des_tol', 'ones', 'scale', 'shift',
                 'in_type', 'alabels', 'rot', 'zeros']

    def __init__(self, settings):
        """ Generate the encoding matrices for a categorical variable.

        Args:
            settings (dict): Contains the following keys
                 * 'levels' (int or list, required): The number of levels
                   for the variable (when int) or the names of each valid
                   category (when a list).

        """

        # Error handling and extracting input types
        jittable = True
        if isinstance(settings, dict) and 'levels' in settings:
            levels = settings['levels']
            if isinstance(levels, int):
                if levels < 2:
                    raise ValueError("a categorical variable must "
                                     "have at least 2 levels")
                n_lvls = levels
                self.alabels = jnp.array([i for i in range(levels)], dtype=int)
                self.in_type = 'i4'
            elif isinstance(levels, list):
                n_lvls = len(levels)
                if n_lvls < 2:
                    raise ValueError("a categorical variable must "
                                     "have at least 2 levels")
                if not (all([isinstance(li, int) for li in levels]) or
                        all([isinstance(li, str) for li in levels])):
                    raise TypeError("all levels of categorical variable "
                                    "must have the same type (int or str)")
                try:
                    self.alabels = jnp.array(levels)
                    self.in_type = 'i4'
                except TypeError:
                    self.alabels = np.array(levels)
                    self.in_type = 'U25'
                    jittable = False
            else:
                raise TypeError("settings['levels'] must be an int or list")
        elif isinstance(settings, dict):
            raise KeyError("'levels' key is missing for categorical variable")
        else:
            raise TypeError("settings must be a dictionary")
        # Calculate the embedding as a sequence of matrix operations via SVD
        n = n_lvls - 1
        self.cent = jnp.ones(n_lvls) / n_lvls
        u, sigma, vT = jnp.linalg.svd(jnp.eye(n_lvls) - self.cent)
        self.rot = vT[:n, :].T
        self.shift = jnp.min(u[:, :n]) * jnp.ones(n)
        self.scale = jnp.max(u[:, :n]) - self.shift
        self.des_tol = np.sqrt(0.5) / self.scale
        self.zeros = jnp.zeros(n_lvls)
        self.ones = jnp.ones(n_lvls)
        # Jit the embedder and extractor
        if jittable:
            self.embed = jit(self._embed)
            self.extract = jit(self._extract)
        else:
            self.embed = self._embed
            self.extract = self._extract
        return

    def getLatentDesTols(self):
        """ Get the design tolerances along each dimension of the embedding.

        Returns:
            numpy.ndarray: array of design space tolerances after embedding

        """

        return self.des_tol

    def getFeatureDesTols(self):
        """ Get the design tolerances in the feature space (pre-embedding).

        Returns:
            float: the design tolerance in the feature space -- a value of
            0 indicates a discrete variable

        """

        return 0.0

    def getEmbeddingSize(self):
        """ Get dimension of embedded space.

        Returns:
            int: the dimension of the embedded space.

        """
        
        return self.des_tol.size

    def getInputType(self):
        """ Get the input type for this embedder.

        Returns:
            str: A numpy string representation of the input type from the
            feature space.

        """
        
        return self.in_type

    def getLowerBounds(self):
        """ Get a vector of lower bounds for the embedded space.

        Returns:
            numpy.ndarray: array of lower bounds in embedded space

        """

        return np.zeros(self.des_tol.size)

    def getUpperBounds(self):
        """ Get a vector of upper bounds for the embedded space.

        Returns:
            numpy.ndarray: array of upper bounds in the embedded space

        """

        return np.ones(self.des_tol.size)

    def _embed(self, x):
        """ Embed a design input as n-dimensional vector for ParMOO.

        Args:
            x (int or str): a category from the list of categories or an int
                less than the number of categories

        Returns:
            numpy.ndarray: a 1d array containing the embedded category

        """

        xx = jnp.where(self.alabels == x, self.ones, self.zeros)
        xx1 = (jnp.dot(xx - self.cent, self.rot) - self.shift) / self.scale
        return jnp.minimum(jnp.maximum(xx1, 0), 1)

    def _extract(self, x):
        """ Extract a design variable from an n-dimensional vector.

        Args:
            x (numpy.ndarray): a 1d array containing the embedded category in
                vector form

        Returns:
            int or str: the extracted category

        """

        ind = jnp.dot(x * self.scale + self.shift, self.rot.T) + self.cent
        return self.alabels[jnp.argmax(ind).astype(int)]


class IdentityEmbedder(Embedder):
    """ An Embedder class for continuous variables that leaves them raw
    and un-scaled. """

    __slots__ = ['des_tol', 'lb', 'ub']

    def __init__(self, settings):
        """ Generate identity embedding for an unscaled design variable.

        Args:
            settings (dict): Contains the following keys
                 * 'lb' (float or int, required): This specifies the lower
                   bound for the design variable. This value must be strictly
                   less than 'ub' (below) up to the tolerance (below).
                 * 'ub' (float or int, required): This specifies the upper
                   bound for the design variable. This value must be strictly
                   greater than 'lb' (above) up to the tolerance (below).
                 * 'des_tol' (float, optional): This specifies the design
                   tolerance for this variable, i.e., the minimum spacing
                   before two design values are considered equivalent up to
                   measurement error. If not specified, the
                   default value is 1.0e-8 * max(ub - lb, 1.0e-4).

        """

        # Error handling and extracting input bounds
        if isinstance(settings, dict):
            if 'lb' in settings:
                try:
                    self.lb = float(settings['lb'])
                except BaseException:
                    raise TypeError("lower bound must contain a float or int")
            else:
                raise KeyError("'lb' is a required key for continuous "
                               "design variables")
            if 'ub' in settings:
                try:
                    self.ub = float(settings['ub'])
                except BaseException:
                    raise TypeError("upper bound must contain a float or int")
            else:
                raise KeyError("'ub' is a required key for continuous "
                               "design variables")
            if 'des_tol' in settings:
                self.des_tol = settings['des_tol']
                if not isinstance(self.des_tol, float):
                    raise TypeError("design tolerance must be a float type")
                if self.des_tol <= 0:
                    raise ValueError("design tolerance must be strictly "
                                     "greater than 0")
            else:
                self.des_tol = 1.0e-8 * max(self.ub - self.lb, 1.0e-4)
            if self.lb + self.des_tol > self.ub:
                raise ValueError("lower bound must be strictly less than "
                                 "upper bound for all design variables "
                                 "up to the design tolerance")
        else:
            raise TypeError("settings must be a dictionary")
        # Jit the embedder and extractor
        self.embed = jit(self._embed)
        self.extract = jit(self._extract)
        return

    def getLatentDesTols(self):
        """ Get the design tolerances along each dimension of the embedding.

        Returns:
            numpy.ndarray: array of design space tolerances after embedding

        """

        return self.des_tol * np.ones(1)

    def getFeatureDesTols(self):
        """ Get the design tolerances in the feature space (pre-embedding).

        Returns:
            float: the design tolerance in the feature space -- a value of
            0 indicates a discrete variable

        """

        return float(self.des_tol)

    def getEmbeddingSize(self):
        """ Get dimension of embedded space.

        Returns:
            int: the dimension of the embedded space.

        """
        
        return 1

    def getInputType(self):
        """ Get the input type for this embedder.

        Returns:
            str: A numpy string representation of the input type from the
            feature space.

        """
        
        return 'f8'

    def getLowerBounds(self):
        """ Get a vector of lower bounds for the embedded space.

        Returns:
            numpy.ndarray: array of lower bounds in embedded space

        """

        return self.lb * np.ones(1)

    def getUpperBounds(self):
        """ Get a vector of upper bounds for the embedded space.

        Returns:
            numpy.ndarray: array of upper bounds in the embedded space

        """

        return self.ub * np.ones(1)

    def _embed(self, x):
        """ Embed a design input as n-dimensional vector for ParMOO.

        Args:
            x (float or int): The value of the design variable to embed.

        Returns:
            numpy.ndarray: A numpy array of length 1 containing x

        """

        return jnp.ones(1) * x

    def _extract(self, x):
        """ Extract a design variable from an n-dimensional vector.

        Args:
            x (numpy.ndarray): A numpy array of length 1 containing the
                value to extract.

        Returns:
            float: The value of x (but as a scalar, not a singleton array)

        """

        return x[0]
