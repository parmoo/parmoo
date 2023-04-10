""" This module contains some simple simulation function implementations
for testing purposes.

The full list of simulation functions in this module includes the kernel
functions:
 * ``LinearSim``
 * ``QuadraticSim``

"""

from parmoo.simulations import sim_func
from parmoo.util import unpack
import numpy as np


class LinearSim(sim_func):
    """ Class defining a simulation function with linear form.

    result[j] = sum_{i=1}^n (x_i - floor(j / m) / o),  j = 0, ..., m*o - 1

    Contains 2 methods:
     * ``__init__(des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the linear kernel.

    """

    def __init__(self, des, num_obj=3, num_r=5):
        """ Constructor for LinearSim class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used to calculate the value of linear model.

            num_r (int, optional): The number of simulation outputs
                (residuals) per objective. Defaults to 5.

        """

        super().__init__(des)
        self.o = num_obj
        self.m = num_r
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

        Returns:
            numpy.ndarray: The output of this simulation for the input x.

        """

        # Extract x into xx, if names are used
        xx = unpack(x, self.des_type)
        # Calculate output
        result = np.zeros(self.m * self.o)
        for j in range(self.m * self.o):
            result[j] = np.sum(xx[:] - (j // self.m) / self.o) / self.n
        return result


class QuadraticSim(sim_func):
    """ Class defining a simulation function with quadratic form.

    result[j] = sum_{j=mo}^{mo+m} (sum_{i=1}^n (x_i - floor(j / m) / o)) ** 2,
                j = 0, ..., o

    Contains 2 methods:
     * ``__init__(des, num_obj)``
     * ``__call__(x)``

    The ``__init__`` method inherits from the sim_func ABC.

    The ``__call__`` method performs an evaluation of the quadratic kernel.

    """

    def __init__(self, des, num_obj=3, num_r=5):
        """ Constructor for QuadraticModel class.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            num_obj (int, optional): The number of objectives, which is
                used to calculate the value of quadratic model.

            num_r (int, optional): The number of residual values in quadratic
                form per objective (m in formula above). Defaults to 5.

        """

        super().__init__(des)
        self.o = num_obj
        self.m = num_r
        return

    def __call__(self, x):
        """ Define simulation evaluation.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

        Returns:
            numpy.ndarray: The output of this simulation for the input x.

        """

        # Extract x into xx, if names are used
        xx = unpack(x, self.des_type)
        # Calculate output
        result = np.zeros(self.o)
        temp = np.zeros(self.m * self.o)
        for j in range(self.m * self.o):
            temp[j] = np.sum(xx[:] - (j // self.m) / self.o) / self.n
        for j in range(self.o):
            result[j] = np.sum(temp[(self.m * j):(self.m * (j + 1))]**2)
        return result
