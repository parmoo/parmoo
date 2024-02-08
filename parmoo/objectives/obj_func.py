""" Abstract base class (ABC) for objective functions.

Defines an ABC for the callable ``ObjectiveFunction`` class.

"""

import numpy as np
from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
    """ Abstract base class (ABC) for objective function outputs.

    Contains 2 methods:
     * ``__init__(des_type, sim_type)``
     * ``__call__(x, sx)``

    The ``__init__`` method is already implemented, and is the constructor.

    The ``__call__`` method is left to be implemented, and performs the
    objective evaluation.

    """

    __slots__ = ['n', 'm', 'des_type', 'sim_type']

    def __init__(self, des_type, sim_type):
        """ Constructor for objective functions.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (list or tuple): The numpy.dtype of the simulation
                outputs.

        """

        # Try to read design variable type
        try:
            self.des_type = np.dtype(des_type)
        except TypeError:
            raise TypeError("des_type must contain a valid numpy.dtype")
        self.n = len(self.des_type.names)
        if self.n <= 0:
            raise ValueError("An illegal des_type was given")
        # Try to read simulation variable type
        try:
            self.sim_type = np.dtype(sim_type)
        except TypeError:
            raise TypeError("sim_type must contain a valid numpy.dtype")
        self.m = 0
        for name in self.sim_type.names:
            self.m += np.maximum(np.sum(self.sim_type[name].shape), 1)
        if self.m <= 0:
            raise ValueError("An illegal sim_type was given")
        return

    @abstractmethod
    def __call__(self, x, sx):
        """ Make ObjectiveFunction objects callable.

        Args:
            x (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of a design point to evaluate.

            sx (dict or structured array): A Python dictionary or numpy
                structured array containing the names (as keys) and
                corresponding values of the simulation output(s) at x.

        Returns:
            float: The output of this objective for the input x.

        """
