""" Abstract base class (ABC) for simulation function outputs.

Defines an ABC for the callable ``sim_func`` class.

"""

import numpy as np
from abc import ABC, abstractmethod


class sim_func(ABC):
    """ Abstract base class (ABC) for simulation function outputs.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method is already implemented, and is the constructor.

    The ``__call__`` method is left to be implemented, and performs the
    simulation evaluation.

    """

    __slots__ = ['n', 'des_type']

    def __init__(self, des):
        """ Constructor for simulation functions.

        Args:
            des (numpy.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

        """

        # Try to read design variable type
        try:
            self.des_type = np.dtype(des)
        except TypeError:
            raise TypeError("des must contain a valid numpy.dtype")
        self.n = len(self.des_type.names)
        if self.n == 0:
            raise ValueError("An illegal des_type was given")
        return

    @abstractmethod
    def __call__(self, x):
        """ Make sim_func objects callable.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

        Returns:
            numpy.ndarray: The output of this simulation for the input x.

        """
