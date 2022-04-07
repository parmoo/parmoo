""" Abstract base class (ABC) for simulation function outputs.

Defines an ABC for the callable ``sim_func`` class.

"""

import numpy as np
from abc import ABC


class sim_func(ABC):
    """ Abstract base class (ABC) for simulation function outputs.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method is already implemented, and is the constructor.

    The ``__call__`` method is left to be implemented, and performs the
    simulation evaluation.

    """

    __slots__ = ['n', 'des_type', 'use_names']

    def __init__(self, des):
        """ Constructor for simulation functions.

        Args:
            des (list, tuple, or int): Either the numpy.dtype of the
                design variables (list or tuple) or the number of design
                variables (assumed to all be continuous, unnamed).

        """

        # Decide whether to use named variables
        self.use_names = False
        # Try to read design variable type
        try:
            self.des_type = np.dtype(des)
        except TypeError:
            if isinstance(des, int):
                self.des_type = np.dtype(("f8", (des, )))
            else:
                raise TypeError("des must contain a valid numpy.dtype or int")
        if self.des_type.names is not None:
            self.n = len(self.des_type.names)
            self.use_names = True
        else:
            self.n = sum(self.des_type.shape)
        if self.n == 0:
            raise ValueError("An illegal des_type was given")
        return

    def __call__(self, x):
        """ Make sim_func objects callable.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

        Returns:
            numpy.ndarray: The output of this simulation for the input x.

        """

        raise NotImplementedError("This method has not yet been implemented")
