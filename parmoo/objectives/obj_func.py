""" Abstract base class (ABC) for simulation function outputs.

Defines an ABC for the callable ``sim_func`` class.

"""

import numpy as np
from abc import ABC


class obj_func(ABC):
    """ Abstract base class (ABC) for objective function outputs.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x)``

    The ``__init__`` method is already implemented, and is the constructor.

    The ``__call__`` method is left to be implemented, and performs the
    simulation evaluation.

    """

    __slots__ = ['n', 'm', 'des_type', 'sim_type', 'use_names']

    def __init__(self, des, sim):
        """ Constructor for simulation functions.

        Args:
            des (np.dtype or int): Either the numpy.dtype of the
                design variables or the number of design variables,
                assumed to all be continuous and unnamed.

            sim (list, tuple, or int): Either the numpy.dtype of the
                simultation outputs (list or tuple) or the number of
                simulation outputs (assumed to all be continuous,
                unnamed).

        """

        # Decide whether to use named variables
        self.use_names = False
        # Try to read design variable type
        if isinstance(des, list) or isinstance(des, tuple):
            try:
                np.zeros(1, dtype=des)
                self.des_type = des
                self.n = len(des)
            except TypeError:
                raise TypeError("des must contain a valid numpy.dtype")
            if isinstance(des, list):
                self.use_names = True
        elif isinstance(des, int):
            if des > 0:
                self.n = des
                self.des_type = ("f8", self.n)
            else:
                raise ValueError("des must be a positive number")
        else:
            raise TypeError("des must be a list, tuple, or int")
        # Try to read simulation output type
        if isinstance(sim, list) and self.use_names:
            try:
                np.zeros(1, dtype=sim)
                self.sim_type = sim
                self.m = len(sim)
            except TypeError:
                raise TypeError("sim must contain a valid numpy.dtype")
        elif isinstance(sim, tuple) and not self.use_names:
            try:
                np.zeros(1, dtype=sim)
                self.sim_type = sim
                self.m = len(sim)
            except TypeError:
                raise TypeError("sim must contain a valid numpy.dtype")
        elif isinstance(sim, int) and not self.use_names:
            if sim > 0:
                self.m = sim
                self.sim_type = ("f8", self.m)
            else:
                raise ValueError("sim must be a positive number")
        else:
            raise TypeError("sim must be a list, tuple, or int, and " +
                            "match the type of des")
        return

    def __call__(self, x, der=0):
        """ Make obj_func objects callable.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

        Returns:
            numpy.ndarray: The output of this simulation for the input x.

        """

        raise NotImplementedError("This method has not yet been implemented")
