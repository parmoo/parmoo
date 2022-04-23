""" Abstract base class (ABC) for objective functions.

Defines an ABC for the callable ``obj_func`` class.

"""

import numpy as np
from abc import ABC


class obj_func(ABC):
    """ Abstract base class (ABC) for objective function outputs.

    Contains 2 methods:
     * ``__init__(des, sim)``
     * ``__call__(x, sx, der=0)``

    The ``__init__`` method is already implemented, and is the constructor.

    The ``__call__`` method is left to be implemented, and performs the
    objective evaluation.

    """

    __slots__ = ['n', 'm', 'des_type', 'sim_type', 'use_names']

    def __init__(self, des, sim):
        """ Constructor for objective functions.

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
        if self.n <= 0:
            raise ValueError("An illegal des_type was given")
        # Try to read simulation variable type
        if (not self.use_names) and isinstance(sim, int):
            self.sim_type = np.dtype(("f8", (sim, )))
            self.m = sim
        else:
            try:
                self.sim_type = np.dtype(sim)
            except TypeError:
                if isinstance(sim, int):
                    self.sim_type = np.dtype(("f8", (sim, )))
                else:
                    raise TypeError("sim must contain a numpy.dtype or int")
            if self.sim_type.names is not None:
                self.m = 0
                for name in self.sim_type.names:
                    if len(self.sim_type[name].shape) >= 1:
                        self.m += sum(self.sim_type[name].shape)
                    else:
                        self.m += 1
            else:
                if len(self.sim_type.shape) >= 1:
                    self.m = sum(self.sim_type.shape)
                else:
                    self.m = 1
            if (self.sim_type.names is not None) != self.use_names:
                raise ValueError("When using names for des_type, sim_type " +
                                 "must also give named fields")
        if self.m <= 0:
            raise ValueError("An illegal sim_type was given")
        return

    def __call__(self, x, sx, der=0):
        """ Make obj_func objects callable.

        Args:
            x (numpy.array): A numpy.ndarray (unnamed) or numpy structured
                array (named), containing the design point to evaluate.

        Returns:
            float: The output of this objective for the input x.

        """

        raise NotImplementedError("This method has not yet been implemented")
