""" An abstract base class (ABC) for ParMOO objective functions. """

from abc import ABC, abstractmethod
import numpy as np


class ObjectiveFunction(ABC):
    """ ABC defining ParMOO objective functions.

    Extend this class to create a callable object that matches ParMOO can
    use as a objective function, such as an Objective or its Gradient.

    Contains 2 methods:
     * ``__init__(des_type, sim_type)``
     * ``__call__(x, sx)``

    The ``__init__`` method is already implemented, and is the constructor.
    It can be overwritten if additional inputs (besides the design variable
    and simulation output types) are needed.

    The ``__call__`` method is left to be implemented, and performs the
    objective function evaluation.

    """

    __slots__ = ['n', 'm', 'des_type', 'sim_type']

    def __init__(self, des_type, sim_type):
        """ Constructor for ObjectiveFunction class.

        Args:
            des_type (np.dtype): The numpy.dtype of the design variables.

            sim_type (np.dtype): The numpy.dtype of the simulation outputs.

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
            float: The output of this objective for the input x (when
            defining objective functions).

            OR

            dict, dict: Dictionaries with the same keys as x and sx, whose
            corresponding values contain the partials with respect to x and
            sx, respectively.

        """
