""" The abstract base class (ABC) for GlobalSearch methods. """

from abc import ABC, abstractmethod


class GlobalSearch(ABC):
    """ ABC describing global search techniques.

    This class contains the following methods.
     * ``startSearch(lb, ub)``
     * ``resumeSearch()``
     * ``save(filename)``
     * ``load(filename)``

    """

    @abstractmethod
    def __init__(self, o, lb, ub, hyperparams):
        """ Constructor for the GlobalSearch class.

        Args:
            o (int): The number of objectives.

            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

            hyperparams (dict): A dictionary of hyperparameters for the
                global search. It may contain any inputs specific to the
                search algorithm.

        Returns:
            GlobalSearch: A new GlobalSearch object.

        """

    @abstractmethod
    def startSearch(self, lb, ub):
        """ Begin a new global search.

        Args:
            lb (ndarray): A 1D array of lower bounds for the design space.

            ub (ndarray): A 1D array of upper bounds for the design space.

        Returns:
            ndarray: A 2D design matrix.

        """

    def resumeSearch(self):
        """ Resume a global search.

        Returns:
            ndarray: A 2D design matrix.

        """

        raise NotImplementedError("This class method has not been implemented")

    def save(self, filename):
        """ Save important data from this class so that it can be reloaded.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the load method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data should be saved.

        """

        raise NotImplementedError("This class method has not been implemented")

    def load(self, filename):
        """ Reload important data into this class after a previous save.

        Note: If this function is left unimplemented, ParMOO will reinitialize
        a fresh instance after a save/load. If this is the desired behavior,
        then this method and the save method need not be implemented.

        Args:
            filename (string): The relative or absolute path to the file
                where all reload data has been saved.

        """

        raise NotImplementedError("This class method has not been implemented")
