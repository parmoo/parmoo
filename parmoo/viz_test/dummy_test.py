""" This module contains testing utilities.

The functions are:

  * ``dummyFunction(moop)`` -- place functions here for testing


"""

from .plot import *


def vizTest(moop):
    """ Dummy function for development purposes.

    Functions to be tested in examples should be placed here.

    Args:
        moop (MOOP): A ParMOO MOOP for testing functions on.

    Returns:
        None

    """

    scatter(moop, db='pf', hot_reload=False)
    # parallel_coordinates(moop, db='obj', objectives_only=True, export='html')
    # radar(moop)
    # db = moop.getPF(format='pandas')
    # db = moop.getPF(format='ndarray')
    # db = moop.getObjectiveData(format='pandas')
    # db = moop.getObjectiveData()
    # db = moop.getSimulationData(format='ndarray')
    # db = moop.getSimulationData(format='pandas')