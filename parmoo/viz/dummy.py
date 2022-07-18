""" This module contains testing utilities.

The functions are:

  * ``dummyFunction(moop)`` -- place functions here for testing


"""

from .plot import (scatter,
                   radar,
                   parallel_coordinates)


def vizTest(moop):
    """ Dummy function for development purposes.

    Functions to be tested in examples should be placed here.

    Args:
        moop (MOOP): A ParMOO MOOP for testing functions on.

    Returns:
        None

    """

    scatter(moop, db='pf', hot_reload=False, port='http://127.0.0.1:8050/')
    # parallel_coordinates(moop, port='http://127.0.0.1:8050/')
    # radar(moop, db='pf', port='http://127.0.0.1:8050/')
