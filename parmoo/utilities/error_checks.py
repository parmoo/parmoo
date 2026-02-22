""" This module contains several auxiliary functions to aide in error handling.

These functions may also be of external interest. They are:
 * `check_names(name, des_schema, sim_schema, obj_schema, con_schema)`
 * `gradient_error(arg1, arg2)`
 * `get_hp(key, hyperparams, expected_type, is_legal, default_value)`
 * `xerror(o, lb, ub, hyperparams)`

"""

import numpy as np


def check_names(name, *args):
    """ Typecheck the input arguments for a new variable name.

    Args:
        name (str): A str that could be used as the variable name.

        *args (list of tuples): 1 or more lists of existing variable names to
            check against in order to guarantee that name is unique.

    """

    if not isinstance(name, str):
        raise TypeError("Every variable name must be a string type")
    for arg in args:
        if any([name == ni[0] for ni in arg]):
            raise ValueError(f"The variable name {name} is already in use")


def gradient_error(arg1, arg2):
    """ Raises an error, warning users that the gradient is undefined.

    Args:
        arg1 (un-used): Here to match the interface of a gradient or bwd pass
            function.
        arg2 (un-used): Here to match the interface of a gradient or bwd pass
            function.

    """

    raise ValueError("1 or more grad func is undefined")


def get_hp(key, hyperparams, expected_type, is_legal, default_value):
    """ Search for key in the hyperparams dictionary.

    Args:
        key (hashable): The key to search for
        hyperparams (dict): The hyperparameters dictionary to search
        expected_type (type): The expected type for hyperparams[key]
        is_legal (callable): A callable object to check the legalality
            of hyperparams[key]. Returns True if and only if hyperparams[key]
            contains a legal value
        default_value (any): The default value to assign when key is not in
            hyperparams. When the value is None, key is required in
            hyperparams.

    """

    if not isinstance(hyperparams, dict):
        raise TypeError("hyperparams must be a Python dict type")
    if key not in hyperparams and default_value is None:
        raise KeyError(
            f"'{key}' is a required key in hyperparams for one or more of "
            "the methods used"
        )
    elif key not in hyperparams:
        return default_value
    if not isinstance(hyperparams[key], expected_type):
        raise TypeError(
            f"hyperparams[{key}] must be an instance of "
            f"{expected_type.__name__}"
        )
    if not is_legal(hyperparams[key]):
        raise ValueError(
            f"hyperparams[{key}] contains an illegal value"
        )
    return hyperparams[key]


def xerror(o=1, lb=None, ub=None, hyperparams=None):
    """ Typecheck the input arguments for a class interface.

    Args:
        o (int): The number of objectives should be an int greater than or
            equal to 1.

        lb (np.ndarray): The lower bounds should be a 1d array.

        ub (np.ndarray): The upper bounds should be a 1d array with the same
            length as lb, and must satisfy lb[:] < ub[:].

        hyperparams (dict): The hyperparameters must be supplied in a
            dictionary.

    """

    # Assign default values, if needed
    if lb is None:
        lb = np.zeros(1)
    if ub is None:
        ub = np.ones(1)
    if hyperparams is None:
        hyperparams = {}
    # Check the objective count
    if not isinstance(o, int):
        raise TypeError("o must be an integer")
    if o < 1:
        raise ValueError("o must be positive")
    # Check that the bounds are legal
    if not isinstance(lb, np.ndarray):
        raise TypeError("lb must be a numpy array")
    if not isinstance(ub, np.ndarray):
        raise TypeError("ub must be a numpy array")
    if np.size(ub) != np.size(lb):
        raise ValueError("the dimensions of lb and ub must match")
    if np.any(lb >= ub):
        raise ValueError("ub must be strictly greater than lb")
    # Check the hyperparams dict
    if not isinstance(hyperparams, dict):
        raise TypeError("hyperparams must be a dictionary")
