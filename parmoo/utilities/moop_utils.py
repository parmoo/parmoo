""" This module contains several auxiliary functions used throughout ParMOO.

These functions may also be of external interest. They are:
 * `lex_leq(a, b)`
 * `updatePF(data, nondom)`
 * `to_array(x, dtype)`
 * `from_array(x, dtype)`
 * `approx_equal(x1, x2, des_tols)`

"""

from jax import numpy as jnp
import numpy as np


def lex_leq(a, b):
    """ Lexicographically compare two vectors from back to front.

    Check whether the vector a is lexicographically less than or equal to b,
    starting from the last element and working back to the first element.

    Args:
        a (numpy.ndarray): The first vector to compare.

        b (numpy.ndarray): The second vector to compare.

    Returns:
        Boolean: Whether a <= b in the lexicographical sense.

    """
    if a.size < 1 or b.size < 1:
        return True
    elif a[-1] < b[-1]:
        return True
    elif a[-1] > b[-1]:
        return False
    else:
        return lex_leq(a[:-1], b[:-1])


def updatePF(data, nondom):
    """ Update the Pareto front and efficient set by resorting.

    Returns:
        dict: A dictionary containing a discrete approximation of the
        Pareto front and efficient set.
         * f_vals (numpy.ndarray): A list of nondominated points
           discretely approximating the Pareto front.
         * x_vals (numpy.ndarray): A list of corresponding
           efficient design points.
         * c_vals (numpy.ndarray): A list of corresponding
           constraint satisfaction scores, all less than or equal to 0.

    """

    # Lexicographically sort all new points by 'f_vals'
    new_ind = np.lexsort(data['f_vals'][:, :].transpose())
    # Get problem size and allocate output array
    if 'f_vals' in nondom:
        nondom_len = nondom['f_vals'].shape[0]
        nondom_out = {'x_vals': np.zeros((nondom['x_vals'].shape[0]
                                          + data['x_vals'].shape[0],
                                          data['x_vals'].shape[1])),
                      'f_vals': np.zeros((nondom['f_vals'].shape[0]
                                          + data['f_vals'].shape[0],
                                          data['f_vals'].shape[1])),
                      'c_vals': np.zeros((nondom['c_vals'].shape[0]
                                          + data['c_vals'].shape[0],
                                          data['c_vals'].shape[1]))}
    else:
        nondom_len = 0
        nondom_out = {'x_vals': np.zeros((data['x_vals'].shape[0],
                                          data['x_vals'].shape[1])),
                      'f_vals': np.zeros((data['f_vals'].shape[0],
                                          data['f_vals'].shape[1])),
                      'c_vals': np.zeros((data['c_vals'].shape[0],
                                          data['c_vals'].shape[1]))}
    # Get total number of points to merge
    n_dat = nondom_len + data['f_vals'].shape[0]
    # Merge sorted lists
    j = 0
    k = 0
    for i in new_ind:
        # Add all points from nondom that are lexicographically first
        isNonDom = True
        while j < nondom_len and isNonDom:
            if lex_leq(nondom['f_vals'][j, :], data['f_vals'][i, :]):
                nondom_out['x_vals'][k, :] = nondom['x_vals'][j, :]
                nondom_out['f_vals'][k, :] = nondom['f_vals'][j, :]
                nondom_out['c_vals'][k, :] = nondom['c_vals'][j, :]
                j += 1
                k += 1
            else:
                isNonDom = False
        # Check for constraint violations
        if np.any(data['c_vals'][i, :] > 0.0):
            n_dat -= 1
        else:
            # If no constraints violated, append to the output
            nondom_out['x_vals'][k, :] = data['x_vals'][i, :]
            nondom_out['f_vals'][k, :] = data['f_vals'][i, :]
            nondom_out['c_vals'][k, :] = data['c_vals'][i, :]
            k += 1
    # Add remaining nondominated points from list
    while j < nondom_len:
        nondom_out['x_vals'][k, :] = nondom['x_vals'][j, :]
        nondom_out['f_vals'][k, :] = nondom['f_vals'][j, :]
        nondom_out['c_vals'][k, :] = nondom['c_vals'][j, :]
        j += 1
        k += 1
    # Loop over all points and look for nondominated points
    ndpts = 0  # counter for number of nondominated points
    for i in range(n_dat):
        # Check if data['f_vals'][i] is nondominated
        if np.all(np.any(nondom_out['f_vals'][i, :] <
                         nondom_out['f_vals'][:ndpts, :], axis=1)):
            # Swap entries at indices i and ndpts
            nondom_out['f_vals'][(i, ndpts), :] = \
                nondom_out['f_vals'][(ndpts, i), :]
            nondom_out['x_vals'][(i, ndpts), :] = \
                nondom_out['x_vals'][(ndpts, i), :]
            nondom_out['c_vals'][(i, ndpts), :] = \
                nondom_out['c_vals'][(ndpts, i), :]
            # Increment ndpts
            ndpts += 1
    # Return the solutions in a new dictionary
    return {'x_vals': nondom_out['x_vals'][:ndpts, :],
            'f_vals': nondom_out['f_vals'][:ndpts, :],
            'c_vals': nondom_out['c_vals'][:ndpts, :]}


def to_array(x, dtype):
    """ Unroll a ParMOO variable of the given dtype into a flat numpy array.

    Args:
        x (dict or numpy structured array): A ParMOO variable, which needs to
            be unrolled in a ndarray for convenience.

        dtype (numpy.dtype): The numpy.dtype of x.

    Returns:
        ndarray: a 1D array containing the values in x unrolled into an
        ndarray format, ordered by their order in the dtype.names.

    """

    xx = []
    for namei in dtype.names:
        xx.append(jnp.array([x[namei]]))
    return jnp.concatenate(xx, axis=None)


def from_array(x, dtype):
    """ Roll a flat ndarray back up into a ParMOO variable of the given dtype.

    Args:
        x (ndarray): A 1D array whose length matches the sum of the lengths of
            all fields for the dtype.

        dtype (numpy.dtype): The numpy.dtype of the needed output.

    Returns:
        dict: a ParMOO dictionary representation of x whose keys match the
        names in dtype and contains the values in x.

    """

    x1 = jnp.asarray(x).reshape((max(x.size, 1), ))
    xx = {}
    istart = 0
    for namei in dtype.names:
        if len(dtype[namei].shape) > 0:
            iend = istart + dtype[namei].shape[0]
        else:
            iend = istart + 1
        xx[namei] = x1[istart:iend].copy()
        istart = iend
    return xx


def approx_equal(x1, x2, des_tols):
    """ Check if two dictionaries contain equal values up to the tolerance.

    Note:  This function allows that a value in x1/x2 could be non numeric, in
    which case exact equality is checked. This is triggered if the
    corresponding value in des_tols <= 0.

    Args:
        x1 (dict): A dictionary of design variable names and corresponding
            values.
        x2 (dict): A dictionary of design variable names and corresponding
            values. Must contain all the keys in x1, but could contain
            additional keys that are not in x1.
        des_tols (dict): A dictionary of design variable names and
            corresponding tolerances. Keys must match those in x1 and x2.
            Values must be numerical. Any value less than or equal to zero
            results in direct comparison for the corresponding

    """

    for key in x1:
        if (
            (des_tols[key] > 0 and abs(x1[key] - x2[key]) >= des_tols[key]) or
            (des_tols[key] <= 0 and x1[key] != x2[key])
        ):
            return False
    return True
