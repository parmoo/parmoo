""" This module contains several auxiliary functions used throughout ParMOO.

These functions may also be of external interest. They are:
 * xerror(o, lb, ub, hyperparams)
 * check_sims(n, arg1, arg2, ...)
 * lex_leq(a, b)
 * updatePF(data, nondom)
 * unpack(x, dtype)

"""

import numpy as np
from parmoo import structs
import inspect


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
    if isinstance(o, int):
        if o < 1:
            raise ValueError("o must be positive")
    else:
        raise TypeError("o must be an integer")
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
    return


def check_sims(n, *args):
    """ Check simulation dictionaries for bad input.

    Args:
        n (int): The dimension of the design space. Used for confirming
            any simulation databases provided in args.

        *args (dict): An unpacked array of dictionaries, each specifying
            one of the simulations. The following keys are used:
             * name (String, optional): The name of this simulation
               (defaults to "sim" + str(i), where i = 1, 2, 3, ... for
               the first, second, third, ... simulation added to the
               MOOP).
             * m (int): The number of outputs for this simulation.
             * sim_func (function): An implementation of the simulation
               function, mapping from R^n -> R^m. The interface should
               match:
               `sim_out = sim_func(x, der=False)`,
               where `der` is an optional argument specifying whether
               to take the derivative of the simulation. Unless
               otherwise specified by your solver, `der` is always
               omitted by ParMOO's internal structures, and need not
               be implemented.
             * search (GlobalSearch): A GlobalSearch object for performing
               the initial search over this simulation's design space.
             * surrogate (SurrogateFunction): A SurrogateFunction object
               specifying how this simulation's outputs will be modeled.
             * des_tol (float): The tolerance for this simulation's
               design space; a new design point that is closer than
               des_tol to a point that is already in this simulation's
               database will not be reevaluated.
             * hyperparams (dict): A dictionary of hyperparameters, which
               will be passed to the surrogate and search routines.
               Most notably, search_budget (int) can be specified
               here.
             * sim_db (dict, optional): A dictionary of previous
               simulation evaluations. When present, contains:
                * x_vals (np.ndarray): A 2d array of pre-evaluated
                  design points.
                * s_vals (np.ndarray): A 2d array of corresponding
                  simulation outputs.
                * g_vals (np.ndarray): A 3d array of corresponding
                  Jacobian values. This value is only needed
                  if the provided SurrogateFunction uses gradients.

    """

    # Iterate through args to check each sim
    s = 0
    m = 0
    for arg in args:
        if isinstance(arg, dict):
            if 'name' in arg:
                if not isinstance(arg['name'], str):
                    raise TypeError("sims[" + str(s) + "]['name']"
                                    + " must be a string when present")
            # Check the number of sim outputs
            if 'm' in arg:
                if isinstance(arg['m'], int):
                    if arg['m'] > 0:
                        m = arg['m']
                    else:
                        raise ValueError("sims[" + str(s) + "]['m']"
                                         + " must be greater than zero")
                else:
                    raise TypeError("sims[" + str(s) + "] : 'm'"
                                    + " must be an int")
            else:
                raise AttributeError("sims[" + str(s)
                                     + "] is missing the key 'm'")
            # Get the hyperparameter dict
            if 'hyperparams' in arg:
                if not isinstance(arg['hyperparams'], dict):
                    raise TypeError("sims[" + str(s)
                                    + "]: 'hyperparams'"
                                    + " key must be a dict when present")
            # Check the search technique
            if 'search' in arg:
                try:
                    assert(isinstance(arg['search'](m, np.zeros(n),
                                                    np.ones(n), {}),
                                      structs.GlobalSearch))
                except BaseException:
                    raise TypeError("sims[" + str(s) + "]['search']"
                                    + " must be a derivative of the"
                                    + " GlobalSearch abstract class")
            else:
                raise AttributeError("sims[" + str(s) + "] is missing"
                                     + " the key 'search'")
            # Check the des_tol, if present
            if 'des_tol' in arg:
                if isinstance(arg['des_tol'], float):
                    if arg['des_tol'] <= 0.0:
                        raise ValueError("sims[" + str(s)
                                         + "]['des_tol']"
                                         + " must be greater than 0")
                else:
                    raise TypeError("sims[" + str(s)
                                    + "]['des_tol'] must"
                                    + " be a float")
            # Get the surrogate function
            if 'surrogate' in arg:
                try:
                    if not isinstance(arg['surrogate'](m, np.zeros(n),
                                                       np.ones(n), {}),
                                      structs.SurrogateFunction):
                        raise TypeError("sims[" + str(s) + "] :"
                                        + " 'surrogate' must be a"
                                        + " derivative of the"
                                        + " SurrogateFunction abstract"
                                        + " class")
                except BaseException:
                    raise TypeError("sims[" + str(s)
                                    + "]['surrogate']"
                                    + " must be a derivative of the"
                                    + " SurrogateFunction abstract class")
            else:
                raise AttributeError("sims[" + str(s) + "] is missing"
                                     + " the key 'surrogate'")
            # Get the simulation function
            if 'sim_func' in arg:
                if callable(arg['sim_func']):
                    if len(inspect.signature(arg['sim_func']).parameters) \
                       != 1 and \
                       len(inspect.signature(arg['sim_func']).parameters) \
                       != 2:
                        raise ValueError("sims[" + str(s) + "]["
                                         + "'sim_func'] must accept"
                                         + " one or two inputs")
                else:
                    raise TypeError("sims[" + str(s)
                                    + "]['sim_func']"
                                    + " must be callable")
            else:
                raise AttributeError("sims[" + str(s) + "] is missing"
                                     + " the key 'sim_func'")
            # Get the starting database, if present
            if 'sim_db' in arg:
                if isinstance(arg['sim_db'], dict):
                    if 'x_vals' in arg['sim_db'] and \
                       's_vals' in arg['sim_db']:
                        try:
                            # Cast arg['sim_db'] contents to np.ndarrays
                            xvals = np.asarray(arg['sim_db']['x_vals'])
                            svals = np.asarray(arg['sim_db']['s_vals'],
                                               dtype=np.float64)
                        except BaseException:
                            raise TypeError("sims[" + str(s)
                                            + "]['sim_db']"
                                            + "['x_vals'] or sims["
                                            + str(s) + "]['sim_db']"
                                            + "['s_vals'] could not be"
                                            + " cast as a numpy array")
                        # Check the resulting dimensions
                        if xvals.size != 0 and svals.size != 0:
                            if xvals.ndim > 1:
                                if xvals.shape[1] != n:
                                    raise ValueError("sims[" + str(s)
                                                     + "]['sim_db']['x_vals']"
                                                     + " does not have"
                                                     + " n cols per row")
                                elif xvals.shape[0] != svals.shape[0]:
                                    raise ValueError("sims[" + str(s)
                                                     + "]['sim_db']['x_vals']"
                                                     + " does not have same"
                                                     + " number of rows as"
                                                     + " sims[" + str(s)
                                                     + "]['sim_db']['s_vals']")
                            if svals.shape[1] != m:
                                raise ValueError("sims[" + str(s)
                                                 + "]['sim_db']['s_vals']"
                                                 + " does not have"
                                                 + " sims[" + str(s)
                                                 + "]['m'] cols per row")
                        elif xvals.size != svals.size:
                            raise ValueError("sims[" + str(s)
                                             + "]['sim_db']['x_vals']"
                                             + " cannot be empty when"
                                             + " sims[" + str(s)
                                             + "]['sim_db']['s_vals']"
                                             + " is nonempty, and vice"
                                             + " versa")
                    elif 'x_vals' in arg['sim_db'] or \
                         's_vals' in arg['sim_db']:
                        raise AttributeError("sims[" + str(s) + "] cannot"
                                             + " contain a sim_db with"
                                             + " 'x_vals' but not 's_vals'"
                                             + " or vice versa")
                else:
                    raise TypeError("sims[" + str(s) + "]['sim_db']"
                                    + " must be a dict")
            s += 1
        else:
            raise TypeError("sims[" + str(s) + "] is not a dict")
        return


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


def unpack(x, dtype):
    """ Unpack an input vector of given dtype into a numpy.ndarray.

    Args:
        x (numpy.ndarray or numpy structured array): The input vector,
            which needs to be unpacked.

        dtype (numpy.dtype): The dtype of the input x.

    Returns:
        numpy.ndarray: x unpacked into a 1-dimensional numpy.ndarray.

    """

    # Check for illegal inputs
    try:
        xdt = np.dtype(dtype)
    except BaseException:
        raise TypeError("dtype does not match any known numpy dtype")
    try:
        x_in = np.array(x)
    except BaseException:
        raise TypeError("x could not be cast as a numpy.array")
    if (xdt.names is None) != (x_in.dtype.names is None):
        raise TypeError("x and given dtype are incompatible")
    elif (xdt.names is not None) and any([name not in x_in.dtype.names
                                          for name in xdt.names]):
        raise TypeError("x and given dtype are incompatible")
    elif (xdt.names is None) and np.prod(xdt.shape) != np.prod(x_in.shape):
        raise TypeError("x and given dtype are incompatible")
    # Convert inputs to a numpy ndarray if necessary
    use_names = xdt.names is not None
    if use_names:
        # Allocate output ndarray
        n = 0
        for name in xdt.names:
            n += x_in[name].size
        xx = np.zeros(n)
        # Unpack the x vector
        i = 0
        for name in xdt.names:
            j = x_in[name].size
            xx[i:i+j] = x_in[name].flatten()
            i += j
    # Otherwise, do nothing
    else:
        xx = x_in
    return xx
