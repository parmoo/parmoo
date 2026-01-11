""" This module contains several type-checking functions for the MOOP class.

They are:
 * `check_sims(n, arg1, arg2, ...)`

"""

import numpy as np
from parmoo.searches.global_search import GlobalSearch
from parmoo.surrogates.surrogate_function import SurrogateFunction
import inspect


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
                    assert isinstance(
                        arg['search'](m, np.zeros(n), np.ones(n), {}),
                        GlobalSearch
                    )
                except BaseException:
                    raise TypeError(
                        f"sims[{s}]['search'] must be a derivative of "
                        "the GlobalSearch abstract class"
                    )
            else:
                raise AttributeError(f"sims[{s}] is missing the key 'search'")
            # Check the des_tol, if present
            if 'des_tol' in arg:
                if isinstance(arg['des_tol'], float):
                    if arg['des_tol'] <= 0.0:
                        raise ValueError(
                            f"sims[{s}]['des_tol'] must be greater than 0"
                        )
                else:
                    raise TypeError(f"sims[{s}]['des_tol'] must be a float")
            # Get the surrogate function
            if 'surrogate' in arg:
                try:
                    if not isinstance(arg['surrogate'](m, np.zeros(n),
                                                       np.ones(n), {}),
                                      SurrogateFunction):
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
