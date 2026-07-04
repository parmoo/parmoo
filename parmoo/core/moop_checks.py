""" This module contains several type-checking functions for the MOOP class.

They are:
 * `check_sims(n, arg1, arg2, ...)`

"""

import numpy as np
from parmoo.searches.global_search import GlobalSearch
from parmoo.surrogates.surrogate_function import SurrogateFunction
from parmoo.utilities.error_checks import get_hp
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

    for s, arg in enumerate(args):
        if not isinstance(arg, dict):
            raise TypeError("sims[" + str(s) + "] is not a dict")
        if "name" in arg and not isinstance(arg["name"], str):
            raise TypeError(
                    f"sims[{s}]['name'] must be a string when present"
            )
        if "m" not in arg:
            raise KeyError(f"sims[{s}] is missing the key 'm'")
        if not isinstance(arg["m"], int):
            raise TypeError(f"sims[{s}] : 'm' must be an int")
        if arg["m"] <= 0:
            raise ValueError(f"sims[{s}]['m'] must be greater than zero")
        m = arg["m"]
        if "hyperparams" in arg and not isinstance(arg["hyperparams"], dict):
            raise TypeError(
                    f"sims[{s}]: 'hyperparams' key must be a dict when present"
            )
        if "search" not in arg:
            raise KeyError(f"sims[{s}] is missing the key 'search'")
        try:
            assert isinstance(
                arg['search'](m, np.zeros(n), np.ones(n), {}), GlobalSearch
            )
        except BaseException:
            raise TypeError(
                f"sims[{s}]['search'] must be a derivative of the "
                "GlobalSearch abstract class"
            )
        if "surrogate" not in arg:
            raise KeyError(f"sims[{s}] is missing the key 'surrogate'")
        try:
            assert isinstance(
                arg["surrogate"](m, np.zeros(n), np.ones(n), {}),
                SurrogateFunction
            )
        except BaseException:
            raise TypeError(
                    f"sims[{s}]['surrogate'] must be a derivative of the "
                    "SurrogateFunction abstract class"
            )
        if "sim_func" not in arg:
            raise KeyError(f"sims[{s}] is missing the key 'sim_func'")
        if not callable(arg["sim_func"]):
            raise TypeError("sims[{s}]['sim_func'] must be callable")
        if len(inspect.signature(arg['sim_func']).parameters) not in [1, 2]:
            raise ValueError("sims[{s}]['sim_func'] must accept 1 or 2 inputs")
        if "des_tol" in arg:
            _ = get_hp("des_tol", arg, float, lambda x: x > 0.0, 1.0e-8)

        if "sim_db" not in arg:
            continue
        if not isinstance(arg["sim_db"], dict):
            raise TypeError(f"sims[{s}]['sim_db'] must be a dict")
        if ("x_vals" in arg["sim_db"]) != ("s_vals" in arg["sim_db"]):
            raise KeyError(
                    f"sims[{s}] cannot contain a sim_db with 'x_vals' but "
                    "not 's_vals' or vice versa"
            )

        if "x_vals" not in arg["sim_db"]:
            continue

        try:
            xvals = np.asarray(arg['sim_db']['x_vals'])
            svals = np.asarray(arg['sim_db']['s_vals'], dtype=np.float64)
        except BaseException:
            raise TypeError(
                    f"Either sims[{s}]['sim_db']['x_vals'] or "
                    f"sims[{s}]['sim_db']['s_vals'] could not be cast as a "
                    "numpy array"
            )
        if (xvals.size == 0) != (svals.size == 0):
            raise ValueError(
                    f"sims[{s}]['sim_db']['x_vals'] cannot be empty "
                    f"when sims[{s}]['sim_db']['s_vals'] is nonempty, "
                    "and vice versa"
            )
        if xvals.size == 0:
            continue
        if xvals.ndim < 2 or xvals.shape[1] != n:
            raise ValueError(
                    f"sims[{s}]['sim_db']['x_vals'] does not have "
                    f"{n} columns per row"
            )
        if xvals.shape[0] != svals.shape[0]:
            raise ValueError(
                    f"sims[{s}]['sim_db']['x_vals'] does not have the same"
                    f" number of rows as sims[{s}]['sim_db']['s_vals']"
            )
        if svals.ndim < 2 or svals.shape[1] != m:
            raise ValueError(
                    f"sims[{s}]['sim_db']['s_vals'] does not have "
                    f"sims[{s}]['m'] columns per row"
            )
