
""" Class for defining a multiobjective optimization problem.

This module contains the MOOP class for defining a multiobjective
optimization problem.

The MOOP class is the master class for representing and solving a
multiobjective optimization problem. Each MOOP object may contain several
simulations, specified using dictionaries.

"""

import numpy as np
import json
from parmoo import structs
import inspect


class MOOP:
    """ Class for defining a multiobjective optimization problem (MOOP).

    Upon initialization, one must supply a scalar optimization procedure
    and dictionary of hyperparameters. Class methods are summarized below.

    Objectives and algebraic constraints on the design variables and
    objective values can be added using the following functions:
     * ``addDesign(*args)``
     * ``addSimulation(*args)``
     * ``addObjective(*args)``
     * ``addConstraint(*args)``

    Acquisition functions (used for scalarizing problems/setting targets) are
    added using:
     * ``addAcquisition(*args)``

    After creating a MOOP, the following methods are used to get the
    numpy.dtype used to create each of the following input/output arrays:
     * ``getDesignType()``
     * ``getSimulationType()``
     * ``getObjectiveType()``
     * ``getConstraintType()``

    The following methods are used to save/load ParMOO objects from memory.
     * ``setCheckpoint(checkpoint, checkpoint_data=False, filename="parmoo")``
     * ``save(filename="parmoo")``
     * ``load(filename="parmoo")``

    The following methods are used for solving the MOOP and managing the
    internal simulation/objective databases:
     * ``check_sim_db(x, s_name)``
     * ``update_sim_db(x, sx, s_name)``
     * ``evaluateSimulation(x, s_name)``
     * ``addData(x, sx)``
     * ``iterate(k)``
     * ``updateAll(k, batch)``
     * ``solve(budget)``

    Finally, the following methods are used to retrieve data after the
    problem has been solved:
     * ``getPF()``
     * ``getSimulationData()``
     * ``getObjectiveData()``

    The following methods are not recommended for external usage, but
    are included for internal usage primarily:

     * ``__extract__(x)``
     * ``__embed__(x)``
     * ``__generate_encoding__()``
     * ``__unpack_sim__(sx)``
     * ``__pack_sim__(sx)``
     * ``fitSurrogates()``
     * ``updateSurrogates()``
     * ``evaluateSurrogates(x)``
     * ``resetSurrogates(center)``
     * ``evaluateConstraints(x)``
     * ``evaluateLagrangian(x)``
     * ``evaluateGradients(x)``

    """

    # Slots for the MOOP class
    __slots__ = ['n', 'm', 'm_total', 'o', 'p', 's', 'n_dat', 'lb', 'ub',
                 'n_cat_d', 'cat_lb', 'cat_scale', 'RSVT', 'mean', 'n_cat',
                 'n_cont', 'n_lvls', 'des_order', 'cat_names',
                 'sim_names', 'des_names', 'obj_names', 'const_names',
                 'lam', 'objectives', 'data', 'sim_funcs', 'sim_db',
                 'des_tols', 'searches', 'surrogates', 'optimizer',
                 'constraints', 'hyperparams', 'acquisitions', 'history',
                 'scale', 'scaled_lb', 'scaled_ub', 'scaled_des_tols',
                 'cat_des_tols', 'use_names', 'iteration', 'checkpoint',
                 'checkpointfile', 'checkpoint_data', 'new_checkpoint',
                 'new_data']

    def __embed__(self, x):
        """ Embed a design input as n-dimensional vector for ParMOO.

        Args:
            x (numpy.ndarray): Either a numpy structured array (when 'name'
                key was given for all design variables) or a 1D array
                containing design variable values (in the order that they
                were added to the MOOP).

        Returns:
            numpy.ndarray: A 1D array containing the embedded design vector.

        """

        # Unpack x into an ordered unstructured array
        x_tmp = np.zeros(self.n_cont + self.n_cat)
        if self.use_names:
            x_labels = []
            for d_name in self.des_names:
                x_labels.append(x[d_name[0]])
            for i, j in enumerate(self.des_order):
                if (j >= self.n_cont) and (len(self.cat_names[j - self.n_cont])
                                           > 0):
                    x_tmp[i] = float(self.cat_names[j - self.n_cont].index(
                                                                x_labels[j]))
                else:
                    x_tmp[i] = x_labels[j]
        else:
            x_tmp = x[self.des_order]
        # Create the output array
        xx = np.zeros(self.n)
        # Rescale the continuous variables
        start = 0
        end = self.n_cont
        xx[start:end] = ((x_tmp[start:end] - self.lb[start:end]) /
                         self.scale[start:end] + self.scaled_lb[start:end])
        # Pull inside bounding box, in case perturbed outside
        xx[start:end] = np.maximum(xx[start:end], self.scaled_lb[start:end])
        xx[start:end] = np.minimum(xx[start:end], self.scaled_ub[start:end])
        # Embed the categorical variables
        if self.n_cat_d > 0:
            bvec = np.zeros(sum(self.n_lvls))
            count = 0
            for i, n_lvl in enumerate(self.n_lvls):
                bvec[count + int(x_tmp[self.n_cont + i])] = 1.0
                count += n_lvl
            bvec -= self.mean
            start = self.n_cont
            end = self.n_cont + self.n_cat_d
            xx[start:end] = ((np.matmul(self.RSVT, bvec) - self.cat_lb[:])
                             / self.scale[start:end]
                             + self.scaled_lb[start:end])
            # Pull inside bounding box, in case perturbed outside
            xx[start:end] = np.maximum(xx[start:end],
                                       self.scaled_lb[start:end])
            xx[start:end] = np.minimum(xx[start:end],
                                       self.scaled_ub[start:end])
        return xx

    def __extract__(self, x):
        """ Extract a design variable from an n-dimensional vector.

        Args:
            x (numpy.ndarray): A 1D array containing the embedded design
                vector.

        Returns:
            numpy.ndarray: Either a numpy structured array (when 'name'
            key was given for all design variables) or a 1D array
            containing design variable values (in the order that they
            were added to the MOOP).

        """

        # Create the output array
        xx = np.zeros(self.n_cont + self.n_cat)
        # Descale the continuous variables
        start = 0
        end = self.n_cont
        xx[start:end] = ((x[start:end] - self.scaled_lb[start:end])
                         * self.scale[start:end] + self.lb[start:end])
        # Pull inside bounding box, in case perturbed outside
        xx[start:end] = np.maximum(xx[start:end], self.lb[start:end])
        xx[start:end] = np.minimum(xx[start:end], self.ub[start:end])
        # Extract categorical variables
        if self.n_cat_d > 0:
            start = self.n_cont
            end = self.n_cont + self.n_cat_d
            bvec = (np.matmul(np.transpose(self.RSVT),
                              (x[start:end] - self.scaled_lb[start:end])
                              * self.scale[start:end] + self.cat_lb[:])
                    + self.mean)
            count = 0
            for i, n_lvl in enumerate(self.n_lvls):
                xx[start+i] = np.argmax(bvec[count:count+n_lvl])
                count += n_lvl
        # Unshuffle xx and pack into a numpy structured array
        if self.use_names:
            out = np.zeros(1, dtype=np.dtype(self.des_names))
            for i, j in enumerate(self.des_order):
                if (i >= self.n_cont) and (len(self.cat_names[i - self.n_cont])
                                           > 0):
                    out[self.des_names[i][0]] = (self.cat_names[i -
                                                                self.n_cont]
                                                               [int(xx[j])])
                else:
                    out[self.des_names[i][0]] = xx[j]
            return out[0]
        else:
            return xx[self.des_order]

    def __generate_encoding__(self):
        """ Generate the encoding matrices for this MOOP.

        """

        # Generate every valid one-hot-encoding
        codex = np.zeros((np.prod(self.n_lvls), sum(self.n_lvls)))
        self.mean = np.zeros(sum(self.n_lvls))
        count = 0
        stride = 1
        for i in self.n_lvls:
            for j in range(np.prod(self.n_lvls)):
                index = (j // stride) % i + count
                codex[j, index] = 1.0
                self.mean[index] += 1.0
            stride *= i
            count += i
        self.mean /= float(np.prod(self.n_lvls))
        # Subtract out the mean
        for i in range(codex.shape[0]):
            codex[i, :] = codex[i, :] - self.mean
        # Take SVD of normalized one-hot design matrix
        U, S, VT = np.linalg.svd(codex)
        self.n_cat_d = 0
        for si in S:
            if si > 1.0e-8:
                self.n_cat_d += 1
        # Store transposed right singular vectors for encoding/decoding
        self.RSVT = VT[:self.n_cat_d, :]
        # Calculate the scaling
        classes = []
        for xi in codex:
            classes.append(np.matmul(self.RSVT, xi - self.mean))
        # Calculate the componentwise upper bounds, lower bounds, and mindists
        mindists = np.ones(self.n_cat_d) * float(self.n)
        maxvals = -np.ones(self.n_cat_d) * float(self.n)
        minvals = np.ones(self.n_cat_d) * float(self.n)
        for xi in classes:
            # Check upper/lower bounds
            for j in range(self.n_cat_d):
                if xi[j] < minvals[j]:
                    minvals[j] = xi[j]
                if xi[j] > maxvals[j]:
                    maxvals[j] = xi[j]
            # Check pairwise distances
            for xj in classes:
                if any(xj != xi):
                    for k in range(self.n_cat_d):
                        if abs(xi[k] - xj[k]) < mindists[k] and \
                           abs(xi[k] - xj[k]) > 1.0e-4:
                            mindists[k] = abs(xi[k] - xj[k])
        self.cat_lb = minvals
        self.cat_scale = maxvals - minvals
        self.cat_des_tols = mindists / self.cat_scale
        return

    def __unpack_sim__(self, sx):
        """ Extract a simulation output from a m-dimensional vector.

        Args:
            sx (numpy.ndarray): A 1D array containing the vectorized sim
                output.

        Returns:
            numpy.ndarray: Either a numpy structured array (when 'name'
            key was given for all design variables) or the identity
            mapping, otherwise.

        """

        # Check whether we are using named sims
        if self.use_names:
            # Unpack sx into a numpy structured array
            sxx = np.zeros(1, dtype=self.sim_names)
            m_count = 0
            for i, mi in enumerate(self.m):
                sxx[self.sim_names[i][0]] = sx[m_count:m_count+mi]
                m_count += mi
            return sxx[0]
        else:
            # Do nothing
            return sx

    def __pack_sim__(self, sx):
        """ Pack a simulation output into a m-dimensional vector.

        Args:
            sx (numpy.ndarray): A numpy structured array (when all design
                variables are named) or a m-dimensional vector.

        Returns:
            numpy.ndarray: A 1D numpy.ndarray of length m.

        """

        # Check whether we are using named sims
        if self.use_names:
            # Pack sx into a numpy.ndarray
            sxx = np.zeros(self.m_total)
            m_count = 0
            for i, mi in enumerate(self.m):
                sxx[m_count:m_count+mi] = sx[self.sim_names[i][0]]
                m_count += mi
            return sxx
        else:
            # Do nothing
            return sx

    def __init__(self, opt_func, hyperparams=None):
        """ Initializer for the MOOP class.

        Args:
            opt_func (SurrogateOptimizer): A solver for the surrogate problems.

            hyperparams (dict, optional): A dictionary of hyperparameters for
                the opt_func, and any other procedures that will be used.

        Returns:
            MOOP: A new MOOP object with no design variables, objectives, or
            constraints.

        """

        # Set the hyperparams optional input.
        if hyperparams is None:
            self.hyperparams = {}
        else:
            if isinstance(hyperparams, dict):
                self.hyperparams = hyperparams
            else:
                raise TypeError("hyperparams must be a Python dict")
        # Set up problem dimensions
        self.n = 0
        self.des_names = []
        self.des_order = []
        self.des_tols = []
        self.n_cont = 0
        self.lb = []
        self.ub = []
        self.n_cat = 0
        self.cat_names = []
        self.n_cat_d = 0
        self.n_lvls = []
        # Initialize the scale
        self.scale = []
        self.scaled_lb = []
        self.scaled_ub = []
        self.scaled_des_tols = []
        self.cat_des_tols = []
        self.cat_lb = []
        self.cat_scale = []
        # Initialize the embedding transformation
        self.RSVT = []
        self.mean = []
        # Initialize lists for storing simulation information
        self.s = 0
        self.m = []
        self.m_total = 0
        self.sim_names = []
        self.searches = []
        self.sim_funcs = []
        self.sim_db = []
        self.surrogates = []
        # Initialize the objective and constraint lists
        self.o = 0
        self.obj_names = []
        self.objectives = []
        self.p = 0
        self.const_names = []
        self.constraints = []
        # Initialize the empty acquisition function list
        self.acquisitions = []
        # Initialize empty history dict
        self.history = {}
        # Initialize the augmented Lagrange multiplier
        self.lam = 1.0
        # Reset the database
        self.n_dat = 0
        self.data = {}
        # Use names
        self.use_names = True
        # Initialize the iteration counter
        self.iteration = 0
        # Initialize the checkpointing
        self.checkpoint = False
        self.checkpoint_data = False
        self.checkpointfile = "parmoo"
        # Track whether we are creating a new checkpoint and/or datafile
        self.new_checkpoint = True
        self.new_data = True
        # Set up the surrogate optimizer
        try:
            # Try initializing the optimizer, to check that it can be done
            opt = opt_func(1, np.zeros(1), np.ones(1), self.hyperparams)
            assert(isinstance(opt, structs.SurrogateOptimizer))
        except BaseException:
            raise TypeError("opt_func must be a derivative of the "
                            + "SurrogateOptimizer abstract class")
        self.optimizer = opt_func
        return

    def addDesign(self, *args):
        """ Add a new design variables to the MOOP.

        Append new design variables to the problem. Note that every design
        variable must be added before any simulations or acquisition functions
        can be added since the number of design variables is used to infer
        the size of simulation databases and acquisition function policies.

        Args:
            args (dict): Each argument is a dictionary representing one design
                variable. The dictionary contains information about that
                design variable, including:
                 * 'name' (String, optional): The name of this design
                   if any are left blank, then ALL names are considered
                   unspecified.
                 * 'des_type' (String): The type for this design variable.
                   Currently supported options are:
                    * 'continuous'
                    * 'categorical'
                 * 'lb' (float): When des_type is 'continuous', this specifies
                   the lower bound for the design variable. This value must
                   be specified, and must be strictly less than 'ub'
                   (below) up to the tolerance (below).
                 * 'ub' (float): When des_type is 'continuous', this specifies
                   the upper bound for the design variable. This value
                   must be specified, and must be strictly greater than
                   'lb' (above) up to the tolerance (below).
                 * 'tol' (float): When des_type is 'continuous', this specifies
                   the tolerance, i.e., the minimum spacing along this
                   dimension, before two design values are considered to
                   have equal values in this dimension. If not specified, the
                   default value is 1.0e-8.
                 * 'levels' (int): When des_type is 'categorical', this
                   specifies the number of levels for the variable.

        """

        if len(self.acquisitions) > 0:
            raise RuntimeError("Cannot add more design variables after"
                               + " adding acquisition functions")
        if len(self.sim_funcs) > 0:
            raise RuntimeError("Cannot add more design variables after"
                               + " adding simulation functions")
        for arg in args:
            # Check arg for correct types
            if not isinstance(arg, dict):
                raise TypeError("Each argument must be a Python dict")
            # Check for design variable type
            if 'des_type' in arg.keys():
                if not isinstance(arg['des_type'], str):
                    raise TypeError("args['des_type'] must be a String")
                # Append a new continous design variable to the list
                if arg['des_type'] == "continuous":
                    if 'des_tol' in arg.keys():
                        if isinstance(arg['des_tol'], float):
                            if arg['des_tol'] > 0.0:
                                des_tol = arg['des_tol']
                            else:
                                raise ValueError("args['des_tol'] must be "
                                                 + "positive")
                        else:
                            raise TypeError("args['des_tol'] must be a float")
                    else:
                        des_tol = 1.0e-8
                    if 'lb' in arg.keys() and 'ub' in arg.keys():
                        if not (isinstance(arg['lb'], float) and
                                isinstance(arg['ub'], float)):
                            raise TypeError("args['lb'] and args['ub'] must "
                                            + "be float types")
                        if arg['lb'] + des_tol > arg['ub']:
                            raise ValueError("args['lb'] must be less than "
                                             + "args['ub'] up to the design "
                                             + "space tolerance")
                    else:
                        raise AttributeError("'lb' and 'ub' keys must be "
                                             + "present when 'des_type' is "
                                             + "'continuous'")
                    if 'name' in arg.keys():
                        if isinstance(arg['name'], str):
                            if any([arg['name'] == dname[0]
                                    for dname in self.des_names]):
                                raise ValueError("arg['name'] must be unique")
                            self.des_names.append((arg['name'], 'f8'))
                        else:
                            raise TypeError("When present, 'name' must be a "
                                            + "String type")
                    else:
                        self.use_names = False
                        name = "x" + str(self.n_cont + self.n_cat + 1)
                        self.des_names.append((name, 'f8', ))
                    # Keep track of design variable indices for bookkeeping
                    for i in range(len(self.des_order)):
                        # Add 1 to all categorical variable indices
                        if self.des_order[i] >= self.n_cont:
                            self.des_order[i] += 1
                    self.des_order.append(self.n_cont)
                    self.n_cont += 1
                    self.des_tols.append(des_tol)
                    self.lb.append(arg['lb'])
                    self.ub.append(arg['ub'])
                # Append a new categorical design variable to the list
                elif arg['des_type'] == "categorical":
                    if 'levels' in arg.keys():
                        if isinstance(arg['levels'], int):
                            if arg['levels'] < 2:
                                raise ValueError("args['levels'] must be at "
                                                 + "least 2")
                        elif isinstance(arg['levels'], list):
                            if len(arg['levels']) < 2:
                                raise ValueError("args['levels'] must contain "
                                                 + "at least 2 categories")
                            if any([not isinstance(lvl, str)
                                    for lvl in arg['levels']]):
                                raise TypeError("args['levels'] must contain "
                                                + "strings, if a list")
                        else:
                            raise TypeError("args['levels'] must be an int")
                    else:
                        raise AttributeError("'levels' must be present when "
                                             + "'des_type' is 'categorical'")
                    if 'name' in arg.keys():
                        if not isinstance(arg['name'], str):
                            raise TypeError("When present, 'name' must be a "
                                            + "String type")
                    else:
                        self.use_names = False
                        name = "x" + str(self.n_cont + self.n_cat + 1)
                        self.des_names.append((name, 'i4', ))
                    # Keep track of design variable indices for bookkeeping
                    self.des_order.append(self.n_cont + self.n_cat)
                    self.n_cat += 1
                    self.des_tols.append(0.5)
                    if isinstance(arg['levels'], int):
                        self.n_lvls.append(arg['levels'])
                        self.cat_names.append([])
                        if 'name' in arg.keys():
                            self.des_names.append((arg['name'], 'i4'))
                    else:
                        self.n_lvls.append(len(arg['levels']))
                        self.cat_names.append(arg['levels'])
                        if 'name' in arg.keys():
                            self.des_names.append((arg['name'], 'U25'))
                    self.__generate_encoding__()
                else:
                    raise(ValueError("des_type=" + arg['des_type'] +
                                     " is not a recognized value"))
            else:
                # The default des_type is continuous
                if 'des_tol' in arg.keys():
                    if not isinstance(arg['des_tol'], float):
                        raise TypeError("args['des_tol'] must be a float")
                    if arg['des_tol'] > 0.0:
                        des_tol = arg['des_tol']
                    else:
                        raise ValueError("args['des_tol'] must be positive")
                else:
                    des_tol = 1.0e-8
                if 'lb' in arg.keys() and 'ub' in arg.keys():
                    if not (isinstance(arg['lb'], float) and
                            isinstance(arg['ub'], float)):
                        raise TypeError("args['lb'] and args['ub'] must "
                                        + "be float types")
                    if arg['lb'] + des_tol > arg['ub']:
                        raise ValueError("args['lb'] must be less than "
                                         + "args['ub'] up to the design "
                                         + "space tolerance")
                else:
                    raise AttributeError("'lb' and 'ub' keys must be "
                                         + "present when 'des_type' is "
                                         + "'continuous'")
                if 'name' in arg.keys():
                    if isinstance(arg['name'], str):
                        self.des_names.append((arg['name'], 'f8'))
                    else:
                        raise TypeError("When present, 'name' must be a "
                                        + "String type")
                else:
                    self.use_names = False
                    name = "x" + str(self.n_cont + self.n_cat + 1)
                    self.des_names.append((name, 'f8', ))
                # Keep track of design variable indices for bookkeeping
                for i in range(len(self.des_order)):
                    # Add 1 to all categorical variable indices
                    if self.des_order[i] >= self.n_cont:
                        self.des_order[i] += 1
                self.des_order.append(self.n_cont)
                self.n_cont += 1
                self.des_tols.append(des_tol)
                self.lb.append(arg['lb'])
                self.ub.append(arg['ub'])
        # Set the effective design dimension
        self.n = self.n_cat_d + self.n_cont
        # Set the problem scaling
        self.scaled_lb = np.zeros(self.n)
        self.scaled_ub = np.ones(self.n)
        self.scale = np.ones(self.n)
        for i in range(self.n_cont):
            self.scale[i] = self.ub[i] - self.lb[i]
        self.scaled_des_tols = np.zeros(self.n)
        self.scaled_des_tols[:self.n_cont] = (self.des_tols[:self.n_cont] /
                                              self.scale[:self.n_cont])
        if self.n_cat_d > 0:
            self.scaled_des_tols[self.n_cont:self.n_cont+self.n_cat_d] = \
                self.cat_des_tols[:]
            self.scale[self.n_cont:self.n_cont+self.n_cat_d] = \
                self.cat_scale[:]
        # Reset the database
        self.n_dat = 0
        self.data = {}
        return

    def addSimulation(self, *args):
        """ Add new simulations to the MOOP.

        Append new simulation functions to the problem.

        Args:
            args (dict): Each argument is a dictionary representing one
                simulation function. The dictionary must contain information
                about that simulation function, including:
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

        from parmoo.util import check_sims

        # Iterate through args to add each sim
        check_sims(self.n_cont + self.n_cat, *args)
        for arg in args:
            # Use the number of sims
            m = arg['m']
            # Keep track of simulation names
            if 'name' in arg.keys():
                if any([arg['name'] == dname[0] for dname in self.sim_names]):
                    raise ValueError("arg['name'] must be unique")
                if m > 1:
                    self.sim_names.append((arg['name'], 'f8', m))
                else:
                    self.sim_names.append((arg['name'], 'f8'))
            else:
                name = "sim" + str(self.s + 1)
                if m > 1:
                    self.sim_names.append((name, 'f8', m))
                else:
                    self.sim_names.append((name, 'f8'))
            # Track the current num of sim outs and the total num of sim outs
            self.m.append(m)
            self.m_total += m
            # Get the hyperparameter dictionary
            if 'hyperparams' in arg.keys():
                hyperparams = arg['hyperparams']
            else:
                hyperparams = {}
            # Get the search technique
            self.searches.append(arg['search'](m,
                                               self.scaled_lb,
                                               self.scaled_ub,
                                               hyperparams))
            # Get the surrogate function
            hyperparams['des_tols'] = np.asarray(self.scaled_des_tols)
            self.surrogates.append(arg['surrogate'](m,
                                                    self.scaled_lb,
                                                    self.scaled_ub,
                                                    hyperparams))
            # Get the simulation function
            self.sim_funcs.append(arg['sim_func'])
            # Get the starting database, if present
            if 'sim_db' in arg.keys():
                if 'x_vals' in arg['sim_db'] and \
                   's_vals' in arg['sim_db']:
                    # If x_vals and s_vals are present, cast to np.ndarray
                    xvals = np.asarray(arg['sim_db']['x_vals'])
                    svals = np.asarray(arg['sim_db']['s_vals'])
                    if xvals.size == 0:
                        # If x_vals/s_vals are empty, create an empty DB
                        self.sim_db.append({'x_vals': np.zeros((1, self.n)),
                                            's_vals': np.zeros((1, m)),
                                            'n': 0,
                                            'old': 0})
                    else:
                        # If nonempty, add xvals and svals to the database
                        xxvals = np.asarray([self.__embed__(xi)
                                             for xi in xvals])
                        self.sim_db.append({'x_vals': xxvals,
                                            's_vals': svals,
                                            'n': xvals.shape[0],
                                            'old': 0})
                else:
                    # If x_vals/s_vals are not in sim_db, create an empty DB
                    self.sim_db.append({'x_vals': np.zeros((1, self.n)),
                                        's_vals': np.zeros((1, m)),
                                        'n': 0,
                                        'old': 0})
            else:
                # If no sim_db was given, create an empty DB
                self.sim_db.append({'x_vals': np.zeros((1, self.n)),
                                    's_vals': np.zeros((1, m)),
                                    'n': 0,
                                    'old': 0})
            # Increment the total sim counter
            self.s += 1
        return

    def addObjective(self, *args):
        """ Add a new objective to the MOOP.

        Append a new objective to the problem. The objective must be an
        algebraic function of the simulations. Note that all objectives
        must be specified before any acquisition functions can be added.

        Args:
            *args (dict): Python dictionary containing objective function
            info, including:
             * 'name' (String, optional): The name of this objective
               (defaults to "obj" + str(i), where i = 1, 2, 3, ... for
               the first, second, third, ... simulation added to the MOOP).
             * 'obj_func' (function): An algebraic objective function that maps
               from R^n X R^m --> R. Interface should match:
               `cost = obj_func(x, sim_func(x), der=0)`,
               where `der` is an optional argument specifying whether to
               take the derivative of the objective function
                * 0 (or any other value) -- not at all,
                * 1 -- wrt x, or
                * 2 -- wrt sim(x).

        """

        # Assert proper order of problem definition
        if len(self.acquisitions) > 0:
            raise RuntimeError("Cannot add more objectives after"
                               + " adding acquisition functions")
        # Check that arg and 'obj_func' field are legal types
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError("Each arg must be a Python dict")
            if 'obj_func' in arg.keys():
                if callable(arg['obj_func']):
                    if not (len(inspect.signature(arg['obj_func']).parameters)
                            == 2 or
                            len(inspect.signature(arg['obj_func']).parameters)
                            == 3):
                        raise ValueError("The 'obj_func' must take 2 "
                                         + "(no derivatives) or 3 "
                                         + "(derivative option) arguments")
                else:
                    raise TypeError("The 'obj_func' must be callable")
            else:
                raise AttributeError("The 'obj_func' field must be "
                                     + "present in each arg")
            # Add the objective name
            if 'name' in arg.keys():
                if not isinstance(arg['name'], str):
                    raise TypeError("When present, 'name' must be a string")
                else:
                    if any([arg['name'] == dname[0]
                            for dname in self.obj_names]):
                        raise ValueError("arg['name'] must be unique")
                    self.obj_names.append((arg['name'], 'f8'))
            else:
                self.obj_names.append(("f" + str(self.o + 1), 'f8'))
            # Finally, if all else passed, add the objective
            self.objectives.append(arg['obj_func'])
            self.o += 1
        return

    def addConstraint(self, *args):
        """ Add a new constraint to the MOOP.

        Append a new design constraint to the problem. The constraint can
        be nonlinear and depend on the design values and simulation outputs.

        Args:
            *args (dict): Python dictionary containing constraint function
            information, including:
             * 'name' (String, optional): The name of this constraint
               (defaults to "const" + str(i), where i = 1, 2, 3, ... for
               the first, second, third, ... constraint added to the MOOP).
             * 'constraint' (function): An algebraic constraint function that
               maps from R^n X R^m --> R and evaluates to zero or a
               negative number when feasible and positive otherwise.
               Interface should match:
               `violation = constraint(x, sim_func(x), der=0)`,
               where `der` is an optional argument specifying whether to
               take the derivative of the constraint function
                * 0 (or any other value) -- not at all,
                * 1 -- wrt x, or
                * 2 -- wrt sim(x).

        """

        # Check that arg and 'constraint' field are legal types
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError("Each arg must be a Python dict")
            if 'constraint' in arg.keys():
                if callable(arg['constraint']):
                    if not (len(inspect.signature(arg['constraint']).
                                parameters) == 2 or
                            len(inspect.signature(arg['constraint']).
                                parameters) == 3):
                        raise ValueError("The 'constraint' must take 2 "
                                         + "(no derivatives) or 3 "
                                         + "(derivative option) arguments")
                else:
                    raise TypeError("The 'constraint' must be callable")
            else:
                raise AttributeError("The 'constraint' field must be "
                                     + "present in each arg")
            # Add the constraint name
            if 'name' in arg.keys():
                if not isinstance(arg['name'], str):
                    raise TypeError("When present, 'name' must be a string")
                else:
                    if any([arg['name'] == dname[0]
                            for dname in self.const_names]):
                        raise ValueError("arg['name'] must be unique")
                    self.const_names.append((arg['name'], 'f8'))
            else:
                self.const_names.append(("c" + str(self.p + 1), 'f8'))
            # Finally, if all else passed, add the constraint
            self.constraints.append(arg['constraint'])
            self.p += 1
        return

    def addAcquisition(self, *args):
        """ Add an acquisition function to the MOOP.

        Append a new acquisition function to the problem. In each iteration,
        each acquisition is used to generate 1 or more points to evaluate.

        Args:
            args (dict): Python dictionary of acquisition function info,
                including:
                 * 'acquisition' (AcquisitionFunction): An acquisition function
                   that maps from R^o --> R for scalarizing outputs.
                 * 'hyperparams' (dict): A dictionary of hyperparams for the
                   acquisition functions. Can be omitted if no hyperparams
                   are needed.

        """

        # Append the acquisition function to the list
        for arg in args:
            if self.n < 1:
                raise ValueError("Cannot add acquisition function without "
                                 + "design variables")
            if self.o < 1:
                raise ValueError("Cannot add acquisition function without "
                                 + "objectives")
            if not isinstance(arg, dict):
                raise TypeError("Every arg must be a Python dict")
            if 'acquisition' not in arg.keys():
                raise AttributeError("'acquisition' field must be present in "
                                     + "every arg")
            if 'hyperparams' in arg.keys():
                if not isinstance(arg['hyperparams'], dict):
                    raise TypeError("When present, 'hyperparams' must be a "
                                    + "Python dict")
                hyperparams = arg['hyperparams']
            else:
                hyperparams = {}
            try:
                acquisition = arg['acquisition'](self.o,
                                                 self.scaled_lb,
                                                 self.scaled_ub,
                                                 hyperparams)
                assert(isinstance(acquisition, structs.AcquisitionFunction))
            except BaseException:
                raise TypeError("'acquisition' must specify a child of the"
                                + " AcquisitionFunction class")
            # If all checks passed, add the acquisition to the list
            self.acquisitions.append(acquisition)
        return

    def setCheckpoint(self, checkpoint,
                      checkpoint_data=True, filename="parmoo"):
        """ Set ParMOO's checkpointing feature.

        Args:
            checkpoint (bool): Turn checkpointing on (True) or off (False).

            checkpoint_data (bool, optional): Also save raw simulation output
                in a separate .json file (True) or rely on ParMOO's internal
                simulation database (False). When omitted, this parameter
                defaults to False.

            filename (str, optional): Set the base checkpoint filename/path.
                The checkpoint file will have the JSON format and the
                extension ".moop" appended to the end of filename.
                Additional checkpoint files may be created with the same
                filename but different extensions, depending on the choice
                of AcquisitionFunction, SurrogateFunction, and GlobalSearch.
                When omitted, this parameter defaults to "parmoo" and
                is saved inside current working directory.

        """

        if not isinstance(checkpoint, bool):
            raise TypeError("checkpoint must have the bool type")
        if not isinstance(filename, str):
            raise TypeError("filename must have the string type")
        # Set internal checkpointing variables
        self.checkpoint = checkpoint
        self.checkpoint_data = checkpoint_data
        self.checkpointfile = filename
        return

    def getDesignType(self):
        """ Get the numpy dtype of a design point for this MOOP.

        Use this type if allocating a numpy array to store the design
        points for this MOOP object.

        Returns:
            The numpy dtype of this MOOP's design points.
            If no design variables have yet been added, returns None.

        """

        if self.n_cont + self.n_cat < 1:
            return None
        elif self.use_names:
            return np.dtype(self.des_names)
        else:
            return np.dtype(('f8', (self.n_cont + self.n_cat,)))

    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP.

        Use this type if allocating a numpy array to store the simulation
        outputs of this MOOP object.

        Returns:
            The numpy dtype of this MOOP's simulation outputs.
            If no simulations have been given, returns None.

        """

        if self.m_total < 1:
            return None
        elif self.use_names:
            return np.dtype(self.sim_names)
        else:
            return np.dtype(('f8', (self.m_total,)))

    def getObjectiveType(self):
        """ Get the numpy dtype of an objective point for this MOOP.

        Use this type if allocating a numpy array to store the objective
        values of this MOOP object.

        Returns:
            The numpy dtype of this MOOP's objective points.
            If no objectives have yet been added, returns None.

        """

        if self.o < 1:
            return None
        elif self.use_names:
            return np.dtype(self.obj_names)
        else:
            return np.dtype(('f8', (self.o,)))

    def getConstraintType(self):
        """ Get the numpy dtype of the constraint violations for this MOOP.

        Use this type if allocating a numpy array to store the constraint
        scores output of this MOOP object.

        Returns:
            The numpy dtype of this MOOP's constraint violation output.
            If no constraints have been given, returns None.

        """

        if self.p < 1:
            return None
        elif self.use_names:
            return np.dtype(self.const_names)
        else:
            return np.dtype(('f8', (self.p,)))

    def check_sim_db(self, x, s_name):
        """ Check the sim_db[s_name] in this MOOP for a design point.

        x (np.ndarray): A 1d array specifying the point to check for.

        s_name (String, int): The name or index of the simulation where
            (x, sx) will be added. Note, indices are assigned in the order
            the simulations were listed during initialization.

        """

        # Extract the simulation name
        if isinstance(s_name, str):
            i = -1
            for j, sj in enumerate(self.sim_names):
                if sj[0] == s_name:
                    i = j
                    break
        elif isinstance(s_name, int):
            i = s_name
        else:
            raise TypeError("s_name must be a string or int")
        # Check for errors
        if i < 0 or i > self.s - 1:
            raise ValueError("s_name did not contain a legal name/index")
        # Check the database for previous evaluations of x
        for j in range(self.sim_db[i]['n']):
            if all(abs(self.sim_db[i]['x_vals'][j, :] - self.__embed__(x)) <
                   self.scaled_des_tols):
                # If found, return the sim value
                return self.sim_db[i]['s_vals'][j, :]
        # Nothing found, return None
        return None

    def update_sim_db(self, x, sx, s_name):
        """ Update sim_db[s_name] by adding a new design/objective pair.

        x (np.ndarray): A 1d array specifying the design point to add.

        sx (np.ndarray): A 1d array with the corresponding objective value.

        s_name (String, int): The name or index of the simulation where
            (x, sx) will be added. Note, indices are assigned in the order
            the simulations were listed during initialization.

        """

        # Extract the simulation name
        if isinstance(s_name, str):
            i = -1
            for j, sj in enumerate(self.sim_names):
                if sj[0] == s_name:
                    i = j
                    break
        elif isinstance(s_name, int):
            i = s_name
        else:
            raise TypeError("s_name must be a string or int")
        # Check for errors
        if i < 0 or i > self.s - 1:
            raise ValueError("s_name did not contain a legal name/index")
        # Extract data
        xx = self.__embed__(x)
        # Check whether sim_db[i]['n'] > 0
        if self.sim_db[i]['n'] > 0:
            # If sim_db[i]['n'] > 0, then append to the database
            self.sim_db[i]['x_vals'] = np.append(self.sim_db[i]['x_vals'],
                                                 [xx], axis=0)
            self.sim_db[i]['s_vals'] = np.append(self.sim_db[i]['s_vals'],
                                                 [sx], axis=0)
            self.sim_db[i]['n'] += 1
        else:
            # If sim_db[i]['n'] == 0, then set the zeroeth value
            self.sim_db[i]['x_vals'][0, :] = xx
            self.sim_db[i]['s_vals'][0, :] = sx
            self.sim_db[i]['n'] += 1
        # If data-saving is on, append the sim output to a json
        if self.checkpoint_data:
            self.savedata(x, sx, s_name, filename=self.checkpointfile)
        # If checkpointing is on, save the moop before continuing
        if self.checkpoint:
            self.save(filename=self.checkpointfile)
        return

    def evaluateSimulation(self, x, s_name):
        """ Evaluate the simulation[s_name] and store the result.

        Args:
            x (numpy.ndarray): Either a numpy structured array (when 'name'
                key was given for all design variables) or a 1D array
                containing design variable values (in the order that they
                were added to the MOOP).

            s_name (String, int): The name or index of the simulation to
                evaluate. Note, indices are assigned in the order
                the simulations were listed during initialization.

        Returns:
            numpy.ndarray: A 1d array containing the output from the
            simulation[s_name] at x.

        """

        # Extract the simulation name
        if isinstance(s_name, str):
            i = -1
            for j, sj in enumerate(self.sim_names):
                if sj[0] == s_name:
                    i = j
                    break
        elif isinstance(s_name, int):
            i = s_name
        else:
            raise TypeError("s_name must be a string or int")
        # Check for errors
        if i < 0 or i > self.s - 1:
            raise ValueError("s_name did not contain a legal name/index")
        # Check the sim database for x
        sx = self.check_sim_db(x, s_name)
        # If not found, evaluate the sim and add to the database
        if sx is None:
            sx = np.asarray(self.sim_funcs[i](x))
            self.update_sim_db(x, sx, s_name)
        # Return the result
        return sx

    def fitSurrogates(self):
        """ Fit the surrogate models using the current internal databases.

        """

        # Call self.surrogates.fit() to fit the surrogate models
        for i in range(self.s):
            n_new = self.sim_db[i]['n']
            self.surrogates[i].fit(self.sim_db[i]['x_vals'][:n_new, :],
                                   self.sim_db[i]['s_vals'][:n_new, :])
            self.sim_db[i]['old'] = self.sim_db[i]['n']
        return

    def updateSurrogates(self):
        """ Update the surrogate models using the current internal databases.

        """

        # Call self.surrogates.update() to update the surrogate models
        for i in range(self.s):
            n_old = self.sim_db[i]['old']
            n_new = self.sim_db[i]['n']
            self.surrogates[i].update(self.sim_db[i]['x_vals'][n_old:n_new, :],
                                      self.sim_db[i]['s_vals'][n_old:n_new, :])
            self.sim_db[i]['old'] = self.sim_db[i]['n']
        return

    def resetSurrogates(self, center):
        """ Reset the surrogates using SurrogateFunction.setCenter(center).

        Args:
            center (numpy.ndarray): A 1d array containing the coordinates
                of the new center.


        Returns:
            float: The minimum over the recommended trust region radius
            for all surrogates.

        """

        rad = max(self.scaled_ub - self.scaled_lb)
        for si in self.surrogates:
            rad = min(si.setCenter(center), rad)
        return rad

    def evaluateSurrogates(self, x):
        """ Evaluate all objectives using the simulation surrogates as needed.

        Args:
            x (numpy.ndarray): A 1d array containing the (embedded) design
                point to evaluate.

        Returns:
            numpy.ndarray: A 1d array containing the result of the evaluation.

        """

        # Check for illegal input
        if isinstance(x, np.ndarray):
            if x.shape[0] != self.n:
                raise ValueError("x must have length n")
        else:
            raise ValueError("x must be a numpy array")
        # Evaluate the surrogate models to approximate the simulation outputs
        sim = np.zeros(self.m_total)
        m_count = 0
        for i, surrogate in enumerate(self.surrogates):
            sim[m_count:m_count+self.m[i]] = surrogate.evaluate(x)
            m_count += self.m[i]
        # Evaluate the objective functions
        fx = np.zeros(self.o)
        for i, obj_func in enumerate(self.objectives):
            fx[i] = obj_func(self.__extract__(x), self.__unpack_sim__(sim))
        # Return the result
        return fx

    def evaluateConstraints(self, x):
        """ Evaluate the constraints using the simulation surrogates as needed.

        Args:
            x (numpy.ndarray): A 1d array containing the (embedded) design
                point to evaluate.

        Returns:
            numpy.ndarray: A 1d array containing the list of constraint
            violations (zero if no violation).

        """

        # Check for illegal input
        if isinstance(x, np.ndarray):
            if x.shape[0] != self.n:
                raise ValueError("x must have length n")
        else:
            raise ValueError("x must be a numpy array")
        # Special case if there are no constraints, just return [0]
        if self.p == 0:
            return np.zeros(1)
        # Otherwise, calculate the constraint violations
        else:
            # Evaluate the surrogate models to approximate the sim outputs
            sim = np.zeros(self.m_total)
            m_count = 0
            for i, surrogate in enumerate(self.surrogates):
                sim[m_count:m_count + self.m[i]] = surrogate.evaluate(x)
                m_count += self.m[i]
            # Evaluate the constraint functions
            cx = np.zeros(self.p)
            for i, constraint_func in enumerate(self.constraints):
                cx[i] = constraint_func(self.__extract__(x),
                                        self.__unpack_sim__(sim))
            # Return the constraint violations
            return cx

    def evaluateLagrangian(self, x):
        """ Evaluate the augmented Lagrangian using the surrogates as needed.

        Args:
            x (numpy.ndarray): A 1d array containing the (embedded) design
                point to evaluate.

        Returns:
            numpy.ndarray: A 1d array containing the result of the evaluation.

        """

        # Check for illegal input
        if isinstance(x, np.ndarray):
            if x.shape[0] != self.n:
                raise ValueError("x must have length n")
        else:
            raise ValueError("x must be a numpy array")
        # Evaluate the surrogate models to approximate the simulation outputs
        sim = np.zeros(self.m_total)
        m_count = 0
        for i, surrogate in enumerate(self.surrogates):
            sim[m_count:m_count+self.m[i]] = surrogate.evaluate(x)
            m_count += self.m[i]
        # Evaluate the objective functions
        fx = np.zeros(self.o)
        for i, obj_func in enumerate(self.objectives):
            fx[i] = obj_func(self.__extract__(x), self.__unpack_sim__(sim))
        # Evaluate the constraint functions
        Lx = np.zeros(self.o)
        if self.p > 0:
            for constraint_func in self.constraints:
                cx = constraint_func(self.__extract__(x),
                                     self.__unpack_sim__(sim))
                if cx > 0.0:
                    Lx[:] = Lx[:] + cx
        # Compute the augmented Lagrangian
        Lx[:] = self.lam * Lx[:] + fx[:]
        # Return the result
        return Lx

    def evaluateGradients(self, x):
        """ Evaluate the gradient of the augmented Lagrangian using surrogates.

        Args:
            x (numpy.ndarray): A 1d array containing the (embedded) design
                point to evaluate.

        Returns:
            numpy.ndarray: A 1d array containing the result of the evaluation.

        """

        # Check for illegal input
        if isinstance(x, np.ndarray):
            if x.shape[0] != self.n:
                raise ValueError("x must have length n")
        else:
            raise ValueError("x must be a numpy array")
        # Evaluate the surrogate models to approximate the simulation outputs
        sim = np.zeros(self.m_total)
        m_count = 0
        for i, surrogate in enumerate(self.surrogates):
            sim[m_count:m_count+self.m[i]] = surrogate.evaluate(x)
            m_count += self.m[i]
        # Evaluate the gradients of the surrogates
        if self.m_total > 0:
            dsim_dx = np.zeros((self.m_total, self.n))
            m_count = 0
            for i, surrogate in enumerate(self.surrogates):
                dsim_dx[m_count:m_count+self.m[i], :] = \
                        surrogate.gradient(x)
                m_count += self.m[i]
        # Evaluate the gradients of the objective functions
        df_dx = np.zeros((self.o, self.n))
        for i, obj_func in enumerate(self.objectives):
            # If names are used, unpack the derivative
            if self.use_names:
                df_dx_tmp = obj_func(self.__extract__(x),
                                     self.__unpack_sim__(sim),
                                     der=1)
                for j, d_name in enumerate(self.des_names):
                    if self.des_order[j] < self.n_cont:
                        df_dx[i, j] = df_dx_tmp[d_name[0]]
                df_dx[i, :self.n_cont] = ((df_dx[i, self.des_order])
                                          [:self.n_cont] /
                                          self.scale[:self.n_cont])
            # Otherwise, evaluate normally
            else:
                df_dx[i, :self.n_cont] = (obj_func(self.__extract__(x),
                                                   self.__unpack_sim__(sim),
                                                   der=1)[:self.n_cont] /
                                          self.scale[:self.n_cont])
        # Now evaluate wrt the sims
        if self.m_total > 0:
            df_dsim = np.zeros((self.o, self.m_total))
            # If names are used, pack the sims
            if self.use_names:
                for i, obj_func in enumerate(self.objectives):
                    df_dsim_tmp = obj_func(self.__extract__(x),
                                           self.__unpack_sim__(sim),
                                           der=2)
                    df_dsim[i, :] = self.__pack_sim__(df_dsim_tmp)
            else:
                for i, obj_func in enumerate(self.objectives):
                    df_dsim[i, :] = obj_func(self.__extract__(x),
                                             self.__unpack_sim__(sim),
                                             der=2)
        # Finally, evaluate the full objective Jacobian
        dfx = np.zeros((self.o, self.n))
        dfx = df_dx
        if self.m_total > 0:
            dfx = dfx + np.dot(df_dsim, dsim_dx)
        # If there are no constraints, just return zeros
        dcx = np.zeros(self.n)
        # Otherwise, calculate the constraint violation Jacobian
        if self.p > 0:
            # Evaluate the constraint functions
            cx = np.zeros(self.p)
            for i, constraint_func in enumerate(self.constraints):
                cx[i] = constraint_func(self.__extract__(x),
                                        self.__unpack_sim__(sim))
            # Evaluate the gradients of the constraint functions
            dc_dx = np.zeros((self.p, self.n))
            for i, constraint_func in enumerate(self.constraints):
                # If names are used, unpack the derivative
                if self.use_names:
                    dc_dx_tmp = constraint_func(self.__extract__(x),
                                                self.__unpack_sim__(sim),
                                                der=1)
                    for j, d_name in enumerate(self.des_names):
                        if self.des_order[j] < self.n_cont:
                            dc_dx[i, j] = dc_dx_tmp[d_name[0]]
                    dc_dx[i, :self.n_cont] = ((dc_dx[i, self.des_order])
                                              [:self.n_cont] /
                                              self.scale[:self.n_cont])
                # Otherwise, evaluate normally
                else:
                    dc_dx[i, :self.n_cont] = (constraint_func(
                                                          self.__extract__(x),
                                                          sim,
                                                          der=1)[:self.n_cont]
                                              / self.scale[:self.n_cont])
            # Now evaluate wrt the sims
            if self.m_total > 0:
                dc_dsim = np.zeros((self.p, self.m_total))
                # If names are used, sims need to be packed
                if self.use_names:
                    for i, constraint_func in enumerate(self.constraints):
                        sxx = self.__unpack_sim__(sim)
                        dc_dsim_tmp = constraint_func(self.__extract__(x),
                                                      sxx, der=2)
                        dc_dsim[i, :] = self.__pack_sim__(dc_dsim_tmp)
                # Otherwise, evaluate normally
                else:
                    for i, constraint_func in enumerate(self.constraints):
                        sxx = self.__unpack_sim__(sim)
                        dc_dsim[i, :] = constraint_func(self.__extract__(x),
                                                        sxx, der=2)
            # Finally, evaluate the full Jacobian of the constraints
            for i in range(len(self.constraints)):
                if cx[i] > 0:
                    dcx[:] = dcx[:] + dc_dx[i, :]
                    if self.m_total > 0:
                        dcx[:] = dcx[:] + np.dot(dc_dsim[i, :], dsim_dx[:, :])
        # Construct the Jacobian of the augmented Lagrangian
        dLx = np.zeros((self.o, self.n))
        for i in range(self.o):
            dLx[i, :] = dfx[i, :] + self.lam * dcx[:]
        # Return the result
        return dLx

    def addData(self, x, sx):
        """ Update the internal objective database by truly evaluating x.

        Args:
            x (numpy.ndarray): Either a numpy structured array (when 'name'
                key was given for all design variables) or a 1D array
                containing design variable values (in the order that they
                were added to the MOOP).

            sx (numpy.ndarray): The corresponding simulation outputs.

        """

        # Initialize the database if needed
        if self.n_dat == 0:
            self.data['x_vals'][0, :] = self.__embed__(x)
            self.data['f_vals'] = np.zeros((1, self.o))
            for i, obj_func in enumerate(self.objectives):
                self.data['f_vals'][0, i] = obj_func(x, sx)
            # Check if there are constraint violations to maintain
            if self.p > 0:
                self.data['c_vals'] = np.zeros((1, self.p))
                for i, constraint_func in enumerate(self.constraints):
                    self.data['c_vals'][0, i] = constraint_func(x, sx)
            else:
                self.data['c_vals'] = np.zeros((1, 1))
            self.n_dat = 1
        # Check for duplicate values (up to the design tolerance)
        elif any([np.all(np.abs(self.__embed__(x) - xj) < self.scaled_des_tols)
                  for xj in self.data['x_vals']]):
            return
        # Otherwise append the objectives
        else:
            self.data['x_vals'] = np.append(self.data['x_vals'],
                                            [self.__embed__(x)], axis=0)
            fx = np.zeros(self.o)
            for i, obj_func in enumerate(self.objectives):
                fx[i] = obj_func(x, sx)
            self.data['f_vals'] = np.append(self.data['f_vals'],
                                            [fx], axis=0)
            # Check if there are constraint violations to maintain
            if self.p > 0:
                cx = np.zeros(self.p)
                for i, constraint_func in enumerate(self.constraints):
                    cx[i] = constraint_func(x, sx)
                self.data['c_vals'] = np.append(self.data['c_vals'],
                                                [cx], axis=0)
            else:
                self.data['c_vals'] = np.append(self.data['c_vals'],
                                                [np.zeros(1)], axis=0)
            self.n_dat += 1
        return

    def iterate(self, k):
        """ Perform one iteration of ParMOO and generate a batch of candidates.

        Args:
            k (int): The iteration counter.

        Returns:
            (list): A list of ordered pairs.
            The first entry is either a 1D numpy structured array (when
            'name' key was given for all design variables) or a 2D ndarray
            where each row contains design variable values in the order
            that they were added to the MOOP. This output specifies the
            list of design points that ParMOO suggests for evaluation
            in this iteration.
            The second entry is either the name of the simulation to
            evaluate (when 'name' key was given for all design variables)
            or the integer index of the simulation to evaluate.

        """

        # Check that the iterate is a legal integer
        if isinstance(k, int):
            if k < 0:
                raise ValueError("k must be nonnegative")
        else:
            raise TypeError("k must be an int type")
        # Check that there are design variables for this problem
        if self.n == 0:
            raise AttributeError("there are no design vars for this problem")
        # Check that there are objectives
        if self.o == 0:
            raise AttributeError("there are no objectives for this problem")

        # Prepare a batch to return
        batch = []
        # Special rule for the k=0 iteration
        if k == 0:
            # Initialize the database
            self.n_dat = 0
            self.data = {'x_vals': np.zeros((1, self.n)),
                         'f_vals': np.zeros((1, self.o)),
                         'c_vals': np.zeros((1, 1))}
            # Generate search data
            for j, search in enumerate(self.searches):
                des = search.startSearch(self.scaled_lb, self.scaled_ub)
                for xi in des:
                    if self.use_names:
                        batch.append((self.__extract__(xi),
                                      self.sim_names[j][0]))
                    else:
                        batch.append((self.__extract__(xi), j))
        # Now the main loop
        else:
            x0 = np.zeros((len(self.acquisitions), self.n))
            # Add acquisition functions
            for i, acquisition in enumerate(self.acquisitions):
                x0[i, :] = acquisition.setTarget(self.data,
                                                 self.evaluateConstraints,
                                                 self.history)
            # Set up the surrogate problem
            opt = self.optimizer(self.o, self.scaled_lb, self.scaled_ub,
                                 self.hyperparams)
            opt.setObjective(self.evaluateSurrogates)
            opt.setLagrangian(self.evaluateLagrangian, self.evaluateGradients)
            opt.setConstraints(self.evaluateConstraints)
            opt.addAcquisition(*self.acquisitions)
            opt.setReset(self.resetSurrogates)
            # Solve the surrogate problem
            x_vals = opt.solve(x0)
            # Evaluate all of the simulations at the candidate solutions
            if self.s > 0:
                # For each design in the database
                for xi in x_vals:
                    # Check whether it has been evaluated by any simulation
                    for i in range(self.s):
                        xxi = self.__extract__(xi)
                        if self.use_names:
                            namei = self.sim_names[i][0]
                        else:
                            namei = i
                        if not any([np.all(np.abs(xi - self.__embed__(xj)) <
                                    self.scaled_des_tols)
                                    and namei == j for (xj, j) in batch]) \
                           and self.check_sim_db(xxi, i) is None:
                            # If not, add it to the batch
                            batch.append((xxi, namei))
                        else:
                            # Try to improve surrogate (locally then globally)
                            x_improv = self.surrogates[i].improve(xi, False)
                            while (any([any([np.all(np.abs(self.__embed__(xj)
                                                           - xk) <
                                                    self.scaled_des_tols)
                                             and namei == j for (xj, j)
                                             in batch])
                                        for xk in x_improv]) or
                                   any([self.check_sim_db(self.__extract__(xk),
                                                          i)
                                        is not None for xk in x_improv])):
                                x_improv = self.surrogates[i].improve(xi,
                                                                      True)
                            # Add improvement points to the batch
                            for xj in x_improv:
                                batch.append((self.__extract__(xj), namei))
            else:
                # If there were no simulations, just add all points to batch
                for xi in x_vals:
                    xxi = self.__extract__(xi)
                    if not any([np.all(np.abs(xxi - xj) < self.des_tols)
                                for (xj, j) in batch]):
                        batch.append((xxi, -1))
        return batch

    def updateAll(self, k, batch):
        """ Update all surrogates given a batch of freshly evaluated data.

        Args:
            k (int): The iteration counter.

            batch (list): A list of design point (x) simulation index (i)
                pairs: [(x1, i1), (x2, i2), ...]. Each 'x' is either
                a numpy structured array (when 'name' key was given for
                all design variables) or a 1D array containing design
                variable values (in the order that they were added to
                the MOOP).

        """

        # Special rules for k=0, vs k>0
        if k == 0:
            # Fit the surrogates
            self.fitSurrogates()
            # Add all points that have been fully evaluated to the database
            if self.s > 0:
                # Check every point in sim_db[0]
                for xi, si in zip(self.sim_db[0]['x_vals'],
                                  self.sim_db[0]['s_vals']):
                    # Keep track of the sim value
                    sim = np.zeros(self.m_total)
                    sim[0:self.m[0]] = si[:]
                    m_count = self.m[0]
                    is_shared = True
                    # Check against every other sim_db
                    for j in range(1, self.s):
                        is_shared = False
                        for xj, sj in zip(self.sim_db[j]['x_vals'],
                                          self.sim_db[j]['s_vals']):
                            # If found, update sim value and break loop
                            if np.all(np.abs(xi - xj) < self.scaled_des_tols):
                                sim[m_count:m_count + self.m[j]] = sj[:]
                                m_count = m_count + self.m[j]
                                is_shared = True
                                break
                        # If not found, stop checking
                        if not is_shared:
                            break
                    # If xi was in every sim_db, add it to the database
                    if is_shared:
                        self.addData(self.__extract__(xi),
                                     self.__unpack_sim__(sim))
        else:
            # If constraints are violated, increase lam
            if any([np.any(self.evaluateConstraints(self.__embed__(xi))
                           > 1.0e-4) for (xi, i) in batch]):
                self.lam = self.lam * 2.0
            # Update the surrogates
            self.updateSurrogates()
            # Add new points that have been fully evaluated to the database
            for xi in batch:
                (x, i) = xi
                xx = self.__embed__(x)
                is_shared = True
                sim = np.zeros(self.m_total)
                m_count = 0
                # Check against every other group
                if self.s > 0:
                    # Check against every other sim_db
                    for j in range(self.s):
                        is_shared = False
                        for xj, sj in zip(self.sim_db[j]['x_vals'],
                                          self.sim_db[j]['s_vals']):
                            # If found, update sim value and break loop
                            if np.all(np.abs(xx - xj) < self.scaled_des_tols):
                                sim[m_count:m_count + self.m[j]] = sj[:]
                                m_count = m_count + self.m[j]
                                is_shared = True
                                break
                        # If not found, stop checking
                        if not is_shared:
                            break
                # If xi was in every sim_db, add it to the database
                if is_shared:
                    self.addData(x, self.__unpack_sim__(sim))
        # If checkpointing is on, save the moop before continuing
        if self.checkpoint:
            self.save(filename=self.checkpointfile)
        return

    def solve(self, budget):
        """ Solve a MOOP using ParMOO.

        Args:
            budget (int): The number of iterations for the solver to run.

        """

        import logging

        # Check that the budget is a legal integer
        if isinstance(budget, int):
            if budget < 0:
                raise ValueError("budget must be nonnegative")
        else:
            raise ValueError("budget must be an int type")

        # Print logging info summary of problem setup
        logging.info(" Beginning new run of ParMOO...")
        logging.info(" summary of settings:")
        logging.info(f"   {self.n} design dimensions")
        logging.info(f"     continuous design variables: {self.n_cont}")
        logging.info(f"     categorical design variables: {self.n_cat}")
        logging.info(f"   {self.m_total} simulation outputs")
        logging.info(f"   {self.s} simulations")
        for i in range(self.s):
            logging.info(f"     {self.m[i]} outputs for simulation {i}")
            logging.info(f"     {self.searches[i].budget} search evaluations" +
                         f" in iteration 0 for simulation {i}")
        logging.info(f"   {self.o} objectives")
        logging.info(f"   {self.p} constraints")
        logging.info(f"   {len(self.acquisitions)} acquisition functions")
        logging.info("   estimated simulation evaluations per iteration:" +
                     f" {len(self.acquisitions) * self.s}")
        logging.info(f"   iteration limit: {budget}")
        logging.info(" Done.")

        # Perform iterations until budget is exceeded
        logging.info(" Entering main iteration loop:")

        # Reset the iteration start
        start = self.iteration
        for k in range(start, budget + 1):
            # Track iteration counter
            self.iteration = k
            # Generate a batch by running one iteration
            logging.info(f"   Iteration {self.iteration: >4}:")
            logging.info("     generating batch...")
            batch = self.iterate(self.iteration)
            logging.info(f"     {len(batch)} candidate designs generated.")
            if self.s > 0:
                # Evaluate the batch
                logging.info("     evaluating batch...")
                for xi in batch:
                    (x, i) = xi
                    logging.info(f"       evaluating design: {x}" +
                                 f" for simulation: {i}...")
                    sx = self.evaluateSimulation(x, i)
                    logging.info(f"         result: {sx}")
                logging.info(f"     finished evaluating {len(batch)}" +
                             " simulations.")
            logging.info("     updating models and internal databases...")
            # Update the database
            self.updateAll(self.iteration, batch)
            logging.info("   Done.")
        logging.info(" Done.")
        logging.info(f" ParMOO has successfully completed {self.iteration} " +
                     "iterations.")
        return

    def getPF(self):
        """ Extract the nondominated and efficient sets from internal database.

        Returns:
            A discrete approximation of the Pareto front and efficient set.

            If all design names were given, then this is a 1d numpy
            structured array whose fields match the names for design
            variables, objectives, and constraints (if any).

            Otherwise, this is a dict containing the following keys:
             * x_vals (numpy.ndarray): A 2d array containing a list
               of nondominated points discretely approximating the
               Pareto front.
             * f_vals (numpy.ndarray): A 2d array containing the list
               of corresponding efficient design points.
             * c_vals (numpy.ndarray): A 2d array containing the list
               of corresponding constraint satisfaction scores,
               all less than or equal to 0.

        """

        from parmoo.util import updatePF

        # Get the solutions using function call
        if self.n_dat > 0:
            pf = updatePF(self.data, {})
        else:
            pf = {'x_vals': np.zeros(0),
                  'f_vals': np.zeros(0),
                  'c_vals': np.zeros(0)}
        # Check if names are used
        if self.use_names:
            # Build the data type
            dt = self.des_names.copy()
            for fname in self.obj_names:
                dt.append(fname)
            for cname in self.const_names:
                dt.append(cname)
            # Initialize result array
            result = np.zeros(pf['x_vals'].shape[0], dtype=dt)
            # Extract all results
            if self.n_dat > 0:
                x_vals = np.asarray([self.__extract__(xi)
                                     for xi in pf['x_vals']])
                for (name, t) in self.des_names:
                    result[name][:] = x_vals[name][:]
                for i, (name, t) in enumerate(self.obj_names):
                    result[name][:] = pf['f_vals'][:, i]
                for i, (name, t) in enumerate(self.const_names):
                    result[name][:] = pf['c_vals'][:, i]
        else:
            result = {'x_vals': np.zeros(0), 'f_vals': np.zeros(0)}
            if self.n_dat > 0:
                result = {'x_vals': np.asarray([self.__extract__(xi)
                                                for xi in pf['x_vals']]),
                          'f_vals': pf['f_vals'].copy()}
            if self.p > 0:
                result['c_vals'] = pf['c_vals'].copy()
        return result

    def getSimulationData(self):
        """ Extract all computed simulation outputs from database.

        Returns:
            (dict or list) Either a dictionary or list of dictionaries
            containing every point where a simulation was evaluated.

            If all design names were given, then the result is a dict.
            Each key is the name for a different simulation, and each value
            is a 1d numpy structured array whose keys match the
            names for each design variables plus an
            additional 'out' key for simulation outputs.

            Otherwise, this is a list of s (number of simulations) dicts,
            each dict containing the following keys:
             * x_vals (numpy.ndarray): A 2d array containing a list
               of design points that have been evaluated for this
               simulation.
             * s_vals (numpy.ndarray): A 1d or 2d array containing
               the list of corresponding simulation outputs.

        """

        # Check if names are used
        if self.use_names:
            # Initialize result dictionary
            result = {}
            # For each simulation
            for i, sname in enumerate(self.sim_names):
                # Extract all results
                x_vals = np.asarray([self.__extract__(xi)
                                     for xi in self.sim_db[i]['x_vals']])
                # Build the datatype
                dt = self.des_names.copy()
                if len(sname) == 2:
                    dt.append(('out', sname[1]))
                else:
                    dt.append(('out', sname[1], sname[2]))
                # Initialize result array for sname[i]
                result[sname[0]] = np.zeros(self.sim_db[i]['n'], dtype=dt)
                if self.sim_db[i]['n'] > 0:
                    # Copy results
                    for (name, t) in self.des_names:
                        result[sname[0]][name][:] = x_vals[name][:]
                    if len(sname) == 2:
                        result[sname[0]]['out'] = self.sim_db[i]['s_vals'][:,
                                                                           0]
                    else:
                        result[sname[0]]['out'] = self.sim_db[i]['s_vals']
            return result
        else:
            # Initialize result list
            result = []
            # For each simulation
            for i in range(self.s):
                if self.sim_db[i]['n'] > 0:
                    # Extract all results
                    x_vals = np.asarray([self.__extract__(xi)
                                         for xi in self.sim_db[i]['x_vals']])
                    result.append({'x_vals': x_vals,
                                   's_vals': self.sim_db[i]['s_vals'].copy()})
                else:
                    result.append({'x_vals': np.zeros(0),
                                   's_vals': np.zeros(0)})
            return result

    def getObjectiveData(self):
        """ Extract all computed objective scores from database.

        Returns:
            A database of all designs that have been fully evaluated,
            and their corresponding objective scores.

            If all design names were given, then this is a 1d numpy
            structured array whose fields match the names for design
            variables, objectives, and constraints (if any).

            Otherwise, this is a dict containing the following keys:
             * x_vals (numpy.ndarray): A 2d array containing a list
               of all fully evaluated design points.
             * f_vals (numpy.ndarray): A 2d array containing the list
               of corresponding objective values.
             * c_vals (numpy.ndarray): A 2d array containing the list
               of corresponding constraint satisfaction scores,
               all less than or equal to 0.

        """

        # Check if names are used
        if self.use_names:
            # Build the data type
            dt = self.des_names.copy()
            for fname in self.obj_names:
                dt.append(fname)
            for cname in self.const_names:
                dt.append(cname)
            # Initialize result array
            if self.n_dat > 0:
                result = np.zeros(self.data['x_vals'].shape[0], dtype=dt)
            else:
                result = np.zeros(0, dtype=dt)
            # Extract all results
            if self.n_dat > 0:
                x_vals = np.asarray([self.__extract__(xi)
                                     for xi in self.data['x_vals']])
                for (name, t) in self.des_names:
                    result[name][:] = x_vals[name][:]
                for i, (name, t) in enumerate(self.obj_names):
                    result[name][:] = self.data['f_vals'][:, i]
                for i, (name, t) in enumerate(self.const_names):
                    result[name][:] = self.data['c_vals'][:, i]
        else:
            result = {'x_vals': np.zeros(0), 'f_vals': np.zeros(0)}
            if self.n_dat > 0:
                result = {'x_vals': np.asarray([self.__extract__(xi) for
                                                xi in self.data['x_vals']]),
                          'f_vals': self.data['f_vals'].copy()}
                if self.p > 0:
                    result['c_vals'] = self.data['c_vals'].copy()
        return result

    def save(self, filename="parmoo"):
        """ Serialize and save the MOOP object and all of its dependencies.

        Args:
            filename (string, optional): The filepath to serialized
                checkpointing file(s). Do not include file extensions,
                they will be appended automaically. May create
                several save files with extensions of this name, in order
                to recursively save dependencies objects. Defaults to
                the value "parmoo" (filename will be "parmoo.moop").

        """

        import shutil
        import pickle
        import codecs
        from os.path import exists as file_exists

        # Check whether the file exists first
        exists = file_exists(filename + ".moop")
        if exists and self.new_checkpoint:
            raise OSError("Creating a new checkpoint file, but " +
                          filename + ".moop already exists! " +
                          "Move the existing file to a new location or " +
                          "delete it, so that ParMOO doesn't accidentally " +
                          "overwrite your data...")
        # Create a serializable ParMOO dictionary by replacing function refs
        # with funcion/module names
        parmoo_state = {'n': self.n,
                        'm': self.m,
                        'm_total': self.m_total,
                        'o': self.o,
                        'p': self.p,
                        's': self.s,
                        'n_dat': self.n_dat,
                        'lb': self.lb,
                        'ub': self.ub,
                        'n_cat_d': self.n_cat_d,
                        'n_cat': self.n_cat,
                        'n_cont': self.n_cont,
                        'n_lvls': self.n_lvls,
                        'des_order': self.des_order,
                        'cat_names': self.cat_names,
                        'sim_names': self.sim_names,
                        'des_names': self.des_names,
                        'obj_names': self.obj_names,
                        'const_names': self.const_names,
                        'lam': self.lam,
                        'des_tols': self.des_tols,
                        'hyperparams': self.hyperparams,
                        'history': self.history,
                        'use_names': self.use_names,
                        'iteration': self.iteration,
                        'checkpoint': self.checkpoint,
                        'checkpoint_data': self.checkpoint_data,
                        'checkpointfile': self.checkpointfile}
        # Serialize numpy arrays
        if isinstance(self.scale, np.ndarray):
            parmoo_state['scale'] = self.scale.tolist()
        else:
            parmoo_state['scale'] = self.scale
        if isinstance(self.scaled_lb, np.ndarray):
            parmoo_state['scaled_lb'] = self.scaled_lb.tolist()
        else:
            parmoo_state['scaled_lb'] = self.scaled_lb
        if isinstance(self.scaled_ub, np.ndarray):
            parmoo_state['scaled_ub'] = self.scaled_ub.tolist()
        else:
            parmoo_state['scaled_ub'] = self.scaled_ub
        if isinstance(self.scaled_des_tols, np.ndarray):
            parmoo_state['scaled_des_tols'] = self.scaled_des_tols.tolist()
        else:
            parmoo_state['scaled_des_tols'] = self.scaled_des_tols
        if isinstance(self.cat_des_tols, np.ndarray):
            parmoo_state['cat_des_tols'] = self.cat_des_tols.tolist()
        else:
            parmoo_state['cat_des_tols'] = self.cat_des_tols
        if isinstance(self.cat_lb, np.ndarray):
            parmoo_state['cat_lb'] = self.cat_lb.tolist()
        else:
            parmoo_state['cat_lb'] = self.cat_lb
        if isinstance(self.cat_scale, np.ndarray):
            parmoo_state['cat_scale'] = self.cat_scale.tolist()
        else:
            parmoo_state['cat_scale'] = self.cat_scale
        if isinstance(self.RSVT, np.ndarray):
            parmoo_state['RSVT'] = self.RSVT.tolist()
        else:
            parmoo_state['RSVT'] = self.RSVT
        if isinstance(self.mean, np.ndarray):
            parmoo_state['mean'] = self.mean.tolist()
        else:
            parmoo_state['mean'] = self.mean
        # Serialize internal databases
        parmoo_state['data'] = {}
        if 'x_vals' in self.data.keys():
            parmoo_state['data']['x_vals'] = self.data['x_vals'].tolist()
        if 'f_vals' in self.data.keys():
            parmoo_state['data']['f_vals'] = self.data['f_vals'].tolist()
        if 'c_vals' in self.data.keys():
            parmoo_state['data']['c_vals'] = self.data['c_vals'].tolist()
        parmoo_state['sim_db'] = []
        for dbi in self.sim_db:
            parmoo_state['sim_db'].append({'x_vals': dbi['x_vals'].tolist(),
                                           's_vals': dbi['s_vals'].tolist(),
                                           'n': dbi['n'],
                                           'old': dbi['old']})
        # Add names for all callables (functions/objects)
        parmoo_state['objectives'] = []
        parmoo_state['objectives_info'] = []
        for fi in self.objectives:
            if type(fi).__name__ == "function":
                parmoo_state['objectives'].append((fi.__name__, fi.__module__))
                parmoo_state['objectives_info'].append("function")
            else:
                parmoo_state['objectives'].append((fi.__class__.__name__,
                                                   fi.__class__.__module__))
                parmoo_state['objectives_info'].append(
                        codecs.encode(pickle.dumps(fi), "base64").decode())
        parmoo_state['sim_funcs'] = []
        parmoo_state['sim_funcs_info'] = []
        for si in self.sim_funcs:
            if type(si).__name__ == "function":
                parmoo_state['sim_funcs'].append((si.__name__, si.__module__))
                parmoo_state['sim_funcs_info'].append("function")
            else:
                parmoo_state['sim_funcs'].append((si.__class__.__name__,
                                                  si.__class__.__module__))
                parmoo_state['sim_funcs_info'].append(
                        codecs.encode(pickle.dumps(si), "base64").decode())
        parmoo_state['constraints'] = []
        parmoo_state['constraints_info'] = []
        for ci in self.constraints:
            if type(si).__name__ == "function":
                parmoo_state['constraints'].append((ci.__name__,
                                                    ci.__module__))
                parmoo_state['constraints_info'].append("function")
            else:
                parmoo_state['constraints'].append((ci.__class__.__name__,
                                                    ci.__class__.__module__))
                parmoo_state['constraints_info'].append(
                        codecs.encode(pickle.dumps(ci), "base64").decode())
        # Store names/modules of object classes
        parmoo_state['optimizer'] = (self.optimizer.__name__,
                                     self.optimizer.__module__)
        # Store names/modules of object instances
        parmoo_state['searches'] = [(search.__class__.__name__,
                                     search.__class__.__module__)
                                    for search in self.searches]
        parmoo_state['surrogates'] = [(sur.__class__.__name__,
                                       sur.__class__.__module__)
                                      for sur in self.surrogates]
        parmoo_state['acquisitions'] = [(acq.__class__.__name__,
                                         acq.__class__.__module__)
                                        for acq in self.acquisitions]
        # Try to save search states
        for i, search in enumerate(self.searches):
            try:
                fname = filename + ".search." + str(i + 1)
                fname_tmp = "." + fname + ".swap"
                search.save(fname_tmp)
                shutil.move(fname_tmp, fname)
            except NotImplementedError:
                pass
        # Try to save surrogate states
        for i, surrogate in enumerate(self.surrogates):
            try:
                fname = filename + ".surrogate." + str(i + 1)
                fname_tmp = "." + fname + ".swap"
                surrogate.save(fname_tmp)
                shutil.move(fname_tmp, fname)
            except NotImplementedError:
                pass
        # Try to save acquisition states
        for i, acquisition in enumerate(self.acquisitions):
            try:
                fname = filename + ".acquisition." + str(i + 1)
                fname_tmp = "." + fname + ".swap"
                acquisition.save(fname_tmp)
                shutil.move(fname_tmp, fname)
            except NotImplementedError:
                pass
        # Save serialized dictionary object
        fname = filename + ".moop"
        fname_tmp = "." + fname + ".swap"
        with open(fname_tmp, 'w') as fp:
            json.dump(parmoo_state, fp)
        shutil.move(fname_tmp, fname)
        self.new_checkpoint = False
        return

    def load(self, filename="parmoo"):
        """ Load a serialized MOOP object and all of its dependencies.

        Args:
            filename (string, optional): The filepath to serialized
                checkpointing file(s). Do not include file extensions,
                they will be appended automaically. May also load from
                other save files with different extensions of this name,
                in order to recursively load dependencies objects.
                Defaults to the value "parmoo" (filename will be
                "parmoo.moop").

        """

        from importlib import import_module
        import pickle
        import codecs

        PYDOCS = "https://docs.python.org/3/tutorial/modules.html" + \
                 "#the-module-search-path"

        # Load the serialized dictionary object
        fname = filename + ".moop"
        with open(fname, 'r') as fp:
            parmoo_state = json.load(fp)

        # Reload serialized intrinsic types (scalar values and Python lists)
        self.n = parmoo_state['n']
        self.m = parmoo_state['m']
        self.m_total = parmoo_state['m_total']
        self.o = parmoo_state['o']
        self.p = parmoo_state['p']
        self.s = parmoo_state['s']
        self.n_dat = parmoo_state['n_dat']
        self.lb = parmoo_state['lb']
        self.ub = parmoo_state['ub']
        self.n_cat_d = parmoo_state['n_cat_d']
        self.n_cat = parmoo_state['n_cat']
        self.n_cont = parmoo_state['n_cont']
        self.n_lvls = parmoo_state['n_lvls']
        self.des_order = parmoo_state['des_order']
        self.cat_names = parmoo_state['cat_names']
        self.sim_names = parmoo_state['sim_names']
        self.des_names = parmoo_state['des_names']
        self.obj_names = parmoo_state['obj_names']
        self.const_names = parmoo_state['const_names']
        self.lam = parmoo_state['lam']
        self.des_tols = parmoo_state['des_tols']
        self.hyperparams = parmoo_state['hyperparams']
        self.history = parmoo_state['history']
        self.use_names = parmoo_state['use_names']
        self.iteration = parmoo_state['iteration']
        self.checkpoint = parmoo_state['checkpoint']
        self.checkpoint_data = parmoo_state['checkpoint_data']
        self.checkpointfile = parmoo_state['checkpointfile']
        # Reload serialize numpy arrays
        self.scale = np.array(parmoo_state['scale'])
        self.scaled_lb = np.array(parmoo_state['scaled_lb'])
        self.scaled_ub = np.array(parmoo_state['scaled_ub'])
        self.scaled_des_tols = np.array(parmoo_state['scaled_des_tols'])
        self.cat_des_tols = np.array(parmoo_state['cat_des_tols'])
        self.cat_lb = np.array(parmoo_state['cat_lb'])
        self.cat_scale = np.array(parmoo_state['cat_scale'])
        self.RSVT = np.array(parmoo_state['RSVT'])
        self.mean = np.array(parmoo_state['mean'])
        # Reload serialized internal databases
        self.data = {}
        if 'x_vals' in parmoo_state['data'].keys():
            self.data['x_vals'] = np.array(parmoo_state['data']['x_vals'])
        if 'f_vals' in parmoo_state['data'].keys():
            self.data['f_vals'] = np.array(parmoo_state['data']['f_vals'])
        if 'c_vals' in parmoo_state['data'].keys():
            self.data['c_vals'] = np.array(parmoo_state['data']['c_vals'])
        self.sim_db = []
        for dbi in parmoo_state['sim_db']:
            self.sim_db.append({'x_vals': np.array(dbi['x_vals']),
                                's_vals': np.array(dbi['s_vals']),
                                'n': dbi['n'],
                                'old': dbi['old']})
        # Recover callables (functions/objects) by name
        self.objectives = []
        for (obj_name, obj_mod), info in zip(parmoo_state['objectives'],
                                             parmoo_state['objectives_info']):
            try:
                mod = import_module(obj_mod)
            except ModuleNotFoundError:
                raise ModuleNotFoundError("module: " + obj_mod +
                                          " could not be loaded. " +
                                          "Please make sure that " + obj_mod +
                                          " exists on this machine and is " +
                                          "part of the module " +
                                          " search path: " + PYDOCS)
            try:
                obj_ptr = getattr(mod, obj_name)
            except KeyError:
                raise KeyError("function: " + obj_name +
                               " defined in " + obj_mod +
                               " could not be loaded." +
                               "Please make sure that " + obj_name +
                               " is defined in " + obj_mod +
                               " with global scope and try again.")
            if info == "function":
                toadd = obj_ptr
            else:
                toadd = pickle.loads(codecs.decode(info.encode(), "base64"))
            self.objectives.append(toadd)
        self.sim_funcs = []
        for (sim_name, sim_mod), info in zip(parmoo_state['sim_funcs'],
                                             parmoo_state['sim_funcs_info']):
            try:
                mod = import_module(sim_mod)
            except ModuleNotFoundError:
                raise ModuleNotFoundError("module: " + sim_mod +
                                          " could not be loaded. " +
                                          "Please make sure that " + sim_mod +
                                          " exists on this machine and is " +
                                          "part of the Module " +
                                          " search path: " + PYDOCS)
            try:
                sim_ptr = getattr(mod, sim_name)
            except KeyError:
                raise KeyError("function: " + sim_name +
                               " defined in " + sim_mod +
                               " could not be loaded." +
                               "Please make sure that " + sim_name +
                               " is defined in " + sim_mod +
                               " with global scope and try again.")
            if info == "function":
                toadd = sim_ptr
            else:
                toadd = pickle.loads(codecs.decode(info.encode(), "base64"))
            self.sim_funcs.append(toadd)
        self.constraints = []
        for (const_name, const_mod), info in \
                zip(parmoo_state['constraints'],
                    parmoo_state['constraints_info']):
            try:
                mod = import_module(const_mod)
            except ModuleNotFoundError:
                raise ModuleNotFoundError("module: " + const_mod +
                                          " could not be loaded. " + "Please" +
                                          " make sure that " + const_mod +
                                          " exists on this machine and is " +
                                          "part of the module " +
                                          " search path: " + PYDOCS)
            try:
                const_ptr = getattr(mod, const_name)
            except KeyError:
                raise KeyError("function: " + const_name +
                               " defined in " + const_mod +
                               " could not be loaded." +
                               "Please make sure that " + const_name +
                               " is defined in " + const_mod +
                               " with global scope and try again.")
            if info == "function":
                toadd = const_ptr
            else:
                toadd = pickle.loads(codecs.decode(info.encode(), "base64"))
            self.constraints.append(toadd)
        # Recover object classes
        mod = import_module(parmoo_state['optimizer'][1])
        self.optimizer = getattr(mod, parmoo_state['optimizer'][0])
        # Recover names/modules of object instances
        self.searches = []
        for i, (search_name, search_mod) in enumerate(
                                                parmoo_state['searches']):
            mod = import_module(search_mod)
            new_search = getattr(mod, search_name)
            toadd = new_search(self.m[i], self.scaled_lb, self.scaled_ub,
                               self.hyperparams)
            try:
                fname = filename + ".search." + str(i + 1)
                toadd.load(fname)
            except NotImplementedError:
                pass
            self.searches.append(toadd)
        self.surrogates = []
        for i, (sur_name, sur_mod) in enumerate(parmoo_state['surrogates']):
            mod = import_module(sur_mod)
            new_sur = getattr(mod, sur_name)
            toadd = new_sur(self.m[i], self.scaled_lb, self.scaled_ub,
                            self.hyperparams)
            try:
                fname = filename + ".surrogate." + str(i + 1)
                toadd.load(fname)
            except NotImplementedError:
                pass
            self.surrogates.append(toadd)
        self.acquisitions = []
        for i, (acq_name, acq_mod) in enumerate(parmoo_state['acquisitions']):
            mod = import_module(acq_mod)
            new_acq = getattr(mod, acq_name)
            toadd = new_acq(self.o, self.scaled_lb, self.scaled_ub,
                            self.hyperparams)
            try:
                fname = filename + ".acquisition." + str(i + 1)
                toadd.load(fname)
            except NotImplementedError:
                pass
            self.acquisitions.append(toadd)
        self.new_checkpoint = False
        self.new_data = False
        return

    def savedata(self, x, sx, s_name, filename="parmoo"):
        """ Save the current simulation database for this MOOP.

        Args:
            filename (string, optional): The filepath to the checkpointing
                file(s). Do not include file extensions, they will be
                appended automaically. Defaults to the value "parmoo"
                (filename will be "parmoo.simdb.json").

        """

        from os.path import exists as file_exists

        # Check whether file exists first
        exists = file_exists(filename + ".simdb.json")
        if exists and self.new_data:
            raise OSError("Creating a new save file, but " +
                          filename + ".simdb.json already exists! " +
                          "Move the existing file to a new location or " +
                          "delete it so that ParMOO doesn't overwrite your " +
                          "existing data...")
        # Unpack x/sx pair into a dict for saving
        if self.use_names:
            toadd = {'sim_id': s_name}
            for key in x.names:
                toadd[key] = x[key]
            for key in sx.names:
                if isinstance(sx[key], np.ndarray):
                    toadd[key] = sx[key].tolist()
                else:
                    toadd[key] = sx[key]
        else:
            toadd = {'x_vals': x.tolist(),
                     's_vals': sx.tolist(),
                     'sim_id': s_name}
        # Save in file with proper exension
        fname = filename + ".simdb.json"
        with open(fname, 'a') as fp:
            json.dump(toadd, fp)
        self.new_data = False
        return
