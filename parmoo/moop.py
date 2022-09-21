
""" Contains the MOOP class for defining multiobjective optimization problems.

parmoo.moop.MOOP is the base class for defining and solving multiobjective
optimization problems (MOOPs). Each MOOP object may contain several
simulations, specified using dictionaries.

"""

import numpy as np
import json
from parmoo import structs
import inspect
import pandas as pd


class MOOP:
    """ Class for defining a multiobjective optimization problem (MOOP).

    Upon initialization, supply a scalar optimization procedure and
    dictionary of hyperparameters using the default constructor:
     * ``MOOP.__init__(ScalarOpt, [hyperparams={}])``

    Class methods are summarized below.

    To define the MOOP, add each design variable, simulation, objective, and
    constraint (in that order) by using the following functions:
     * ``MOOP.addDesign(*args)``
     * ``MOOP.addSimulation(*args)``
     * ``MOOP.addObjective(*args)``
     * ``MOOP.addConstraint(*args)``

    Next, define your solver.

    Acquisition functions (used for scalarizing problems/setting targets) are
    added using:
     * ``MOOP.addAcquisition(*args)``

    After creating a MOOP, the following methods may be useful for getting
    the numpy.dtype of the input/output arrays:
     * ``MOOP.getDesignType()``
     * ``MOOP.getSimulationType()``
     * ``MOOP.getObjectiveType()``
     * ``MOOP.getConstraintType()``

    The following methods are used to save/load ParMOO objects from memory:
     * ``MOOP.save([filename="parmoo"])``
     * ``MOOP.load([filename="parmoo"])``

    To turn on checkpointing, use:
     * ``MOOP.setCheckpoint(checkpoint, [checkpoint_data, filename])``

    ParMOO also offers logging. To turn on logging, activate INFO-level
    logging by importing Python's built-in logging module.

    After defining the MOOP and setting up checkpointing and logging info,
    use the following method to solve the MOOP (serially):
     * ``MOOP.solve(budget)``

    The following methods are used for solving the MOOP and managing the
    internal simulation/objective databases:
     * ``MOOP.check_sim_db(x, s_name)``
     * ``MOOP.update_sim_db(x, sx, s_name)``
     * ``MOOP.evaluateSimulation(x, s_name)``
     * ``MOOP.addData(x, sx)``
     * ``MOOP.iterate(k)``
     * ``MOOP.updateAll(k, batch)``

    Finally, the following methods are used to retrieve data after the
    problem has been solved:
     * ``MOOP.getPF()``
     * ``MOOP.getSimulationData()``
     * ``MOOP.getObjectiveData()``

    The following methods are not recommended for external usage:
     * ``MOOP.__extract__(x)``
     * ``MOOP.__embed__(x)``
     * ``MOOP.__generate_encoding__()``
     * ``MOOP.__unpack_sim__(sx)``
     * ``MOOP.__pack_sim__(sx)``
     * ``MOOP.fitSurrogates()``
     * ``MOOP.updateSurrogates()``
     * ``MOOP.evaluateSurrogates(x)``
     * ``MOOP.resetSurrogates(center)``
     * ``MOOP.evaluateConstraints(x)``
     * ``MOOP.evaluatePenalty(x)``
     * ``MOOP.evaluateGradients(x)``

    """

    # Slots for the MOOP class
    __slots__ = ['n', 'm', 'm_total', 'o', 'p', 's', 'n_dat',
                 'cont_lb', 'cont_ub', 'int_lb', 'int_ub',
                 'n_cat_d', 'n_custom_d', 'cat_lb', 'cat_scale', 'RSVT',
                 'mean', 'custom_embedders', 'custom_extracters',
                 'n_cat', 'n_cont', 'n_int', 'n_custom', 'n_raw', 'n_lvls',
                 'des_order', 'cat_names', 'sim_names',
                 'des_names', 'obj_names', 'const_names',
                 'lam', 'epsilon', 'objectives', 'data', 'sim_funcs',
                 'sim_db', 'des_tols', 'searches', 'surrogates', 'optimizer',
                 'constraints', 'hyperparams', 'acquisitions', 'history',
                 'scale', 'scaled_lb', 'scaled_ub', 'scaled_des_tols',
                 'cat_des_tols', 'custom_des_tols', 'use_names', 'iteration',
                 'checkpoint', 'checkpointfile', 'checkpoint_data',
                 'new_checkpoint', 'new_data']

    def __embed__(self, x):
        """ Embed a design input as n-dimensional vector for ParMOO.

        Args:
            x (numpy.ndarray or numpy structured array): Either a numpy
                structured array (when working with named design variables)
                or a 1D numpy.ndarray containing design variable values.
                Note that when working in unnamed mode, the design variable
                indices were assigned in the order that they were added to
                the MOOP using `MOOP.addDesign(*args)`.

        Returns:
            numpy.ndarray: A 1D array containing the embedded design vector.

        """

        # Unpack x into an ordered unstructured array
        x_tmp = np.zeros(self.n_cont + self.n_cat + self.n_int + self.n_custom
                         + self.n_raw)
        if self.use_names:
            x_labels = []
            for d_name in self.des_names:
                x_labels.append(x[d_name[0]])
            for i, j in enumerate(self.des_order):
                if ((j in range(self.n_cont+self.n_int,
                                self.n_cont+self.n_int+self.n_cat))
                    and (len(self.cat_names[j - self.n_cont - self.n_int])
                         > 0)):
                    x_tmp[j] = float(self.cat_names[j - self.n_cont -
                                                    self.n_int].index(
                                                                x_labels[i]))
                elif (j in range(self.n_cont+self.n_int+self.n_cat,
                                 self.n_cont+self.n_int+self.n_cat +
                                 self.n_custom)):
                    x_tmp[j] = i
                else:
                    x_tmp[j] = x_labels[i]
        else:
            x_tmp[self.des_order] = x[:]
        # Create the output array
        xx = np.zeros(self.n)
        # Rescale the continuous and integer variables
        start = 0
        end = self.n_cont
        xx[start:end] = ((x_tmp[start:end] - self.cont_lb[:]) /
                         self.scale[start:end] + self.scaled_lb[start:end])
        # Pull inside bounding box, in case perturbed outside
        xx[start:end] = np.maximum(xx[start:end], self.scaled_lb[start:end])
        xx[start:end] = np.minimum(xx[start:end], self.scaled_ub[start:end])
        # Rescale the continuous and integer variables
        start = end
        end = start + self.n_int
        xx[start:end] = ((x_tmp[start:end] - self.int_lb[:]) /
                         self.scale[start:end] + self.scaled_lb[start:end])
        # Pull inside bounding box, in case perturbed outside
        xx[start:end] = np.maximum(xx[start:end], self.scaled_lb[start:end])
        xx[start:end] = np.minimum(xx[start:end], self.scaled_ub[start:end])
        # Embed the categorical variables
        if self.n_cat_d > 0:
            start = end
            end = start + self.n_cat_d
            bvec = np.zeros(sum(self.n_lvls))
            count = 0
            for i, n_lvl in enumerate(self.n_lvls):
                bvec[count + int(x_tmp[start + i])] = 1.0
                count += n_lvl
            bvec -= self.mean
            xx[start:end] = ((np.matmul(self.RSVT, bvec) - self.cat_lb[:])
                             / self.scale[start:end]
                             + self.scaled_lb[start:end])
            # Pull inside bounding box, in case perturbed outside
            xx[start:end] = np.maximum(xx[start:end],
                                       self.scaled_lb[start:end])
            xx[start:end] = np.minimum(xx[start:end],
                                       self.scaled_ub[start:end])
        # Embed the custom variables
        for i, embed_i in enumerate(self.custom_embedders):
            start = end
            end = start + self.n_custom_d[i]
            if self.use_names:
                if end - start > 1:
                    xx[start:end] = embed_i(x_labels[int(x_tmp[self.n_cont +
                                                               self.n_cat +
                                                               self.n_int +
                                                               i])])
                # Special rule for self.n_custom_d = 1
                else:
                    xx[start] = embed_i(x_labels[int(x_tmp[self.n_cont +
                                                           self.n_cat +
                                                           self.n_int + i])])
            else:
                xx[start:end] = embed_i(x_tmp[self.n_cont + self.n_cat +
                                              self.n_int + i])
        # Embed the raw variables
        start = end
        end = start + self.n_raw
        xx[start:end] = x_tmp[self.n_cont + self.n_cat + self.n_int +
                              self.n_custom:]
        return xx

    def __extract__(self, x):
        """ Extract a design variable from an n-dimensional vector.

        Args:
            x (numpy.ndarray): A 1D numpy.ndarray containing the embedded
                design variable.

        Returns:
            numpy.ndarray or numpy structured array: Either a numpy
            structured array (when using named variables) or a 1D
            numpy.ndarray containing the extracted design variable values.
            Note that when working in unnamed mode, the design variable
            indices were assigned in the order that they were added to
            the MOOP using `MOOP.addDesign(*args)`.

        """

        # Create the output array
        xx = np.zeros(self.n_cont + self.n_cat + self.n_int + self.n_custom
                      + self.n_raw)
        # Descale the continuous variables
        start = 0
        end = self.n_cont
        xx[start:end] = ((x[start:end] - self.scaled_lb[start:end])
                         * self.scale[start:end] + self.cont_lb[:])
        # Pull inside bounding box, in case perturbed outside
        xx[start:end] = np.maximum(xx[start:end], self.cont_lb[:])
        xx[start:end] = np.minimum(xx[start:end], self.cont_ub[:])
        # Descale the integer variables
        start = end
        end = start + self.n_int
        xx[start:end] = ((x[start:end] - self.scaled_lb[start:end])
                         * self.scale[start:end] + self.int_lb[:])
        # Pull inside bounding box, in case perturbed outside
        xx[start:end] = np.maximum(xx[start:end], self.int_lb[:])
        xx[start:end] = np.minimum(xx[start:end], self.int_ub[:])
        # Bin the integer variables
        for i in range(self.n_cont, self.n_cont + self.n_int):
            xx[i] = int(xx[i])
        # Extract categorical variables
        if self.n_cat_d > 0:
            start = end
            end = start + self.n_cat_d
            bvec = (np.matmul(np.transpose(self.RSVT),
                              (x[start:end]
                               - self.scaled_lb[start:end])
                              * self.scale[start:end]
                              + self.cat_lb[:])
                    + self.mean)
            count = 0
            for i, n_lvl in enumerate(self.n_lvls):
                xx[start+i] = np.argmax(bvec[count:count+n_lvl])
                count += n_lvl
        # Extract custom variables
        for i, ex_i in enumerate(self.custom_extracters):
            start = end
            end = start + self.n_custom_d[i]
            if not self.use_names:
                xx[self.n_cont + self.n_cat + self.n_int + i] = \
                    ex_i(x[start:end])
        # Extract the raw variables
        start = end
        end = start + self.n_raw
        xx[self.n_cont + self.n_cat + self.n_int + self.n_custom:] = \
            x[start:end]
        # Unshuffle xx and pack into a numpy structured array
        if self.use_names:
            out = np.zeros(1, dtype=np.dtype(self.des_names))
            n_customs = 0
            for i, j in enumerate(self.des_order):
                # Unpack categorical variables when cat_names given
                if ((j in range(self.n_cont+self.n_int,
                                self.n_cont+self.n_int+self.n_cat))
                    and (len(self.cat_names[j - self.n_cont - self.n_int])
                         > 0)):
                    out[self.des_names[i][0]] = (self.cat_names[j -
                                                                self.n_cont -
                                                                self.n_int]
                                                               [int(xx[j])])
                # Unpack custom variables
                elif (j in range(self.n_cont + self.n_int + self.n_cat,
                                 self.n_cont + self.n_int + self.n_cat +
                                 self.n_custom)):
                    start = (self.n_cont + self.n_cat_d + self.n_int +
                             sum(self.n_custom_d[:n_customs]))
                    end = start + self.n_custom_d[n_customs]
                    exi = self.custom_extracters[n_customs]
                    if end - start > 1:
                        out[self.des_names[i][0]] = exi(x[start:end])
                    # Special rule for self.n_custom_d = 1
                    else:
                        out[self.des_names[i][0]] = exi(x[start])
                    n_customs += 1  # increment counter
                else:
                    out[self.des_names[i][0]] = xx[j]
            return out[0]
        else:
            return xx[self.des_order]

    def __generate_encoding__(self):
        """ Generate the encoding matrices for this MOOP. """

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
            sx (numpy.ndarray): A 1D numpy.ndarray containing the vectorized
                simulation output(s).

        Returns:
            numpy.ndarray or numpy structured array: Either a numpy structured
            array (when operating with named variables) or the unmodified
            input sx.

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
            sx (numpy.ndarray or numpy structured array): A numpy structured
                array (when operating with named variables) or a m-dimensional
                vector.

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
        self.cont_lb = []
        self.cont_ub = []
        self.int_lb = []
        self.int_ub = []
        self.n_cat = 0
        self.n_cont = 0
        self.n_int = 0
        self.n_custom = 0
        self.n_raw = 0
        self.cat_names = []
        self.n_cat_d = 0
        self.n_custom_d = []
        self.n_lvls = []
        # Initialize the scale
        self.scale = []
        self.scaled_lb = []
        self.scaled_ub = []
        self.scaled_des_tols = []
        self.cat_des_tols = []
        self.custom_des_tols = []
        self.cat_lb = []
        self.cat_scale = []
        self.epsilon = 1.0e-8
        # Initialize the embedding transformation
        self.RSVT = []
        self.mean = []
        self.custom_embedders = []
        self.custom_extracters = []
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
        # Initialize the penalty / Lagrange multiplier
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
        except BaseException:
            raise TypeError("opt_func must be a derivative of the "
                            + "SurrogateOptimizer abstract class")
        if not isinstance(opt, structs.SurrogateOptimizer):
            raise TypeError("opt_func must be a derivative of the "
                            + "SurrogateOptimizer abstract class")
        self.optimizer = opt_func
        return

    def addDesign(self, *args):
        """ Add a new design variables to the MOOP.

        Append new design variables to the problem. Note that every design
        variable must be added before any simulations or acquisition functions
        can be added since the number of design variables is used to infer
        the size of the simulation databases and the acquisition function
        initialization.

        Args:
            args (dict): Each argument is a dictionary representing one design
                variable. The dictionary contains information about that
                design variable, including:
                 * 'name' (str, optional): The name of this design
                   if any are left blank, then ALL names are considered
                   unspecified.
                 * 'des_type' (str): The type for this design variable.
                   Currently supported options are:
                    * 'continuous' (or 'cont' or 'real')
                    * 'categorical' (or 'cat')
                    * 'integer' (or 'int')
                    * 'custom'
                    * 'raw' -- for advanced use only, not recommended
                 * 'lb' (float): When des_type is 'continuous' or 'integer',
                   this specifies the lower bound for the design variable.
                   This value must be specified, and must be strictly less
                   than 'ub' (below) up to the tolerance (below).
                 * 'ub' (float): When des_type is 'continuous' or 'integer',
                   this specifies the upper bound for the design variable.
                   This value must be specified, and must be strictly greater
                   than 'lb' (above) up to the tolerance (below).
                 * 'des_tol' (float): When des_type is 'continuous', this
                   specifies the tolerance, i.e., the minimum spacing along
                   this dimension, before two design values are considered to
                   have equal values in this dimension. If not specified, the
                   default value is 1.0e-8.
                 * 'levels' (int or list): When des_type is 'categorical', this
                   specifies the number of levels for the variable (when int)
                   or the names of each valid category (when a list).
                 * 'embedding_size' (int): When des_type is 'custom', this
                   specifies the dimension of the custom embedding.
                 * 'dtype' (str): When des_type is 'custom', this contains
                   a string specifying the numpy dtype of the custom input.
                   Only used when operating with named variables, otherwise
                   it must be numeric. When using named variables, defaults
                   to 'U25'.
                 * 'embedder': When des_type is 'custom', this is a custom
                   embedding function, which maps the input to a point in the
                   unit hypercube of dimension 'embedding_size'.
                 * 'extracter': When des_type is 'custom', this is a custom
                   extracting function, which maps a point in the unit
                   hypercube of dimension 'embedding_size' to a legal input
                   value of type 'dtype'.

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
            if 'des_type' in arg:
                if not isinstance(arg['des_type'], str):
                    raise TypeError("args['des_type'] must be a str")
            # Append a new continuous design variable (default) to the list
            if 'des_type' not in arg or \
               arg['des_type'] in ["continuous", "cont", "real"]:
                if 'des_tol' in arg:
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
                if 'lb' in arg and 'ub' in arg:
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
                if 'name' in arg:
                    if isinstance(arg['name'], str):
                        if any([arg['name'] == dname[0]
                                for dname in self.des_names]):
                            raise ValueError("arg['name'] must be unique")
                        self.des_names.append((arg['name'], 'f8'))
                    else:
                        raise TypeError("When present, 'name' must be a "
                                        + "str type")
                else:
                    self.use_names = False
                    name = "x" + str(self.n_cont + self.n_cat + self.n_int
                                     + self.n_custom + self.n_raw + 1)
                    self.des_names.append((name, 'f8', ))
                # Keep track of design variable indices for bookkeeping
                for i in range(len(self.des_order)):
                    # Add 1 to all integer variable indices
                    if self.des_order[i] >= self.n_cont:
                        self.des_order[i] += 1
                self.des_order.append(self.n_cont)
                self.n_cont += 1
                self.des_tols.append(des_tol)
                self.cont_lb.append(arg['lb'])
                self.cont_ub.append(arg['ub'])
            # Append a new categorical design variable to the list
            elif arg['des_type'] in ["categorical", "cat"]:
                if 'levels' in arg:
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
                        raise TypeError("args['levels'] must be a list or "
                                        + "int")
                else:
                    raise AttributeError("'levels' must be present when "
                                         + "'des_type' is 'categorical'")
                if 'name' in arg:
                    if not isinstance(arg['name'], str):
                        raise TypeError("When present, 'name' must be a "
                                        + "str type")
                else:
                    self.use_names = False
                    name = "x" + str(self.n_cont + self.n_cat + self.n_int
                                     + self.n_custom + self.n_raw + 1)
                    self.des_names.append((name, 'i4', ))
                # Keep track of design variable indices for bookkeeping
                for i in range(len(self.des_order)):
                    # Add 1 to all custom variable indices
                    if self.des_order[i] >= (self.n_cont + self.n_int +
                                             self.n_cat):
                        self.des_order[i] += 1
                self.des_order.append(self.n_cont + self.n_int
                                      + self.n_cat)
                self.n_cat += 1
                self.des_tols.append(0.5)
                if isinstance(arg['levels'], int):
                    self.n_lvls.append(arg['levels'])
                    self.cat_names.append([])
                    if 'name' in arg:
                        self.des_names.append((arg['name'], 'i4'))
                else:
                    self.n_lvls.append(len(arg['levels']))
                    self.cat_names.append(arg['levels'])
                    if 'name' in arg:
                        self.des_names.append((arg['name'], 'U25'))
                self.__generate_encoding__()
            # Add an integer design variable
            elif arg['des_type'] in ["integer", "int"]:
                # Relax to a continuous design variable with des_tol = 0.5
                des_tol = 0.5
                if 'lb' in arg and 'ub' in arg:
                    if not (isinstance(arg['lb'], int) and
                            isinstance(arg['ub'], int)):
                        raise TypeError("args['lb'] and args['ub'] must "
                                        + "be int types")
                    if arg['lb'] + des_tol >= arg['ub']:
                        raise ValueError("args['lb'] must be less than "
                                         + "or equal to args['ub'] up to "
                                         + "the design space tolerance")
                else:
                    raise AttributeError("'lb' and 'ub' keys must be "
                                         + "present when 'des_type' is "
                                         + "'integer'")
                if 'name' in arg:
                    if isinstance(arg['name'], str):
                        if any([arg['name'] == dname[0]
                                for dname in self.des_names]):
                            raise ValueError("arg['name'] must be unique")
                        self.des_names.append((arg['name'], 'f8'))
                    else:
                        raise TypeError("When present, 'name' must be a "
                                        + "str type")
                else:
                    self.use_names = False
                    name = "x" + str(self.n_cont + self.n_cat + self.n_int
                                     + self.n_custom + self.n_raw + 1)
                    self.des_names.append((name, 'i4', ))
                # Keep track of design variable indices for bookkeeping
                for i in range(len(self.des_order)):
                    # Add 1 to all categorical variable indices
                    if self.des_order[i] >= self.n_cont + self.n_int:
                        self.des_order[i] += 1
                self.des_order.append(self.n_cont + self.n_int)
                self.n_int += 1
                self.des_tols.append(des_tol)
                self.int_lb.append(arg['lb'])
                self.int_ub.append(arg['ub'])
            # Append a new custom design variable to the list
            elif arg['des_type'] in ["custom"]:
                if 'embedding_size' in arg:
                    if isinstance(arg['embedding_size'], int):
                        if arg['embedding_size'] < 1:
                            raise ValueError("args['embedding_size'] must"
                                             + " be at least 1")
                    else:
                        raise TypeError("args['embedding_size'] must be a "
                                        + "int")
                else:
                    raise AttributeError("'embedding_size' must be present"
                                         + " when 'des_type' is 'custom'")
                if 'dtype' in arg:
                    if not isinstance(arg['dtype'], str):
                        raise TypeError("When present, 'dtype' must be a " +
                                        "str type")
                    else:
                        # Make sure this is a legal numpy dtype
                        np.dtype(arg['dtype'])
                if 'name' in arg:
                    if not isinstance(arg['name'], str):
                        raise TypeError("When present, 'name' must be a "
                                        + "str type")
                    if 'dtype' in arg:
                        self.des_names.append((arg['name'], arg['dtype']))
                    else:
                        self.des_names.append((arg['name'], 'U25'))
                else:
                    self.use_names = False
                    name = "x" + str(self.n_cont + self.n_cat + self.n_int
                                     + self.n_custom + self.n_raw + 1)
                    self.des_names.append((name, 'f8', ))
                # Load the custom embedder/extracter
                if 'embedder' in arg:
                    if not callable(arg['embedder']):
                        raise TypeError("'embedder' must be a "
                                        + "callable object")
                else:
                    raise AttributeError("'embedder' must be present"
                                         + " when 'des_type' is 'custom'")
                if 'extracter' in arg:
                    if not callable(arg['extracter']):
                        raise TypeError("'extracter' must be a "
                                        + "callable object")
                else:
                    raise AttributeError("'extracter' must be present"
                                         + " when 'des_type' is 'custom'")
                self.custom_embedders.append(arg['embedder'])
                self.custom_extracters.append(arg['extracter'])
                # Keep track of design variable indices for bookkeeping
                for i in range(len(self.des_order)):
                    # Add 1 to all raw variable indices
                    if self.des_order[i] >= (self.n_cont + self.n_int +
                                             self.n_cat + self.n_custom):
                        self.des_order[i] += 1
                self.des_order.append(self.n_cont + self.n_int +
                                      self.n_cat + self.n_custom)
                self.n_custom += 1
                self.n_custom_d.append(arg["embedding_size"])
                self.des_tols.append(1.0e-8)
                for i in range(arg["embedding_size"]):
                    self.custom_des_tols.append(1.0e-8)
            # Append a new raw design variable to the list
            elif arg['des_type'] in ["raw"]:
                if 'name' in arg:
                    if not isinstance(arg['name'], str):
                        raise TypeError("When present, 'name' must be a "
                                        + "str type")
                    self.des_names.append((arg['name'], 'f8'))
                else:
                    self.use_names = False
                    name = "x" + str(self.n_cont + self.n_cat + self.n_int
                                     + self.n_custom + self.n_raw + 1)
                    self.des_names.append((name, 'f8', ))
                # Keep track of design variable indices for bookkeeping
                for i in range(len(self.des_order)):
                    # Add 1 to all later variable indices (should be none)
                    if self.des_order[i] >= (self.n_cont + self.n_int +
                                             self.n_cat + self.n_custom +
                                             self.n_raw):
                        self.des_order[i] += 1
                self.des_order.append(self.n_cont + self.n_int +
                                      self.n_cat + self.n_custom + self.n_raw)
                self.n_raw += 1
                self.des_tols.append(1.0e-8)
            else:
                raise(ValueError("des_type=" + arg['des_type'] +
                                 " is not a recognized value"))
        # Set the effective design dimension
        self.n = (self.n_cat_d + self.n_cont + self.n_int +
                  sum(self.n_custom_d) + self.n_raw)
        # Set the problem scaling
        self.scaled_lb = np.zeros(self.n)
        self.scaled_ub = np.ones(self.n)
        self.scale = np.ones(self.n)
        self.scaled_des_tols = np.zeros(self.n)
        n_total = 0
        # Calculate scaling for continuous variables
        for i in range(self.n_cont):
            self.scale[i] = self.cont_ub[i] - self.cont_lb[i]
            self.scaled_des_tols[i] = (self.des_tols[self.des_order.index(i)] /
                                       self.scale[i])
        n_total = n_total + self.n_cont
        # Calculate scaling for continuous variables
        for i in range(self.n_int):
            self.scale[n_total + i] = self.int_ub[i] - self.int_lb[i]
            self.scaled_des_tols[n_total + i] = \
                self.des_tols[self.des_order.index(i)] / self.scale[i]
        n_total = n_total + self.n_int
        # Calculate scaling for categorical variables
        self.scale[n_total:n_total+self.n_cat_d] = self.cat_scale[:]
        self.scaled_des_tols[n_total:n_total+self.n_cat_d] = \
            self.cat_des_tols[:]
        n_total += self.n_cat_d
        # Calculate scaling for custom variables
        self.scale[n_total:n_total+sum(self.n_custom_d)] = 1.0
        self.scaled_des_tols[n_total:n_total+sum(self.n_custom_d)] = \
            self.custom_des_tols[:]
        n_total += sum(self.n_custom_d)
        # Calculate scaling for raw variables
        self.scale[n_total:n_total+self.n_raw] = 1.0
        self.scaled_des_tols[n_total:n_total+self.n_raw] = 1.0e-8
        self.scaled_lb[n_total:n_total+self.n_raw] = -np.inf
        self.scaled_ub[n_total:n_total+self.n_raw] = np.inf
        n_total += self.n_raw
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
                 * name (str, optional): The name of this simulation
                   (defaults to "sim" + str(i), where i = 1, 2, 3, ... for
                   the first, second, third, ... simulation added to the
                   MOOP).
                 * m (int): The number of outputs for this simulation.
                 * sim_func (function): An implementation of the simulation
                   function, mapping from R^n -> R^m. The interface should
                   match: `sim_out = sim_func(x)`.
                 * search (GlobalSearch): A GlobalSearch object for performing
                   the initial search over this simulation's design space.
                 * surrogate (SurrogateFunction): A SurrogateFunction object
                   specifying how this simulation's outputs will be modeled.
                 * hyperparams (dict): A dictionary of hyperparameters, which
                   will be passed to the surrogate and search routines.
                   Most notably, the 'search_budget': (int) can be specified
                   here.

        """

        from parmoo.util import check_sims

        # Iterate through args to add each sim
        check_sims(self.n_cont + self.n_cat + self.n_int + self.n_custom +
                   self.n_raw, *args)
        for arg in args:
            # Use the number of sims
            m = arg['m']
            # Keep track of simulation names
            if 'name' in arg:
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
            if 'hyperparams' in arg:
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
            if 'sim_db' in arg:
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
        algebraic function of the design variables and simulation outputs.
        Note that all objectives must be specified before any acquisition
        functions can be added.

        Args:
            *args (dict): Python dictionary containing objective function
                information, including:
                 * 'name' (str, optional): The name of this objective
                   (defaults to "obj" + str(i), where i = 1, 2, 3, ... for the
                   first, second, third, ... simulation added to the MOOP).
                 * 'obj_func' (function): An algebraic objective function that
                   maps from R^n X R^m --> R. Interface should match:
                   `cost = obj_func(x, sim_func(x), der=0)`,
                   where `der` is an optional argument specifying whether to
                   take the derivative of the objective function
                    * 0 -- no derivative taken, return f(x, sim_func(x))
                    * 1 -- return derivative wrt x, or
                    * 2 -- return derivative wrt sim(x).

        """

        # Assert proper order of problem definition
        if len(self.acquisitions) > 0:
            raise RuntimeError("Cannot add more objectives after"
                               + " adding acquisition functions")
        # Check that arg and 'obj_func' field are legal types
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError("Each arg must be a Python dict")
            if 'obj_func' in arg:
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
            if 'name' in arg:
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

        Append a new constraint to the problem. The constraint can be
        a linear or nonlinear inequality constraint, and may depend on
        the design variables and/or the simulation outputs.

        Args:
            args (dict): Python dictionary containing constraint function
                information, including:
                 * 'name' (str, optional): The name of this constraint
                   (defaults to "const" + str(i), where i = 1, 2, 3, ... for
                   the first, second, third, ... constraint added to the MOOP).
                 * 'constraint' (function): An algebraic constraint function
                   that maps from R^n X R^m --> R and evaluates to zero or a
                   negative number when feasible and positive otherwise.
                   Interface should match:
                   `violation = constraint(x, sim_func(x), der=0)`,
                   where `der` is an optional argument specifying whether to
                   take the derivative of the constraint function
                    * 0 -- no derivative taken, return c(x, sim_func(x))
                    * 1 -- return derivative wrt x, or
                    * 2 -- return derivative wrt sim(x).
                   Note that any
                   ``constraint(x, sim_func(x), der=0) <= 0``
                   indicates that x is feaseible, while
                   ``constraint(x, sim_func(x), der=0) > 0``
                   indicates that x is infeasible, violating the constraint by
                   an amount proportional to the output.
                   It is the user's responsibility to ensure that after adding
                   all constraints, the feasible region is nonempty and has
                   nonzero measure in the design space.

        """

        # Check that arg and 'constraint' field are legal types
        for arg in args:
            if not isinstance(arg, dict):
                raise TypeError("Each arg must be a Python dict")
            if 'constraint' in arg:
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
            if 'name' in arg:
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
        each acquisition is used to generate one or more points to evaluate
        Typically, each acquisition generates one evaluation per simulation
        function.

        Args:
            args (dict): Python dictionary of acquisition function info,
                including:
                 * 'acquisition' (AcquisitionFunction): An acquisition function
                   that maps from R^o --> R for scalarizing outputs.
                 * 'hyperparams' (dict): A dictionary of hyperparameters for
                   the acquisition functions. Can be omitted if no
                   hyperparameters are needed.

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
            if 'acquisition' not in arg:
                raise AttributeError("'acquisition' field must be present in "
                                     + "every arg")
            if 'hyperparams' in arg:
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
            except BaseException:
                raise TypeError("'acquisition' must specify a child of the"
                                + " AcquisitionFunction class")
            if not isinstance(acquisition, structs.AcquisitionFunction):
                raise TypeError("'acquisition' must specify a child of the"
                                + " AcquisitionFunction class")
            # If all checks passed, add the acquisition to the list
            self.acquisitions.append(acquisition)
        return

    def setCheckpoint(self, checkpoint,
                      checkpoint_data=True, filename="parmoo"):
        """ Set ParMOO's checkpointing feature.

        Note that for checkpointing to work, all simulation, objective,
        and constraint functions must be defined in the global scope.
        ParMOO also cannot save lambda functions.

        Args:
            checkpoint (bool): Turn checkpointing on (True) or off (False).

            checkpoint_data (bool, optional): Also save raw simulation output
                in a separate JSON file (True) or rely on ParMOO's internal
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
        """ Get the numpy dtype of all design points for this MOOP.

        Use this type when allocating a numpy array to store the design
        points for this MOOP object.

        Returns:
            np.dtype: The numpy dtype of this MOOP's design points.
            If no design variables have yet been added, returns None.

        """

        if self.n_cont + self.n_cat + self.n_int + self.n_custom + \
           self.n_raw < 1:
            return None
        elif self.use_names:
            return np.dtype(self.des_names)
        else:
            return np.dtype(('f8', (self.n_cont + self.n_cat + self.n_int +
                                    self.n_custom + self.n_raw,)))

    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP.

        Use this type if allocating a numpy array to store the simulation
        outputs of this MOOP object.

        Returns:
            np.dtype: The numpy dtype of this MOOP's simulation outputs.
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
            np.dtype: The numpy dtype of this MOOP's objective points.
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
            np.dtype: The numpy dtype of this MOOP's constraint violation
            outputs. If no constraint functions have been given, returns None.

        """

        if self.p < 1:
            return None
        elif self.use_names:
            return np.dtype(self.const_names)
        else:
            return np.dtype(('f8', (self.p,)))

    def check_sim_db(self, x, s_name):
        """ Check self.sim_db[s_name] to see if the design x was evaluated.

        x (np.ndarray or numpy structured array): A 1d numpy.ndarray or numpy
            structured array specifying the design point to check for.

        s_name (str or int): The name or index of the simulation where
            (x, sx) will be added. Note, indices are assigned in the order
            the simulations were listed during initialization.

        Returns:
            None or numpy.ndarray: returns None if x is not in
            self.sim_db[s_name] (up to the design tolerance). Otherwise,
            returns the corresponding value of sx.

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
        """ Update sim_db[s_name] by adding a design/simulation output pair.

        x (np.ndarray or numpy structured array): A 1d numpy.ndarray or numpy
            structured array specifying the design point to add.


        sx (np.ndarray): A 1d numpy.ndarray containing the corresponding
            simulation output.

        s_name (str or int): The name or index of the simulation to whose
            database the pair (x, sx) will be added. Note, when using unnamed
            variables and simulations, the simulation indices were assigned
            in the same order that the simulations were added to the MOOP
            (using `MOOP.addSimulation(*args)`) during initialization.

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
        """ Evaluate sim_func[s_name] and store the result in the database.

        Args:
            x (numpy.ndarray or numpy structured array): Either a numpy
                structured array (when using named variables) or a 1D
                numpy.ndarray containing the values of the design variable
                to evaluate. Note, when operating with unnamed variables,
                design variables are indices were assigned in the order that
                the design variables were added to the MOOP using
                `MOOP.addDesign(*args)`.

            s_name (str, int): The name or index of the simulation to
                evaluate. Note, when operating with unnamed variables,
                simulation indices were assigned in the order that
                the simulations were added to the MOOP using
                `MOOP.addSimulation(*args)`.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the output from the
            sx = simulation[s_name](x).

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
        """ Fit the surrogate models using the current sim databases.

        Warning: Not recommended for external usage!

        """

        # Call self.surrogates.fit() to fit the surrogate models
        for i in range(self.s):
            n_new = self.sim_db[i]['n']
            self.surrogates[i].fit(self.sim_db[i]['x_vals'][:n_new, :],
                                   self.sim_db[i]['s_vals'][:n_new, :])
            self.sim_db[i]['old'] = self.sim_db[i]['n']
        return

    def updateSurrogates(self):
        """ Update the surrogate models using the current sim databases.

        Warning: Not recommended for external usage!

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

        Warning: Not recommended for external usage!

        Args:
            center (numpy.ndarray): A 1d numpy.ndarray containing the
                (embedded) coordinates of the new center in the rescaled
                design space.

        Returns:
            float: The minimum over the recommended trust region radius
            for all surrogates.

        """

        rad = max(self.scaled_ub - self.scaled_lb)
        for si in self.surrogates:
            try:
                rad = min(si.setCenter(center), rad)
            except NotImplementedError:
                rad = max(self.scaled_ub - self.scaled_lb)
        return rad

    def evaluateSurrogates(self, x):
        """ Evaluate all objectives using the simulation surrogates as needed.

        Warning: Not recommended for external usage!

        Args:
            x (numpy.ndarray): A 1d numpy.ndarray containing the (embedded)
                design point to evaluate.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the result of the
            evaluation.

        """

        # Check for illegal input
        if isinstance(x, np.ndarray):
            if x.shape[0] != self.n:
                raise ValueError("x must have length n")
        else:
            raise TypeError("x must be a numpy array")
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

        Warning: Not recommended for external usage!

        Args:
            x (numpy.ndarray): A 1d numpy.ndarray containing the (embedded)
                design point to evaluate.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the list of constraint
            violations at x (zero if no violation).

        """

        # Check for illegal input
        if isinstance(x, np.ndarray):
            if x.shape[0] != self.n:
                raise ValueError("x must have length n")
        else:
            raise TypeError("x must be a numpy array")
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

    def evaluatePenalty(self, x):
        """ Evaluate the penalized objective using the surrogates as needed.

        Warning: Not recommended for external usage!

        Args:
            x (numpy.ndarray): A 1d numpy.ndarray containing the (embedded)
                design point to evaluate.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the result of the
            evaluation.

        """

        # Check for illegal input
        if isinstance(x, np.ndarray):
            if x.shape[0] != self.n:
                raise ValueError("x must have length n")
        else:
            raise TypeError("x must be a numpy array")
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
        # Compute the penalized objective score
        Lx[:] = self.lam * Lx[:] + fx[:]
        # Return the result
        return Lx

    def evaluateGradients(self, x):
        """ Evaluate the gradient of the penalized objective using surrogates.

        Warning: Not recommended for external usage!

        Args:
            x (numpy.ndarray): A 1d numpy.ndarray containing the (embedded)
                design point to evaluate.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the result of the
            evaluation.

        """

        # Check for illegal input
        if isinstance(x, np.ndarray):
            if x.shape[0] != self.n:
                raise ValueError("x must have length n")
        else:
            raise TypeError("x must be a numpy array")
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
        # Construct the Jacobian of the penalized objective function
        dLx = np.zeros((self.o, self.n))
        for i in range(self.o):
            dLx[i, :] = dfx[i, :] + self.lam * dcx[:]
        # Return the result
        return dLx

    def addData(self, x, sx):
        """ Update the internal objective database by truly evaluating x.

        Args:
            x (numpy.ndarray or numpy structured array): Either a numpy
                structured array (when using named variables) or a 1D
                numpy.ndarray containing the value of the design variable
                to add to ParMOO's database. When operating with unnamed
                variables, the indices were assigned in the order that
                the design variables were added to the MOOP using
                `MOOP.addDesign(*args)`.

            sx (numpy.ndarray or numpy structured array): Either a numpy
                structured array (when using named variables) or a 1D
                numpy.ndarray containing the values of the corresponding
                simulation outputs for ALL simulations involved in this
                MOOP. In named mode, sx['s_name'][:] contains the output(s)
                for sim_func['s_name']. In unnamed mode, simulation indices
                were assigned in the order that they were added using
                `MOOP.addSimulation(*args)`. Then, if each simulation i has
                m_i outputs (i = 0, 1, ...):
                 * sx[:m_0] contains the output(s) of sim_func[0],
                 * sx[m_0:m_0 + m_1] contains output(s) of sim_func[1],
                 * sx[m_0 + m_1:m_0 + m_1 + m_2] contains the output(s) for
                   sim_func[2], etc.

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
        """ Perform an iteration of ParMOO's solver and generate candidates.

        Generates a batch of suggested candidate points
        (design point, simulation name) pairs, for the caller to evaluate
        (externally if needed).

        Args:
            k (int): The iteration counter (corresponding to MOOP.iteration).

        Returns:
            (list): A list of ordered pairs (tuples), specifying the
            (design points, simulation name) that ParMOO suggests for
            evaluation. Specifically:
             * The first entry in each tuple is either a numpy structured
               array (when operating with named variables) or a 1D
               numpy.ndarray (in unnamed mode). When operating with unnamed
               variables, the indices were assigned in the order that
               the design variables were added to the MOOP using
               `MOOP.addDesign(*args)`.
             * The second entry is either the (str) name of the simulation to
               evaluate (when operating with named variables) or the (int)
               index of the simulation to evaluate (when operating in
               unnamed mode). Note, in unnamed mode, simulation indices
               were assigned in the order that they were added using
               `MOOP.addSimulation(*args)`.

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
                                                 self.evaluatePenalty,
                                                 self.history)
            # Set up the surrogate problem
            opt = self.optimizer(self.o, self.scaled_lb, self.scaled_ub,
                                 self.hyperparams)
            opt.setObjective(self.evaluateSurrogates)
            opt.setPenalty(self.evaluatePenalty, self.evaluateGradients)
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
            k (int): The iteration counter (corresponding to MOOP.iteration).

            batch (list): A list of ordered pairs (tuples), each specifying
                a design point that was evaluated in this iteration.
                For each tuple in the list:
                 * The first entry in each tuple is either a numpy structured
                   array (when operating with named variables) or a 1D
                   numpy.ndarray (in unnamed mode). When operating with
                   unnamed variables, the indices were assigned in the order
                   that the design variables were added to the MOOP using
                   `MOOP.addDesign(*args)`.
                 * The second entry is either the (str) name of the simulation
                   to evaluate (when operating with named variables) or the
                   (int) index of the simulation to evaluate (when operating
                   in unnamed mode). Note, in unnamed mode, simulation indices
                   were assigned in the order that they were added using
                   `MOOP.addSimulation(*args)`.

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

        If desired, be sure to turn on checkpointing before starting the
        solve, using:

        ``MOOP.setCheckpoint(checkpoint, [checkpoint_data, filename])``

        and turn on INFO-level logging for verbose output, using:

        ``
        import logging
        logging.basicConfig(level=logging.INFO,
            [format='%(asctime)s %(levelname)-8s %(message)s',
             datefmt='%Y-%m-%d %H:%M:%S'])
        ``

        Args:
            budget (int): The max budget for ParMOO's internal iteration
                counter. ParMOO keeps track of how many iterations it has
                completed internally. This value k specifies the stopping
                criteria for ParMOO.

        """

        import logging

        # Check that the budget is a legal integer
        if isinstance(budget, int):
            if budget < 0:
                raise ValueError("budget must be nonnegative")
        else:
            raise TypeError("budget must be an int type")

        # Print logging info summary of problem setup
        logging.info(" Beginning new run of ParMOO...")
        logging.info(" summary of settings:")
        logging.info(f"   {self.n} design dimensions")
        logging.info(f"     continuous design variables: {self.n_cont}")
        logging.info(f"     categorical design variables: {self.n_cat}")
        logging.info(f"     integer design variables: {self.n_int}")
        logging.info(f"     custom design variables: {self.n_custom}")
        logging.info(f"     raw design variables: {self.n_raw}")
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

    def getPF(self, format='ndarray'):
        """ Extract nondominated and efficient sets from internal databases.

        Returns:
            A discrete approximation of the Pareto front and efficient set.

            If operating with named variables, then this is a 1d numpy
            structured array whose fields match the names for design
            variables, objectives, and constraints (if any).

            Otherwise, this is a dict containing the following keys:
             * x_vals (numpy.ndarray): A 2d numpy.ndarray containing a list
               of nondominated points discretely approximating the
               Pareto front.
             * f_vals (numpy.ndarray): A 2d numpy.ndarray containing the list
               of corresponding efficient design points.
             * c_vals (numpy.ndarray): A 2d numpy.ndarray containing the list
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
                if x_vals is not None and x_vals.shape[0] > 0:
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
        if format == 'pandas':
            return pd.DataFrame(result)
        elif format == 'ndarray':
            return result
        else:
            raise ValueError(str(format) + " is an invalid value for 'format'")

    def getSimulationData(self, format='ndarray'):
        """ Extract all computed simulation outputs from the MOOP's database.

        Returns:
            (dict or list) Either a dictionary or list of dictionaries
            containing every point where a simulation was evaluated.

            If operating with named variables, then the result is a dict.
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
            if format == 'pandas':
                return pd.DataFrame(result)
            elif format == 'ndarray':
                return result
            else:
                raise ValueError(str(format) + "is an invalid value for "
                                 + "'format'")
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
            if format == 'pandas':
                return pd.DataFrame(result)
            elif format == 'ndarray':
                return result
            else:
                raise ValueError(str(format) + " is an invalid value for "
                                 + "'format'")

    def getObjectiveData(self, format='ndarray'):
        """ Extract all computed objective scores from this MOOP's database.

        Returns:
            A database of all designs that have been fully evaluated,
            and their corresponding objective scores.

            If operating with named variables, then this is a 1d numpy
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
        if format == 'pandas':
            return pd.DataFrame(result)
        elif format == 'ndarray':
            return result
        else:
            raise ValueError(str(format) + " is an invalid value for 'format'")

    def save(self, filename="parmoo"):
        """ Serialize and save the MOOP object and all of its dependencies.

        Args:
            filename (str, optional): The filepath to serialized
                checkpointing file(s). Do not include file extensions,
                they will be appended automaically. This method may create
                several additional save files with this same name, but
                different file extensions, in order to recursively save
                dependency objects (such as surrogate models). Defaults to
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
                        'cont_lb': self.cont_lb,
                        'cont_ub': self.cont_ub,
                        'int_lb': self.int_lb,
                        'int_ub': self.int_ub,
                        'n_cat_d': self.n_cat_d,
                        'n_custom_d': self.n_custom_d,
                        'n_cat': self.n_cat,
                        'n_cont': self.n_cont,
                        'n_int': self.n_int,
                        'n_custom': self.n_custom,
                        'n_raw': self.n_raw,
                        'n_lvls': self.n_lvls,
                        'des_order': self.des_order,
                        'cat_names': self.cat_names,
                        'sim_names': self.sim_names,
                        'des_names': self.des_names,
                        'obj_names': self.obj_names,
                        'const_names': self.const_names,
                        'lam': self.lam,
                        'des_tols': self.des_tols,
                        'epsilon': self.epsilon,
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
        if isinstance(self.custom_des_tols, np.ndarray):
            parmoo_state['custom_des_tols'] = self.custom_des_tols.tolist()
        else:
            parmoo_state['custom_des_tols'] = self.custom_des_tols
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
        if 'x_vals' in self.data:
            parmoo_state['data']['x_vals'] = self.data['x_vals'].tolist()
        if 'f_vals' in self.data:
            parmoo_state['data']['f_vals'] = self.data['f_vals'].tolist()
        if 'c_vals' in self.data:
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
        # Store names/modules of custom embedders
        parmoo_state['custom_embedders'] = [(ei.__name__, ei.__module__)
                                            for ei in self.custom_embedders]
        # Store names/modules of custom extracters
        parmoo_state['custom_extracters'] = [(ei.__name__, ei.__module__)
                                             for ei in self.custom_extracters]
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
            filename (str, optional): The filepath to the serialized
                checkpointing file(s). Do not include file extensions,
                they will be appended automaically. This method may also
                load from other save files with the same name, but different
                file extensions, in order to recursively load dependency
                objects (such as surrogate models) as needed.
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
        self.cont_lb = parmoo_state['cont_lb']
        self.cont_ub = parmoo_state['cont_ub']
        self.int_lb = parmoo_state['int_lb']
        self.int_ub = parmoo_state['int_ub']
        self.n_cat_d = parmoo_state['n_cat_d']
        self.n_custom_d = parmoo_state['n_custom_d']
        self.n_cat = parmoo_state['n_cat']
        self.n_cont = parmoo_state['n_cont']
        self.n_int = parmoo_state['n_int']
        self.n_custom = parmoo_state['n_custom']
        self.n_raw = parmoo_state['n_raw']
        self.n_lvls = parmoo_state['n_lvls']
        self.des_order = parmoo_state['des_order']
        self.cat_names = parmoo_state['cat_names']
        self.sim_names = [tuple(item) for item in parmoo_state['sim_names']]
        self.des_names = [tuple(item) for item in parmoo_state['des_names']]
        self.obj_names = [tuple(item) for item in parmoo_state['obj_names']]
        self.const_names = [tuple(item)
                            for item in parmoo_state['const_names']]
        self.lam = parmoo_state['lam']
        self.des_tols = parmoo_state['des_tols']
        self.epsilon = parmoo_state['epsilon']
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
        self.custom_des_tols = np.array(parmoo_state['custom_des_tols'])
        self.cat_lb = np.array(parmoo_state['cat_lb'])
        self.cat_scale = np.array(parmoo_state['cat_scale'])
        self.RSVT = np.array(parmoo_state['RSVT'])
        self.mean = np.array(parmoo_state['mean'])
        # Reload serialized internal databases
        self.data = {}
        if 'x_vals' in parmoo_state['data']:
            self.data['x_vals'] = np.array(parmoo_state['data']['x_vals'])
        if 'f_vals' in parmoo_state['data']:
            self.data['f_vals'] = np.array(parmoo_state['data']['f_vals'])
        if 'c_vals' in parmoo_state['data']:
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
        # Recover custom embedders
        self.custom_embedders = []
        for i, (e_name, e_mod) in enumerate(parmoo_state['custom_embedders']):
            mod = import_module(e_mod)
            new_em = getattr(mod, e_name)
            self.custom_embedders.append(new_em)
        # Recover custom extracters
        self.custom_extracters = []
        for i, (e_name, e_mod) in enumerate(parmoo_state['custom_extracters']):
            mod = import_module(e_mod)
            new_em = getattr(mod, e_name)
            self.custom_extracters.append(new_em)
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
            filename (str, optional): The filepath to the checkpointing
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
            for key in x.dtype.names:
                toadd[key] = x[key]
            if isinstance(sx, np.ndarray):
                toadd['out'] = sx.tolist()
            else:
                toadd['out'] = sx
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
