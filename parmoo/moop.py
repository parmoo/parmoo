
""" Contains the MOOP class for defining multiobjective optimization problems.

parmoo.moop.MOOP is the base class for defining and solving multiobjective
optimization problems (MOOPs). Each MOOP object may contain several
simulations, specified using dictionaries.

"""

import codecs
from importlib import import_module
import inspect
import jax
from jax import numpy as jnp
import json
import logging
import numpy as np
from os.path import exists as file_exists
import pandas as pd
from parmoo import structs
from parmoo.embeddings.default_embedders import *
from parmoo.util import check_names, check_sims, updatePF
import pickle
import shutil
import warnings


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

    To turn on checkpointing use:
     * ``MOOP.setCheckpoint(checkpoint, [checkpoint_data, filename])``

    ParMOO also offers logging. To turn on logging, activate INFO-level
    logging by importing Python's built-in logging module.

    After defining the MOOP and setting up checkpointing and logging info,
    use the following method to solve the MOOP (serially):
     * ``MOOP.solve(iter_max=None, sim_max=None)``

    The following methods are used for solving the MOOP and managing the
    internal simulation/objective databases:
     * ``MOOP.checkSimDb(x, s_name)``
     * ``MOOP.updateSimDb(x, sx, s_name)``
     * ``MOOP.evaluateSimulation(x, s_name)``
     * ``MOOP.addObjData(x, sx)``
     * ``MOOP.iterate(k, ib=None)``
     * ``MOOP.updateAll(k, batch)``

    Finally, the following methods are used to retrieve data after the
    problem has been solved:
     * ``MOOP.getPF(format='ndarray')``
     * ``MOOP.getSimulationData(format='ndarray')``
     * ``MOOP.getObjectiveData(format='ndarray')``

    The following methods are used to save/load the current checkpoint (state):
     * ``MOOP.save([filename="parmoo"])``
     * ``MOOP.load([filename="parmoo"])``

    The following private methods are not recommended for external usage:
     * ``MOOP._embed(x)``
     * ``MOOP._extract(x)``
     * ``MOOP._pack_sim(sx)``
     * ``MOOP._unpack_sim(sx)``
     * ``MOOP._fit_surrogates()``
     * ``MOOP._update_surrogates()``
     * ``MOOP._set_surrogate_tr(center, radius)``
     * ``MOOP._evaluate_surrogates(x)``
     * ``MOOP._surrogate_uncertainty(x)``
     * ``MOOP._evaluate_objectives(x, sx)``
     * ``MOOP._evaluate_constraints(x, sx)``
     * ``MOOP._evaluate_penalty(x, sx)``
     * ``MOOP._vobj_funcs(x, sx)``
     * ``MOOP._vcon_funcs(x, sx)``
     * ``MOOP._vpen_funcs(x, sx, cx)``

    """

    __slots__ = [
                 # Problem dimensions
                 'm', 'm_list', 'n_embed', 'n_feature', 'n_latent',
                 'o', 'p', 's',
                 # Tolerances and bounds
                 'feature_des_tols', 'latent_des_tols',
                 'latent_lb', 'latent_ub',
                 # Schemas
                 'des_schema', 'sim_schema', 'obj_schema', 'con_schema',
                 # Constants, counters, and adaptive parameters
                 'compiled', 'empty', 'epsilon', 'iteration', 'lam',
                 # Checkpointing markers
                 'checkpoint', 'checkpoint_data', 'checkpoint_file',
                 'new_checkpoint', 'new_data',
                 # Design variables, simulations, objectives, and constraints
                 'embedders', 'emb_hp', 'sim_funcs',
                 'obj_funcs', 'obj_grads', 'con_funcs', 'con_grads',
                 # Solver components
                 'acquisitions', 'searches', 'surrogates', 'optimizer',
                 # Database information
                 'data', 'sim_db', 'n_dat',
                 # Temporary solver components and metadata used during setup
                 'acq_tmp', 'opt_tmp', 'search_tmp', 'sur_tmp',
                 'acq_hp', 'opt_hp', 'sim_hp',
                 # Random generator object with state information
                 'np_random_gen',
                 # Compiled function definitions -- These are only defined
                 # after calling the MOOP.compile() method
                 'embed', 'extract', 'pack_sim', 'unpack_sim',
                 'evaluate_objectives', 'evaluate_constraints',
                 'evaluate_penalty',
                 'evaluate_surrogates', 'surrogate_uncertainty',
                 'vobj_funcs', 'vcon_funcs', 'vpen_funcs',
                 'obj_bwd', 'con_bwd', 'pen_bwd'
                ]

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

        # Initialize the problem dimensions
        self.m = 0
        self.m_list, self.n_embed = [], []
        self.n_feature, self.n_latent = 0, 0
        self.o, self.p, self.s = 0, 0, 0
        # Initialize the bounds and tolerances
        self.feature_des_tols, self.latent_des_tols = [], []
        self.latent_lb, self.latent_ub = [], []
        # Initialize the schemas
        self.des_schema, self.sim_schema = [], []
        self.obj_schema, self.con_schema = [], []
        # Initialize the constants, counters, and adaptive parameters
        self.compiled = False
        self.empty = jnp.zeros(0)
        self.epsilon = jnp.sqrt(jnp.finfo(jnp.ones(1)).eps)
        self.iteration = 0
        self.lam = 1.0
        # Initialize checkpointing markers
        self.checkpoint, self.checkpoint_data = False, False
        self.checkpoint_file = "parmoo"
        self.new_checkpoint, self.new_data = True, True
        # Initialize design variable embeddings
        self.embedders, self.emb_hp = [], []
        # Initialize simulations, objectives, constraints, and their metadata
        self.sim_funcs = []
        self.obj_funcs, self.obj_grads = [], []
        self.con_funcs, self.con_grads = [], []
        # Initialize solver components and their metadata
        self.acquisitions, self.searches, self.surrogates = [], [], []
        self.acq_tmp, self.search_tmp, self.sur_tmp = [], [], []
        self.acq_hp, self.sim_hp = [], []
        self.optimizer, self.opt_tmp = None, None
        self.opt_hp = {}
        # Initialize the database
        self.data, self.sim_db = {}, []
        self.n_dat = 0
        # Set up the surrogate optimizer and its hyperparameters
        if hyperparams is not None:
            if isinstance(hyperparams, dict):
                self.opt_hp = hyperparams
            else:
                raise TypeError("hyperparams must be a Python dict")
        if "np_random_gen" in self.opt_hp:
            if isinstance(self.opt_hp["np_random_gen"], np.random.Generator):
                self.np_random_gen = self.opt_hp["np_random_gen"]
            else:
                self.np_random_gen = np.random.default_rng(
                                            seed=self.opt_hp["np_random_gen"])
        else:
            self.np_random_gen = np.random.default_rng()
        self.opt_hp["np_random_gen"] = self.np_random_gen
        try:
            self.optimizer = opt_func(1, np.zeros(1), np.ones(1), self.opt_hp)
        except BaseException:
            raise TypeError("opt_func must be a derivative of the "
                            "SurrogateOptimizer abstract class")
        if not isinstance(self.optimizer, structs.SurrogateOptimizer):
            raise TypeError("opt_func must be a derivative of the "
                            "SurrogateOptimizer abstract class")
        self.opt_tmp = opt_func
        return

    def addDesign(self, *args):
        """ Add a new design variables to the MOOP.

        Args:
            args (dict): Each argument is a dictionary representing one design
                variable. The dictionary contains information about that
                design variable, including:
                 * 'name' (str, optional): The unique name of this design
                   variable, which ultimately serves as its primary key in
                   all of ParMOO's databases. This is also how users should
                   index this variable in all user-defined functions passed
                   to ParMOO.
                   If left blank, it defaults to "xi" where i= 1, 2, 3,...
                   corresponds to the order in which the design variables
                   were added.
                 * 'des_type' (str): The type for this design variable.
                   Currently supported options are:
                    * 'continuous' (or 'cont' or 'real')
                    * 'categorical' (or 'cat')
                    * 'integer' (or 'int')
                    * 'custom' -- an Embedder class must be provided (below)
                    * 'raw' -- no re-scaling is performed: *NOT RECOMMENDED*
                 * 'lb' (float): When des_type is 'continuous', 'integer', or
                   'raw' this specifies the lower bound for the range of
                   values this design variable could take.
                   This value must be specified, and must be strictly less
                   than 'ub' (below) up to the tolerance (below).
                 * 'ub' (float): When des_type is 'continuous', 'integer', or
                   'raw' this specifies the upper bound for the range of
                   values this design variable could take.
                   This value must be specified, and must be strictly greater
                   than 'lb' (above) up to the tolerance (below) or by a whole
                   numer for integer variables.
                 * 'des_tol' (float): When des_type is 'continuous', this
                   specifies the tolerance, i.e., the minimum spacing along
                   this dimension, before two design values are considered to
                   have equal values in this dimension. If not specified, the
                   default value is epsilon * max(ub - lb, 1.0e-4).
                 * 'levels' (int or list): When des_type is 'categorical', this
                   specifies the number of levels for the variable (when int)
                   or the names of each valid category (when a list).
                   *WARNING*: If a list is given and the entries in the list do
                   not have numeric types, then ParMOO will not be able to jit
                   the extractor which will lead to seriously degraded
                   performance.
                 * 'embedder' (parmoo.structs.Embedder): When des_type is
                   'custom', this is a custom Embedder class, which maps the
                   input to a point in the unit hypercube and reports the
                   embedded dimension.

        """

        for arg in args:
            # Check arg and optional inputs for correct types
            if not isinstance(arg, dict):
                raise TypeError("Each argument must be a Python dict")
            if 'des_type' in arg:
                if not isinstance(arg['des_type'], str):
                    raise TypeError("args['des_type'] must be a str")
            if 'name' in arg:
                name = arg['name']
            else:
                name = f"x{len(self.des_schema) + 1}"
            check_names(name, self.des_schema, self.sim_schema,
                        self.obj_schema, self.con_schema)
            # Append each design variable (default) to the schema
            if 'des_type' not in arg or \
               arg['des_type'] in ["continuous", "cont", "real"]:
                arg1 = arg
                embedder = ContinuousEmbedder(arg1)
            elif arg['des_type'] in ["integer", "int"]:
                arg1 = arg
                embedder = IntegerEmbedder(arg1)
            elif arg['des_type'] in ["categorical", "cat"]:
                arg1 = arg
                embedder = CategoricalEmbedder(arg1)
            elif arg['des_type'] in ["custom"]:
                if 'embedder' not in arg:
                    raise AttributeError("For a custom embedder, the "
                                         "'embedder' key must be present.")
                arg1 = {}
                for key in arg:
                    if key != 'embedder':
                        arg1[key] = arg[key]
                arg1['np_random_gen'] = self.np_random_gen
                try:
                    embedder = arg['embedder'](arg1)
                except BaseException:
                    raise TypeError("When present, the 'embedder' key must "
                                    "contain an Embedder class.")
                if not isinstance(embedder, structs.Embedder):
                    raise TypeError("When present, the 'embedder' key must "
                                    "contain an Embedder class.")
            elif arg['des_type'] in ["raw"]:
                arg1 = arg
                embedder = IdentityEmbedder(arg1)
            else:
                raise ValueError("des_type=" + arg['des_type'] +
                                 " is not a recognized value")
            # Collect the metadata for this embedding
            self.n_feature += 1
            self.n_embed.append(embedder.getEmbeddingSize())
            self.n_latent += self.n_embed[-1]
            # Update the des tols and latent bound constraints
            lbs = embedder.getLowerBounds()
            for lb in lbs:
                self.latent_lb.append(lb)
            ubs = embedder.getUpperBounds()
            for ub in ubs:
                self.latent_ub.append(ub)
            self.feature_des_tols.append(embedder.getFeatureDesTols())
            des_tols = embedder.getLatentDesTols()
            for des_tol in des_tols:
                self.latent_des_tols.append(float(des_tol))
            # Update the schema and add the embedder to list
            dtype = embedder.getInputType()
            self.des_schema.append((name, dtype))
            self.embedders.append(embedder)
            self.emb_hp.append(arg1)  # This is saved for re-loading
        return

    def addSimulation(self, *args):
        """ Add new simulations to the MOOP.

        Append new simulation functions to the problem.

        Args:
            args (dict): Each argument is a dictionary representing one
                simulation function. The dictionary must contain information
                about that simulation function, including:
                 * name (str, optional): The name of this simulation
                   (defaults to ``sim{i}``, where i = 1, 2, 3, ... for
                   the first, second, third, ... simulation added to the
                   MOOP).
                 * m (int): The number of outputs for this simulation.
                 * sim_func (function): An implementation of the simulation
                   function, mapping from X -> R^m (where X is the design
                   space). The interface should match:
                   ``sim_out = sim_func(x)``.
                 * search (GlobalSearch): A GlobalSearch object for performing
                   the initial search over this simulation's design space.
                 * surrogate (SurrogateFunction): A SurrogateFunction object
                   specifying how this simulation's outputs will be modeled.
                 * hyperparams (dict): A dictionary of hyperparameters, which
                   will be passed to the surrogate and search routines.
                   Most notably, the 'search_budget': (int) can be specified
                   here.

        """

        # Check that the simulation input is a legal format
        check_sims(self.n_feature, *args)
        for arg in args:
            m = arg['m']
            if 'name' in arg:
                name = arg['name']
            else:
                name = "sim" + str(self.s + 1)
            check_names(name, self.des_schema, self.sim_schema,
                        self.obj_schema, self.con_schema)
            # Update the schema and track the simulation output dimensions
            if m > 1:
                self.sim_schema.append((name, 'f8', m))
            else:
                self.sim_schema.append((name, 'f8'))
            self.m_list.append(m)
            self.m += m
            self.s += 1
            # Initialize the hyperparameter dictionary
            if 'hyperparams' in arg:
                hps = arg['hyperparams']
            else:
                hps = {}
            hps["np_random_gen"] = self.np_random_gen
            # Add the simulation's search and surrogate techniques
            self.search_tmp.append(arg['search'])
            self.sur_tmp.append(arg['surrogate'])
            self.sim_hp.append(hps)
            # Add the simulation function
            self.sim_funcs.append(arg['sim_func'])
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
                   maps from X, S --> R, where X is the design space and S is
                   the space of simulation outputs. Interface should match:
                   ``cost = obj_func(x, sx)`` where the value ``sx`` is
                   given by
                   ``sx = sim_func(x)`` at runtime.
                 * 'obj_grad' (function): Evaluates the gradients of
                   ``obj_func`` wrt s and sx. Interface should match:
                   ``dx, ds = obj_grad(x, sx)`` where the value ``sx`` is
                   given by ``sx = sim_func(x)`` at runtime.
                   The outputs ``dx`` and ``ds`` represent the gradients with
                   respect to ``x`` and ``sx``, respectively.

        """

        for arg in args:
            # Check that the objective dictionary is a legal format
            if not isinstance(arg, dict):
                raise TypeError("Each arg must be a Python dict")
            if 'obj_func' in arg:
                if not callable(arg['obj_func']):
                    raise TypeError("The 'obj_func' must be callable")
                if len(inspect.signature(arg['obj_func']).parameters) != 2:
                    raise ValueError("The 'obj_func' must take 2 args")
            else:
                raise AttributeError("Each arg must contain an 'obj_func'")
            if 'obj_grad' in arg:
                if not callable(arg['obj_grad']):
                    raise TypeError("The 'obj_grad' must be callable")
                if len(inspect.signature(arg['obj_grad']).parameters) != 2:
                    raise ValueError("If present, 'obj_grad' must take 2 args")
            # Check the objective name
            if 'name' in arg:
                name = arg['name']
            else:
                name = f"f{self.o + 1}"
            check_names(name, self.des_schema, self.sim_schema,
                        self.obj_schema, self.con_schema)
            # Finally, if all else passed, add the objective
            self.obj_schema.append((name, 'f8'))
            self.obj_funcs.append(arg['obj_func'])
            if 'obj_grad' in arg:
                self.obj_grads.append(arg['obj_grad'])
            self.o += 1
        return

    def addConstraint(self, *args):
        """ Add a new constraint to the MOOP.

        Args:
            args (dict): Python dictionary containing constraint function
                information, including:
                 * 'name' (str, optional): The name of this constraint
                   (defaults to "const" + str(i), where i = 1, 2, 3, ... for
                   the first, second, third, ... constraint added to the MOOP).
                 * 'con_func' or 'constraint' (function): An algebraic
                   constraint function that maps from X, S --> R where X and
                   S are the design space and space of aggregated simulation
                   outputs, respectively. The constraint function should
                   evaluate to zero or a negative number when feasible and
                   positive otherwise. The interface should match:
                   ``violation = con_func(x, sx)`` where the value ``sx`` is
                   given by
                   ``sx = sim_func(x)`` at runtime.
                   Note that any
                   ``constraint(x, sim_func(x), der=0) <= 0``
                   indicates that x is feaseible, while
                   ``constraint(x, sim_func(x), der=0) > 0``
                   indicates that x is infeasible, violating the constraint by
                   an amount proportional to the output.
                   It is the user's responsibility to ensure that after adding
                   all constraints, the feasible region is nonempty and has
                   nonzero measure in the design space.
                 * 'con_grad' (function): Evaluates the gradients of
                   ``con_func`` wrt s and sx. Interface should match:
                   ``dx, ds = con_grad(x, sx)`` where the value ``sx`` is
                   given by ``sx = sim_func(x)`` at runtime.
                   The outputs ``dx`` and ``ds`` represent the gradients with
                   respect to ``x`` and ``sx``, respectively.

        """

        for arg in args:
            # Check that the constraint dictionary is a legal format
            if not isinstance(arg, dict):
                raise TypeError("Each arg must be a Python dict")
            if 'con_func' in arg:
                if not callable(arg['con_func']):
                    raise TypeError("The 'con_func' must be callable")
                if len(inspect.signature(arg['con_func']).parameters) != 2:
                    raise ValueError("The 'con_func' must take 2 args")
            elif 'constraint' in arg:
                if not callable(arg['constraint']):
                    raise TypeError("The 'constraint' must be callable")
                if len(inspect.signature(arg['constraint']).parameters) != 2:
                    raise ValueError("The 'constraint' must take 2 args")
            else:
                raise AttributeError("Each arg must contain a 'con_func'")
            if 'con_grad' in arg:
                if not callable(arg['con_grad']):
                    raise TypeError("The 'con_grad' must be callable")
                if len(inspect.signature(arg['con_grad']).parameters) != 2:
                    raise ValueError("If present, 'con_grad' must take 2 args")
            # Check the constraint name
            if 'name' in arg:
                name = arg['name']
            else:
                name = f"c{self.p + 1}"
            check_names(name, self.des_schema, self.sim_schema,
                        self.obj_schema, self.con_schema)
            # Finally, if all else passed, add the constraint
            self.con_schema.append((name, 'f8'))
            if 'con_func' in arg:
                self.con_funcs.append(arg['con_func'])
            else:
                self.con_funcs.append(arg['constraint'])
            if 'con_grad' in arg:
                self.con_grads.append(arg['con_grad'])
            self.p += 1
        return

    def addAcquisition(self, *args):
        """ Add an acquisition function to the MOOP.

        Args:
            args (dict): Python dictionary of acquisition function info,
                including:
                 * 'acquisition' (AcquisitionFunction): An acquisition function
                   that maps from R^o --> R for scalarizing outputs.
                 * 'hyperparams' (dict): A dictionary of hyperparameters for
                   the acquisition functions. Can be omitted if no
                   hyperparameters are needed.

        """

        for arg in args:
            # Check that the acquisition dictionary is a legal format
            if not isinstance(arg, dict):
                raise TypeError("Every arg must be a Python dict")
            if 'acquisition' not in arg:
                raise AttributeError("The 'acquisition' key must be present")
            if 'hyperparams' in arg:
                if not isinstance(arg['hyperparams'], dict):
                    raise TypeError("When present, 'hyperparams' must be a "
                                    "Python dictionary")
                hps = arg['hyperparams']
            else:
                hps = {}
            hps["np_random_gen"] = self.np_random_gen
            try:
                acq = arg['acquisition'](1, np.zeros(1), np.ones(1), {})
            except BaseException:
                raise TypeError("'acquisition' must specify a child of the"
                                + " AcquisitionFunction class")
            if not isinstance(acq, structs.AcquisitionFunction):
                raise TypeError("'acquisition' must specify a child of the"
                                + " AcquisitionFunction class")
            # If all checks passed, add the acquisition to the list
            self.acq_tmp.append(arg['acquisition'])
            self.acq_hp.append(hps)
        return

    def compile(self):
        """ Compile the MOOP object and initialize its components.

        This locks the MOOP definition and jits all jit-able methods.

        This must be done *before* adding any simulation or objective data to
        the internal database.

        This cannot be done *after* simulation or objective data has been added
        to the internal database.

        """

        logging.info(" Compiling the MOOP object...")
        # For safety reasons, don't let silly users delete their data
        if self.n_dat > 0 or (len(self.sim_db) > 0 and
                              any([sdi['n'] > 0 for sdi in self.sim_db])):
            raise RuntimeError("Cannot re-compile a MOOP with a nonempty "
                               "database. If that's really what you want, "
                               "then please reset this MOOP.")
        # Verify that the MOOP is in a valid state before compiling
        if self.n_feature <= 0:
            raise RuntimeError("Cannot compile a MOOP with no design "
                               "variables.")
        if self.o <= 0:
            raise RuntimeError("Cannot compile a MOOP with no objectives.")
        if len(self.acq_tmp) == 0:
            warnings.warn("You are compiling a MOOP with no acquisition "
                          "functions. I'll let you do it for analysis "
                          "purposes, but the ``solve()`` command won't "
                          "work correctly until you recompile with "
                          "one or more acquisition functions...")
        logging.info("   Initializing MOOP solver's component objects...")
        # Reset the internal lists
        self.searches, self.surrogates = [], []
        self.acquisitions = []
        self.optimizer = None
        # Pre-create numpy arrays for initialization
        lbs = np.asarray(self.latent_lb)
        ubs = np.asarray(self.latent_ub)
        des_tols = np.asarray(self.latent_des_tols)
        # Jitting ParMOO embedders and extractors
        logging.info("   jitting and testing ParMOO's embedders...")
        xx1 = (lbs + ubs) / 2
        sx1 = np.zeros(self.m)
        try:
            self.extract = jax.jit(self._extract)
            x = self.extract(xx1)
            for key in self.des_schema:
                assert (key[0] in x)
        except BaseException:
            self.extract = self._extract
            x = self.extract(xx1)
            for key in self.des_schema:
                assert (key[0] in x)
            logging.info("     WARNING: 1 or more extractors failed to jit...")
        try:
            self.embed = jax.jit(self._embed)
            xx2 = self.embed(x)
            assert (xx2.shape == xx1.shape)
        except BaseException:
            self.embed = self._embed
            xx2 = self.embed(x)
            assert (xx2.shape == xx1.shape)
            logging.info("     WARNING: 1 or more embedders failed to jit...")
        try:
            self.unpack_sim = jax.jit(self._unpack_sim)
            sx = self.unpack_sim(sx1)
            for key in self.sim_schema:
                assert (key[0] in sx)
        except BaseException:
            self.unpack_sim = self._unpack_sim
            sx = self.unpack_sim(sx1)
            for key in self.sim_schema:
                assert (key[0] in sx)
            logging.info("     WARNING: MOOP._unpack_sim failed to jit...")
        try:
            self.pack_sim = jax.jit(self._pack_sim)
            sx2 = self.pack_sim(sx)
            assert (sx2.shape == sx1.shape)
        except BaseException:
            self.pack_sim = self._pack_sim
            sx2 = self.pack_sim(sx)
            assert (sx2.shape == sx1.shape)
            logging.info("     WARNING: MOOP._pack_sim failed to jit...")
        logging.info("   Done.")
        # Jitting ParMOO objectives and constraints
        logging.info("   jitting ParMOO's objective and constraints...")
        def gerr(x, sx): raise ValueError("1 or more grad func is undefined")
        try:
            self.vobj_funcs = self._vobj_funcs
        except BaseException:
            self.vobj_funcs = self._vobj_funcs
            logging.info("     WARNING: 1 or more obj_funcs failed to jit...")
        try:
            self.vcon_funcs = self._vcon_funcs
        except BaseException:
            self.vcon_funcs = self._vcon_funcs
            logging.info("     WARNING: 1 or more con_funcs failed to jit...")
        try:
            self.vpen_funcs = self._vpen_funcs
        except BaseException:
            self.vpen_funcs = self._vpen_funcs
            logging.info("     WARNING: MOOP._vpen_funcs failed to jit...")
        if len(self.obj_grads) == self.o:
            try:
                self.obj_bwd = self._obj_bwd
            except BaseException:
                self.obj_bwd = self._obj_bwd
                logging.info("     WARNING: 1 or more obj_grads failed to "
                             "jit...")
        else:
            self.obj_bwd = gerr
        if len(self.con_grads) == self.p:
            try:
                self.con_bwd = self._con_bwd
            except BaseException:
                self.con_bwd = self._con_bwd
                logging.info("     WARNING: 1 or more con_grads failed to "
                             "jit...")
        else:
            self.con_bwd = gerr
        if len(self.obj_grads) == self.o and len(self.con_grads) == self.p:
            try:
                self.pen_bwd = self._pen_bwd
            except BaseException:
                self.pen_bwd = self._pen_bwd
                logging.info("     WARNING: MOOP._pen_grads failed to jit...")
        else:
            self.pen_bwd = gerr
        logging.info("   Done.")
        # Initialize the simulation components
        for i in range(self.s):
            mi = self.m_list[i]
            hpi = self.sim_hp[i]
            hpi['des_tols'] = des_tols
            search_i = self.search_tmp[i]
            surrogate_i = self.sur_tmp[i]
            self.searches.append(search_i(mi, lbs, ubs, hpi))
            self.surrogates.append(surrogate_i(mi, lbs, ubs, hpi))
        # Initialize all acquisition functions
        for acqi, hpi in zip(self.acq_tmp, self.acq_hp):
            hpi['des_tols'] = des_tols
            self.acquisitions.append(acqi(self.o, lbs, ubs, hpi))
        # Initialize the surrogate optimizer
        self.opt_hp['des_tols'] = np.asarray(self.latent_des_tols)
        self.optimizer = self.opt_tmp(self.o,
                                      np.asarray(self.latent_lb),
                                      np.asarray(self.latent_ub),
                                      self.opt_hp)
        self.optimizer.setObjective(self._evaluate_objectives)
        self.optimizer.setSimulation(self._evaluate_surrogates,
                                     self._surrogate_uncertainty)
        self.optimizer.setPenalty(self._evaluate_penalty)
        self.optimizer.setConstraints(self._evaluate_constraints)
        for i, acquisition in enumerate(self.acquisitions):
            self.optimizer.addAcquisition(acquisition)
        self.optimizer.setTrFunc(self._set_surrogate_tr)
        logging.info("   Done.")
        # Initialize the optimizer database
        logging.info("   Initializing ParMOO's internal databases...")
        self.n_dat = 0
        self.data = {'x_vals': np.zeros((1, self.n_latent)),
                     'f_vals': np.zeros((1, self.o)),
                     'c_vals': np.zeros((1, 1))}
        # Initialize all the simulation databases
        for stype in self.sim_schema:
            if len(stype) > 2:
                mi = stype[2]
            else:
                mi = 1
            self.sim_db.append({'x_vals': np.zeros((1, self.n_latent)),
                                's_vals': np.zeros((1, mi)),
                                'n': 0,
                                'old': 0})
        logging.info("   Done.")
        # Set compiled flat go True
        logging.info(" Compilation finished.")
        self.compiled = True
        # Print problem summary
        logging.info(" Summary of ParMOO problem and settings:")
        logging.info(f"   {self.n_feature} design dimensions")
        logging.info(f"   {self.n_latent} embedded design dimensions")
        logging.info(f"   {self.m} simulation outputs")
        logging.info(f"   {self.s} simulations")
        for i in range(self.s):
            logging.info(f"     {self.m_list[i]} outputs for simulation {i}")
            logging.info(f"     {self.searches[i].budget} search evaluations" +
                         f" in iteration 0 for simulation {i}")
        logging.info(f"   {self.o} objectives")
        logging.info(f"   {self.p} constraints")
        logging.info(f"   {len(self.acquisitions)} acquisition functions")
        logging.info("   estimated simulation evaluations per iteration:" +
                     f" {len(self.acquisitions) * self.s}")
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
        self.checkpoint = checkpoint
        self.checkpoint_data = checkpoint_data
        self.checkpoint_file = filename
        return

    def getDesignType(self):
        """ Get the numpy dtype of all design points for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's design points.
            If no design variables have yet been added, returns None.

        """

        if self.n_feature < 1:
            return None
        else:
            return np.dtype(self.des_schema)

    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's simulation outputs.
            If no simulations have been given, returns None.

        """

        if self.m < 1:
            return None
        else:
            return np.dtype(self.sim_schema)

    def getObjectiveType(self):
        """ Get the numpy dtype of an objective point for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's objective points.
            If no objectives have yet been added, returns None.

        """

        if self.o < 1:
            return None
        else:
            return np.dtype(self.obj_schema)

    def getConstraintType(self):
        """ Get the numpy dtype of the constraint violations for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's constraint violation
            outputs. If no constraint functions have been given, returns None.

        """

        if self.p < 1:
            return None
        else:
            return np.dtype(self.con_schema)

    def checkSimDb(self, x, s_name):
        """ Check self.sim_db[s_name] to see if the design x was evaluated.

        Args:
            x (dict or numpy structured array): A numpy structured array or a
                Python dictionary specifying the keys/names and corresponding
                values of a design point to search for.

            s_name (str): The name of the simulation whose database will be
                searched.

        Returns:
            None or numpy.ndarray: returns None if x is not in
            self.sim_db[s_name] (up to the design tolerance). Otherwise,
            returns the corresponding value of sx.

        """

        # Extract the simulation name
        i = -1
        for j, sj in enumerate(self.sim_schema):
            if sj[0] == s_name:
                i = j
                break
        if i < 0 or i > self.s - 1:
            raise ValueError("s_name did not contain a legal name/index")
        # Check the database for previous evaluations of x
        xx = self.embed(x)
        des_tols = np.asarray(self.latent_des_tols)
        for j in range(self.sim_db[i]['n']):
            if np.all(np.abs(self.sim_db[i]['x_vals'][j, :] - xx) < des_tols):
                return self.sim_db[i]['s_vals'][j, :]
        return None

    def updateSimDb(self, x, sx, s_name):
        """ Update sim_db[s_name] by adding a design/simulation output pair.

        Args:
            x (dict or numpy structured array): A numpy structured array or a
                Python dictionary specifying the keys/names and corresponding
                values of a design point to add.

            sx (np.ndarray): A 1D array containing the corresponding
                simulation output(s).

            s_name (str): The name of the simulation to whose database the
                pair (x, sx) will be added into.

        """

        if not self.compiled:
            raise RuntimeError("Cannot begin adding items to the database "
                               "before compiling")
        # Extract the simulation name
        i = -1
        for j, sj in enumerate(self.sim_schema):
            if sj[0] == s_name:
                i = j
                break
        if i < 0 or i > self.s - 1:
            raise ValueError("s_name did not contain a legal name/index")
        xx = self.embed(x)
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
        # If various checkpointing modes are on, then save the current states
        if self.checkpoint_data:
            self.savedata(x, sx, s_name, filename=self.checkpoint_file)
        if self.checkpoint:
            self.save(filename=self.checkpoint_file)
        return

    def evaluateSimulation(self, x, s_name):
        """ Evaluate sim_func[s_name] and store the result in the database.

        Args:
            x (dict or numpy structured array): Either a numpy structured
                array or a Python dictionary with keys/names corresponding
                to the design variable names given and values containing
                the corresponding values of the design point to evaluate.

            s_name (str): The name of the simulation to evaluate.

        Returns:
            ndarray: A 1D array containing the output from the evaluation
            sx = simulation[s_name](x).

        """

        sx = self.checkSimDb(x, s_name)
        if sx is None:
            i = -1
            for j, sj in enumerate(self.sim_schema):
                if sj[0] == s_name:
                    i = j
                    break
            if i < 0 or i > self.s - 1:
                raise ValueError("s_name did not contain a legal name/index")
            sx = np.asarray(self.sim_funcs[i](x))
            self.updateSimDb(x, sx, s_name)
        return sx

    def addObjData(self, x, sx):
        """ Update the internal objective database by truly evaluating x.

        Args:
            x (dict or numpy structured array): Either a numpy structured
                array or Python dictionary containing the value of the design
                variable to add to ParMOO's database.

            sx (dict or numpy structured array): Either a numpy structured
                array or Python dictionary containing the values of the
                corresponding simulation outputs for ALL simulations involved
                in this MOOP -- sx['s_name'][:] contains the output(s)
                for sim_func['s_name'].

        """

        xx = self.embed(x)
        des_tols = np.asarray(self.latent_des_tols)
        # Initialize the database if needed
        if self.n_dat == 0:
            self.data['x_vals'][0, :] = xx
            self.data['f_vals'] = np.zeros((1, self.o))
            for i, obj_func in enumerate(self.obj_funcs):
                self.data['f_vals'][0, i] = obj_func(x, sx)
            # Check if there are constraint violations to maintain
            if self.p > 0:
                self.data['c_vals'] = np.zeros((1, self.p))
                for i, constraint_func in enumerate(self.con_funcs):
                    self.data['c_vals'][0, i] = constraint_func(x, sx)
            else:
                self.data['c_vals'] = np.zeros((1, 1))
            self.n_dat = 1
        # Check for duplicate values (up to the design tolerance)
        elif any([np.all(np.abs(xx - xj) < des_tols)
                  for xj in self.data['x_vals']]):
            return
        # Otherwise append the objectives
        else:
            self.data['x_vals'] = np.append(self.data['x_vals'], [xx], axis=0)
            fx = np.zeros(self.o)
            for i, obj_func in enumerate(self.obj_funcs):
                fx[i] = obj_func(x, sx)
            self.data['f_vals'] = np.append(self.data['f_vals'],
                                            [fx], axis=0)
            # Check if there are constraint violations to maintain
            if self.p > 0:
                cx = np.zeros(self.p)
                for i, constraint_func in enumerate(self.con_funcs):
                    cx[i] = constraint_func(x, sx)
                self.data['c_vals'] = np.append(self.data['c_vals'],
                                                [cx], axis=0)
            else:
                self.data['c_vals'] = np.append(self.data['c_vals'],
                                                [np.zeros(1)], axis=0)
            self.n_dat += 1
        return

    def iterate(self, k, ib=None):
        """ Perform an iteration of ParMOO's solver and generate candidates.

        Generates a batch of suggested candidate points (design points)
        or (candidate point, simulation name) pairs and returns to the
        user for further processing. Note, this method may produce
        duplicates.

        Args:
            k (int): The iteration counter (corresponding to MOOP.iteration).

            ib (int, optional): The index of the acquisition function to
                optimize and add to the current batch. Defaults to None,
                which optimizes all acquisition functions and adds all
                resulting candidates to the batch.

        Returns:
            (list): A list of tuples (design points, simulation name)
            specifying the unfiltered list of candidates that ParMOO
            recommends for true simulation evaluations. Specifically:
             * The first entry in each tuple is either a numpy structured
               array or a Python dictionary specifying the design point
               to evaluate.
             * The second entry in the tuple is the (str) name of the
               simulation to evaluate at the design point specified above.

        """

        # Check that the inputs and MOOP states are legal
        if isinstance(k, int):
            if k < 0:
                raise ValueError("k must be nonnegative")
        else:
            raise TypeError("k must be an int type")
        if isinstance(ib, list) and all([isinstance(ibj, int) for ibj in ib]):
            for ibj in ib:
                if ibj < 0 or ibj >= len(self.acquisitions):
                    raise ValueError(f"invalid index found in ib: {ibj}")
        elif ib is not None:
            raise TypeError("when present, ib must be a list of int types")
        else:
            ib = [i for i in range(len(self.acquisitions))]
        if self.n_latent == 0:
            raise AttributeError("there are no design vars for this problem")
        if self.o == 0:
            raise AttributeError("there are no objectives for this problem")
        # Special rule for the k=0 iteration
        xbatch = []
        if k == 0:
            # Compile the MOOP if needed
            if not self.compiled:
                self.compile()
            # Generate search data
            for j, search in enumerate(self.searches):
                des = search.startSearch(np.asarray(self.latent_lb),
                                         np.asarray(self.latent_ub))
                for xi in des:
                    xbatch.append((self.extract(xi),
                                   self.sim_schema[j][0]))
        # General case for k>0 iterations
        else:
            # Set acquisition function targets
            x0 = np.zeros((len(self.acquisitions), self.n_latent))
            for i, acqi in enumerate(self.acquisitions):
                x0[i, :] = acqi.setTarget(self.data, self._evaluate_penalty)
            # Solve the surrogate problem
            x_candidates = self.optimizer.solve(x0)
            # Create a batch for filtering methods
            for i, acqi in enumerate(self.acquisitions):
                if self.s > 0:
                    for sn in self.sim_schema:
                        xbatch.append((self.extract(x_candidates[i, :]), sn[0]))
                else:
                    xbatch.append(self.extract(x_candidates[i, :]))
        return xbatch

    def filterBatch(self, *args):
        """ Filter a batch produced by ParMOO's MOOP.iterate method.

        Accepts one or more batches of candidate design points, produced
        by the MOOP.iterate() method and checks both the batch and ParMOO's
        database for redundancies. Any redundant points (up to the design
        tolerance) are replaced by model improving points, using each
        surrogate's Surrogate.improve() method.

        Args:
            *args (list of tuples): The list of unfiltered candidates
            returned by the ``MOOP.iterate()`` method.

        Returns:
            (list): A filtered list of tuples, matching the format of the
            ``MOOP.iterate()`` output, but with redundant points removed
            and suitably replaced.

        """

        # Create an empty list to store the filtered and embedded batches
        fbatch = []
        ebatch = []
        for xbatch in args:
            # Evaluate all of the simulations at the candidate solutions
            if self.s > 0:
                # For each design in the database
                des_tols = np.asarray(self.latent_des_tols)
                for xtuple in xbatch:
                    # Extract the xtuple into xi/si pair if needed
                    if isinstance(xtuple, tuple):
                        xi = xtuple[0]
                        si = []
                        for i, ssi in enumerate(self.sim_schema):
                            if ssi[0] == xtuple[1]:
                                si.append(i)
                                break
                    else:
                        xi = xtuple
                        si = [i for i in range(self.s)]
                    # This 2nd extract/embed, while redundant, is necessary
                    # for categorical variables to be processed correctly
                    xxi = self.embed(xi)
                    # Check whether it has been evaluated by any simulation
                    for i in si:
                        namei = self.sim_schema[i][0]
                        if all([np.any(np.abs(xxi - xj) > des_tols)
                                or namei != j for (xj, j) in ebatch]) \
                           and self.checkSimDb(xi, namei) is None:
                            # If not, add it to the fbatch and ebatch
                            fbatch.append((xi, namei))
                            ebatch.append((xxi, namei))
                        else:
                            # Try to improve surrogate (locally then globally)
                            x_improv = self.surrogates[i].improve(xxi, False)
                            # Again, this is needed to handle categorical vars
                            ibatch = [self.embed(self.extract(xk))
                                      for xk in x_improv]
                            while (any([any([np.all(np.abs(xj - xk) < des_tols)
                                             and namei == j for (xj, j)
                                             in ebatch])
                                        for xk in ibatch]) or
                                   any([self.checkSimDb(self.extract(xk), namei)
                                        is not None for xk in ibatch])):
                                x_improv = self.surrogates[i].improve(xxi, True)
                                ibatch = [self.embed(self.extract(xk))
                                          for xk in x_improv]
                            # Add improvement points to the fbatch
                            for xj in ibatch:
                                fbatch.append((self.extract(xj), namei))
                                ebatch.append((xj, namei))
            else:
                # If there were no simulations, just add all points to fbatch
                des_tols = np.asarray(self.latent_des_tols)
                for xi in xbatch:
                    # This 2nd extract/embed, while redundant, is necessary
                    # for categorical variables to be processed correctly
                    xxi = self.embed(xi)
                    if all([np.any(np.abs(xxi - xj) > des_tols)
                            for (xj, j) in ebatch]):
                        fbatch.append((xi, -1))
                        ebatch.append((xxi, -1))
        return fbatch

    def updateAll(self, k, batch):
        """ Update all surrogates given a batch of freshly evaluated data.

        Args:
            k (int): The iteration counter (corresponding to MOOP.iteration).

            batch (list): A list of ordered pairs (tuples), each specifying
                a design point that was evaluated in this iteration, whose
                format matches the output of ``MOOP.iterate()``.

        """

        # Special rules for k=0, vs k>0
        if k == 0:
            self._fit_surrogates()
            if self.s > 0:
                # Check every point in sim_db[0]
                des_tols = np.asarray(self.latent_des_tols)
                for xi, si in zip(self.sim_db[0]['x_vals'],
                                  self.sim_db[0]['s_vals']):
                    sim = np.zeros(self.m)
                    sim[0:self.m_list[0]] = si[:]
                    m_count = self.m_list[0]
                    is_shared = True
                    # Check for xi in sim_db[1:s]
                    for j in range(1, self.s):
                        is_shared = False
                        for xj, sj in zip(self.sim_db[j]['x_vals'],
                                          self.sim_db[j]['s_vals']):
                            # If found, update sim value and break loop
                            if np.all(np.abs(xi - xj) < des_tols):
                                sim[m_count:m_count + self.m_list[j]] = sj[:]
                                m_count = m_count + self.m_list[j]
                                is_shared = True
                                break
                        if not is_shared:
                            break
                    # If xi was in every sim_db, add it to the database
                    if is_shared:
                        self.addObjData(self.extract(xi), self.unpack_sim(sim))
        else:
            # If any constraints are violated, increase lam toward the limit
            for (xi, i) in batch:
                xxi = self.embed(xi)
                sxi = self._evaluate_surrogates(xxi)
                eps = np.sqrt(self.epsilon)
                if np.any(self._evaluate_constraints(xxi, sxi) > eps):
                    self.lam = min(1e4, self.lam * 2.0)
                    break
            # Update the models and objective database
            self._update_surrogates()
            for xi in batch:
                (x, i) = xi
                xx = self.embed(x)
                is_shared = True
                sim = np.zeros(self.m)
                m_count = 0
                if self.s > 0:
                    # Check for xi in every sim_db
                    des_tols = np.asarray(self.latent_des_tols)
                    for j in range(self.s):
                        is_shared = False
                        for xj, sj in zip(self.sim_db[j]['x_vals'],
                                          self.sim_db[j]['s_vals']):
                            # If found, update sim value and break loop
                            if np.all(np.abs(xx - xj) < des_tols):
                                sim[m_count:m_count + self.m_list[j]] = sj[:]
                                m_count = m_count + self.m_list[j]
                                is_shared = True
                                break
                        # If not found, stop checking
                        if not is_shared:
                            break
                # If xi was in every sim_db, add it to the database and report
                # to the optimizer
                if is_shared:
                    fx = np.zeros(self.o)
                    sx = self.unpack_sim(sim)
                    sdx = self.unpack_sim(np.zeros(self.m))
                    for i, obj_func in enumerate(self.obj_funcs):
                        fx[i] = obj_func(x, sx)
                    self.addObjData(x, sx)
                    self.optimizer.returnResults(xx, fx, sim, np.zeros(self.m))
        # If checkpointing is on, save the moop before continuing
        if self.checkpoint:
            self.save(filename=self.checkpoint_file)
        return

    def solve(self, iter_max=None, sim_max=None):
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
            iter_max (int): The max limit for ParMOO's internal iteration
                counter. ParMOO keeps track of how many iterations it has
                completed internally. This value k specifies the stopping
                criteria for ParMOO.

            sim_max (int): The max limit for ParMOO's simulation database,
                i.e., the simulation evaluation budget.

        """

        logging.info(" Beginning new run of ParMOO...")
        # Check that at least one budget variable was given
        if iter_max is None and sim_max is None:
            raise ValueError("At least one of the following arguments " +
                             "must be set: 'iter_max' or 'sim_max'")
        # Check that the iter_max is a legal integer
        if isinstance(iter_max, int):
            if iter_max < 0:
                raise ValueError("When present, iter_max must be nonnegative")
        elif iter_max is not None:
            raise TypeError("When present, iter_max must be an int type")
        # Check that the sim_max is a legal integer
        if isinstance(sim_max, int):
            if sim_max < 0:
                raise ValueError("When present, sim_max must be nonnegative")
        elif sim_max is not None:
            raise TypeError("When present, sim_max must be an int type")
        # Set iter_max large enough if None
        if iter_max is None:
            if self.s == 0:
                raise ValueError("If 0 simulations are given, then iter_max" +
                                 "must be provided")
            iter_max = sim_max
        # Compile the MOOP if needed
        if not self.compiled:
            self.compile()
        # Count total sims to exhaust iter_max if sim_max is None
        total_search_budget = 0
        for search in self.searches:
            total_search_budget += search.budget
        if sim_max is None:
            sim_max = total_search_budget
            sim_max += iter_max * len(self.acquisitions) * self.s + 1
        # Warning for the uninitiated
        if sim_max <= total_search_budget:
            warnings.warn("You are running ParMOO with a total search budget" +
                          f" of {total_search_budget} and a sim_max of " +
                          f"just {sim_max}... This will result in pure " +
                          "design space exploration with no exploitation/" +
                          "optimization. Consider increasing the value of " +
                          "sim_max, decreasing your search_budget, " +
                          "or using the iter_max stopping criteria, unless " +
                          "you are really only interested in design space " +
                          "exploration without exploitation/optimization.")

        logging.info(f"   iteration limit: {iter_max}")
        logging.info(f"   total simulation budget: {sim_max}")
        logging.info(" Done.")

        # Perform iterations until budget is exceeded
        logging.info(" Entering main iteration loop:")

        # Reset the iteration start
        start = self.iteration
        total_sims = 0
        for k in range(start, iter_max + 1):
            # Check for the sim_max stop condition
            if total_sims >= sim_max:
                break
            # Track iteration counter
            self.iteration = k
            # Generate a batch by running one iteration and filtering results
            logging.info(f"   Iteration {self.iteration: >4}:")
            logging.info("     generating batch...")
            xbatch = self.iterate(self.iteration)
            fbatch = self.filterBatch(xbatch)
            logging.info(f"     {len(fbatch)} candidate designs generated.")
            if self.s > 0:
                # Evaluate the batch
                logging.info("     evaluating batch...")
                for xi in fbatch:
                    (x, i) = xi
                    logging.info(f"       evaluating design: {x}" +
                                 f" for simulation: {i}...")
                    sx = self.evaluateSimulation(x, i)
                    logging.info(f"         result: {sx}")
                    # Count total simulations taken
                    total_sims += 1
                    if total_sims >= sim_max:
                        logging.info(f"   sim_max of {sim_max} reached")
                logging.info(f"     finished evaluating {len(fbatch)}" +
                             " simulations.")
            logging.info("     updating models and internal databases...")
            # Update the database
            self.updateAll(self.iteration, fbatch)
            logging.info("   Done.")
        logging.info(" Done.")
        logging.info(f" ParMOO has successfully completed {self.iteration} " +
                     "iterations.")
        return

    def getPF(self, format='ndarray'):
        """ Extract nondominated and efficient sets from internal databases.

        Args:
            format (str, optional): Either 'ndarray' (default) or 'pandas',
                in order to produce output as a numpy structured array or
                pandas dataframe. Note: format='pandas' is only valid for
                named inputs.

        Returns:
            numpy structured array or pandas DataFrame: Either a structured
            array or dataframe (depending on the option selected above)
            whose column/key names match the names of the design variables,
            objectives, and constraints. It contains a discrete approximation
            of the Pareto front and efficient set.

        """

        # Get the solutions using function call
        if self.n_dat > 0:
            pf = updatePF(self.data, {})
        else:
            pf = {'x_vals': np.zeros(0),
                  'f_vals': np.zeros(0),
                  'c_vals': np.zeros(0)}
        # Build the data type
        dt = []
        for dname in self.des_schema:
            dt.append((str(dname[0]), dname[1]))
        for fname in self.obj_schema:
            dt.append((str(fname[0]), fname[1]))
        for cname in self.con_schema:
            dt.append((str(cname[0]), cname[1]))
        # Initialize result array
        result = np.zeros(pf['x_vals'].shape[0], dtype=dt)
        # Extract all results
        if self.n_dat > 0:
            for i, xi in enumerate(pf['x_vals']):
                xxi = self.extract(xi)
                for (name, t) in self.des_schema:
                    result[str(name)][i] = xxi[name]
            for i, (name, t) in enumerate(self.obj_schema):
                result[str(name)][:] = pf['f_vals'][:, i]
            for i, (name, t) in enumerate(self.con_schema):
                result[str(name)][:] = pf['c_vals'][:, i]
        if format == 'pandas':
            return pd.DataFrame(result)
        elif format == 'ndarray':
            return result
        else:
            raise ValueError(str(format) + " is an invalid value for 'format'")

    def getSimulationData(self, format='ndarray'):
        """ Extract all computed simulation outputs from the MOOP's database.

        Args:
            format (str, optional): Either 'ndarray' (default) or 'pandas',
                in order to produce output as a numpy structured array or
                pandas dataframe. Note: format='pandas' is only valid for
                named inputs.

        Returns:
            dict: A Python dictionary whose keys match the names of the
            simulations. Each value is either a numpy structured array or
            pandas dataframe (depending on the option selected above)
            whose column/key names match the names of the design variables
            plus either and 'out' field for single-output simulations,
            or 'out_1', 'out_2', ... for multi-output simulations.

        """

        # Build a results dict with a key for each simulation
        result = {}
        for i, sname in enumerate(self.sim_schema):
            # Construct the dtype for this simulation database
            dt = []
            for dname in self.des_schema:
                dt.append((str(dname[0]), dname[1]))
            if len(sname) == 2:
                dt.append(('out', sname[1]))
            else:
                dt.append(('out', sname[1], sname[2]))
            # Fill the results array
            result[sname[0]] = np.zeros(self.sim_db[i]['n'], dtype=dt)
            if self.sim_db[i]['n'] > 0:
                for j, xj in enumerate(self.sim_db[i]['x_vals']):
                    xxj = self.extract(xj)
                    for (name, t) in self.des_schema:
                        result[sname[0]][name][j] = xxj[name]
                if len(sname) > 2:
                    result[sname[0]]['out'] = self.sim_db[i]['s_vals']
                else:
                    result[sname[0]]['out'] = self.sim_db[i]['s_vals'][:, 0]
        if format == 'pandas':
            # For simulation data, converting to pandas is a little more
            # complicated...
            result_pd = {}
            for i, snamei in enumerate(result.keys()):
                rtempi = {}
                for (name, t) in self.des_schema:
                    rtempi[name] = result[snamei][name]
                # Need to break apart the output column manually
                if self.m_list[i] > 1:
                    for i in range(self.m_list[i]):
                        rtempi[f'out_{i}'] = result[snamei]['out'][:, i]
                else:
                    rtempi['out'] = result[snamei]['out'][:, 0]
                # Create dictionary of dataframes, indexed by sim names
                result_pd[snamei] = pd.DataFrame(rtempi)
            return result_pd
        elif format == 'ndarray':
            return result
        else:
            raise ValueError(str(format) + "is an invalid value for 'format'")

    def getObjectiveData(self, format='ndarray'):
        """ Extract all computed objective scores from this MOOP's database.

        Args:
            format (str, optional): Either 'ndarray' (default) or 'pandas',
                in order to produce output as a numpy structured array or
                pandas dataframe. Note: format='pandas' is only valid for
                named inputs.

        Returns:
            numpy structured array or pandas DataFrame: Either a structured
            array or dataframe (depending on the option selected above)
            whose column/key names match the names of the design variables,
            objectives, and constraints. It contains the results for every
            fully evaluated design point.

        """

        # Build the data type
        dt = []
        for dname in self.des_schema:
            dt.append(dname)
        for fname in self.obj_schema:
            dt.append(fname)
        for cname in self.con_schema:
            dt.append(cname)
        # Initialize result array
        if self.n_dat > 0:
            result = np.zeros(self.data['x_vals'].shape[0], dtype=dt)
        else:
            result = np.zeros(0, dtype=dt)
        # Extract all results
        if self.n_dat > 0:
            for i, xi in enumerate(self.data['x_vals']):
                xxi = self.extract(xi)
                for (name, t) in self.des_schema:
                    result[name][i] = xxi[name]
            for i, (name, t) in enumerate(self.obj_schema):
                result[name][:] = self.data['f_vals'][:, i]
            for i, (name, t) in enumerate(self.con_schema):
                result[name][:] = self.data['c_vals'][:, i]
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

        # Check whether the file exists first
        exists = file_exists(filename + ".moop")
        if exists and self.new_checkpoint:
            raise OSError("Creating a new checkpoint file, but " +
                          filename + ".moop already exists! " +
                          "Move the existing file to a new location or " +
                          "delete it, so that ParMOO doesn't accidentally " +
                          "overwrite your data...")
        # Create a serializable ParMOO dictionary
        parmoo_state = {'m': self.m,
                        'm_list': self.m_list,
                        'n_embed': self.n_embed,
                        'n_feature': self.n_feature,
                        'n_latent': self.n_latent,
                        'o': self.o,
                        'p': self.p,
                        's': self.s,
                        'feature_des_tols': self.feature_des_tols,
                        'latent_des_tols': self.latent_des_tols,
                        'latent_lb': self.latent_lb,
                        'latent_ub': self.latent_ub,
                        'des_schema': self.des_schema,
                        'sim_schema': self.sim_schema,
                        'obj_schema': self.obj_schema,
                        'con_schema': self.con_schema,
                        'iteration': self.iteration,
                        'lam': self.lam,
                        'checkpoint': self.checkpoint,
                        'checkpoint_data': self.checkpoint_data,
                        'checkpoint_file': self.checkpoint_file,
                        'np_random_state': self.np_random_gen.get_state(),
                       }
        # Pickle and add a list of the model and solver hyperparameters
        parmoo_state['hyperparams'] = []
        for hpi in [self.emb_hp, self.acq_hp, self.opt_hp, self.sim_hp]:
            parmoo_state['hyperparams'].append(
                codecs.encode(pickle.dumps(hpi), "base64").decode())
        # Add the names/modules for all components of the MOOP definition
        parmoo_state['embedders'] = [(embedder.__class__.__name__,
                                      embedder.__class__.__module__)
                                    for embedder in self.embedders]
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
        parmoo_state['obj_funcs'] = []
        parmoo_state['obj_funcs_info'] = []
        for fi in self.obj_funcs:
            if type(fi).__name__ == "function":
                parmoo_state['obj_funcs'].append((fi.__name__, fi.__module__))
                parmoo_state['obj_funcs_info'].append("function")
            else:
                parmoo_state['obj_funcs'].append((fi.__class__.__name__,
                                                   fi.__class__.__module__))
                parmoo_state['obj_funcs_info'].append(
                        codecs.encode(pickle.dumps(fi), "base64").decode())
        parmoo_state['con_funcs'] = []
        parmoo_state['con_funcs_info'] = []
        for ci in self.con_funcs:
            if type(si).__name__ == "function":
                parmoo_state['con_funcs'].append((ci.__name__,
                                                    ci.__module__))
                parmoo_state['con_funcs_info'].append("function")
            else:
                parmoo_state['con_funcs'].append((ci.__class__.__name__,
                                                    ci.__class__.__module__))
                parmoo_state['con_funcs_info'].append(
                        codecs.encode(pickle.dumps(ci), "base64").decode())
        # Store names/modules of solver component classes
        parmoo_state['optimizer'] = (self.optimizer.__class__.__name__,
                                     self.optimizer.__class__.__module__)
        parmoo_state['searches'] = [(search.__class__.__name__,
                                     search.__class__.__module__)
                                    for search in self.searches]
        parmoo_state['surrogates'] = [(sur.__class__.__name__,
                                       sur.__class__.__module__)
                                      for sur in self.surrogates]
        parmoo_state['acquisitions'] = [(acq.__class__.__name__,
                                         acq.__class__.__module__)
                                        for acq in self.acquisitions]
        # Try to save optimizer object state
        try:
            fname = filename + ".optimizer"
            fname_tmp = "." + fname + ".swap"
            self.optimizer.save(fname_tmp)
            shutil.move(fname_tmp, fname)
        except NotImplementedError:
            pass
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
        # Serialize the internal databases
        parmoo_state['n_dat'] = self.n_dat
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
        # Save the serialized ParMOO dictionary
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

        PYDOCS = "https://docs.python.org/3/tutorial/modules.html" + \
                 "#the-module-search-path"

        # Load the serialized dictionary object
        fname = filename + ".moop"
        with open(fname, 'r') as fp:
            parmoo_state = json.load(fp)
        # Reload intrinsic types (scalar values and Python lists)
        self.m = parmoo_state['m']
        self.m_list = parmoo_state['m_list']
        self.n_embed = parmoo_state['n_embed']
        self.n_feature = parmoo_state['n_feature']
        self.n_latent = parmoo_state['n_latent']
        self.o = parmoo_state['o']
        self.p = parmoo_state['p']
        self.s = parmoo_state['s']
        self.feature_des_tols = parmoo_state['feature_des_tols']
        self.latent_des_tols = parmoo_state['latent_des_tols']
        self.latent_lb = parmoo_state['latent_lb']
        self.latent_ub = parmoo_state['latent_ub']
        self.des_schema = parmoo_state['des_schema']
        self.sim_schema = parmoo_state['sim_schema']
        self.obj_schema = parmoo_state['obj_schema']
        self.con_schema = parmoo_state['con_schema']
        self.iteration = parmoo_state['iteration']
        self.lam = parmoo_state['lam']
        self.checkpoint = parmoo_state['checkpoint']
        self.checkpoint_data = parmoo_state['checkpoint_data']
        self.checkpoint_file = parmoo_state['checkpoint_file']
        self.np_random_gen = np.random.default_rng()
        self.np_random_gen.set_state(parmoo_state['np_random_state'])
        # Recover the pickled hyperparameter dictionaries
        hps = []
        for i, hpi in enumerate(parmoo_state['hyperparams']):
            hps.append(pickle.loads(codecs.decode(hpi.encode(), "base64")))
            hps[i]['np_random_gen'] = self.np_random_gen
        self.emb_hp = hps[0]
        self.acq_hp = hps[1]
        self.opt_hp = hps[2]
        self.sim_hp = hps[3]
        # Recover design vars, sims, objectives, and constraints by module name
        self.embedders = []
        for i, (emb_name, emb_mod) in enumerate(parmoo_state['embedders']):
            try:
                mod = import_module(emb_mod)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f"module: {emb_mod} could not be "
                                          "loaded. Please make sure that "
                                          f"{emb_mod} exists on this machine "
                                          "and is part of the module search "
                                          "path: " + PYDOCS)
            try:
                new_emb = getattr(mod, emb_name)
            except KeyError:
                raise KeyError(f"function: {emb_name} defined in"
                               f"{emb_mod} could not be loaded."
                               f"Please make sure that {emb_name} is "
                               f"defined in {emb_mod} with global scope.")
            toadd = new_emb(self.emb_hp[i])
            self.embedders.append(toadd)
        self.sim_funcs = []
        for (sim_name, sim_mod), info in zip(parmoo_state['sim_funcs'],
                                             parmoo_state['sim_funcs_info']):
            try:
                mod = import_module(sim_mod)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f"module: {sim_mod} could not be "
                                          "loaded. Please make sure that "
                                          f"{sim_mod} exists on this machine "
                                          "and is part of the module search "
                                          "path: " + PYDOCS)
            try:
                sim_ptr = getattr(mod, sim_name)
            except KeyError:
                raise KeyError(f"function: {sim_name} defined in"
                               f"{sim_mod} could not be loaded."
                               f"Please make sure that {sim_name} is "
                               f"defined in {sim_mod} with global scope.")
            if info == "function":
                toadd = sim_ptr
            else:
                toadd = pickle.loads(codecs.decode(info.encode(), "base64"))
            self.sim_funcs.append(toadd)
        self.obj_funcs = []
        for (obj_name, obj_mod), info in zip(parmoo_state['obj_funcs'],
                                             parmoo_state['obj_funcs_info']):
            try:
                mod = import_module(obj_mod)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f"module: {obj_mod} could not be "
                                          "loaded. Please make sure that "
                                          f"{obj_mod} exists on this machine "
                                          "and is part of the module search "
                                          "path: " + PYDOCS)
            try:
                obj_ptr = getattr(mod, obj_name)
            except KeyError:
                raise KeyError(f"function: {obj_name} defined in"
                               f"{obj_mod} could not be loaded."
                               f"Please make sure that {obj_name} is "
                               f"defined in {obj_mod} with global scope.")
            if info == "function":
                toadd = obj_ptr
            else:
                toadd = pickle.loads(codecs.decode(info.encode(), "base64"))
            self.obj_funcs.append(toadd)
        self.con_funcs = []
        for (con_name, con_mod), info in zip(parmoo_state['con_funcs'],
                                             parmoo_state['con_funcs_info']):
            try:
                mod = import_module(con_mod)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f"module: {con_mod} could not be "
                                          "loaded. Please make sure that "
                                          f"{con_mod} exists on this machine "
                                          "and is part of the module search "
                                          "path: " + PYDOCS)
            try:
                con_ptr = getattr(mod, con_name)
            except KeyError:
                raise KeyError(f"function: {con_name} defined in"
                               f"{con_mod} could not be loaded."
                               f"Please make sure that {con_name} is "
                               f"defined in {con_mod} with global scope.")
            if info == "function":
                toadd = con_ptr
            else:
                toadd = pickle.loads(codecs.decode(info.encode(), "base64"))
            self.con_funcs.append(toadd)
        # Recover solver component classes by their module name
        mod = import_module(parmoo_state['optimizer'][1])
        self.opt_tmp = getattr(mod, parmoo_state['optimizer'][0])
        self.search_tmp = []
        for i, (s_name, s_mod) in enumerate(parmoo_state['searches']):
            mod = import_module(s_mod)
            new_search = getattr(mod, s_name)
            self.search_tmp.append(new_search)
        self.sur_tmp = []
        for i, (s_name, s_mod) in enumerate(parmoo_state['surrogates']):
            mod = import_module(s_mod)
            new_sur = getattr(mod, s_name)
            self.sur_tmp.append(new_sur)
        self.acq_tmp = []
        for i, (a_name, a_mod) in enumerate(parmoo_state['acquisitions']):
            mod = import_module(a_mod)
            new_acq = getattr(mod, a_name)
            self.acq_tmp.append(new_acq)
        # Re-compile the MOOP
        self.compile()
        # Try to re-load each solver component's previous state
        try:
            fname = filename + ".optimizer"
            self.optimizer.load(fname)
        except NotImplementedError:
            pass
        for i in range(self.s):
            try:
                fname = filename + ".search." + str(i + 1)
                self.searches[i].load(fname)
            except NotImplementedError:
                pass
            try:
                fname = filename + ".surrogate." + str(i + 1)
                self.surrogates[i].load(fname)
            except NotImplementedError:
                pass
        for i in range(len(self.acquisitions)):
            try:
                fname = filename + ".acquisition." + str(i + 1)
                self.acquisitions[i].load(fname)
            except NotImplementedError:
                pass
        # Re-load the serialized internal databases
        self.n_dat = parmoo_state['n_dat']
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

        # Check whether file exists first
        exists = file_exists(filename + ".simdb.json")
        if exists and self.new_data:
            raise OSError("Creating a new save file, but " +
                          filename + ".simdb.json already exists! " +
                          "Move the existing file to a new location or " +
                          "delete it so that ParMOO doesn't overwrite your " +
                          "existing data...")
        # Unpack x/sx pair into a dict for saving
        toadd = {'sim_id': s_name}
        for dname in self.des_schema:
            key = dname[0]
            if np.issubdtype(x[key], np.integer) or \
               jnp.issubdtype(x[key], jnp.integer):
                toadd[key] = int(x[key])
            elif np.issubdtype(x[key], np.floating) or \
                 jnp.issubdtype(x[key], jnp.floating):
                toadd[key] = float(x[key])
            else:
                toadd[key] = str(x[key])
        if isinstance(sx, np.ndarray) or isinstance(sx, jnp.ndarray):
            toadd['out'] = [float(sxi) for sxi in sx]
        else:
            toadd['out'] = float(sx)
        # Save in file with proper exension
        fname = filename + ".simdb.json"
        with open(fname, 'a') as fp:
            json.dump(toadd, fp)
        self.new_data = False
        return

    def _embed(self, x):
        """ Embed a design input as a n-dimensional vector for ParMOO.

        Args:
            x (dict): Either a numpy structured array or Python dictionary
                whose keys match the design variable names, and whose
                values contain design variable values.

        Returns:
            ndarray: A 1D array of length n_latent containing the embedded
            design vector.

        """

        xx = []
        for i, ei in enumerate(self.embedders):
            xx.append(ei.embed(x[self.des_schema[i][0]]))
        return jnp.concatenate(xx, axis=None)

    def _extract(self, x):
        """ Extract a design variable from an n-dimensional vector.

        Args:
            x (ndarray): A 1D array of length n_latent containing the embedded
                design vector.

        Returns:
            dict: A Python dictionary whose keys match the design variable
            names, and whose values contain design variable values.

        """

        xx = {}
        istart = 0
        for i, ei in enumerate(self.embedders):
            iend = istart + self.n_embed[i]
            xx[self.des_schema[i][0]] = ei.extract(x[istart:iend])
            istart = iend
        return xx

    def _pack_sim(self, sx):
        """ Pack a simulation output into a m-dimensional vector.

        Args:
            sx (dict): A dictionary with keys corresponding to simulation
                names and values corresponding to simulation outputs.

        Returns:
            ndarray: A 1D ndarray of length m containing the vectorized
            simulation outputs.

        """

        sx_list = [self.empty]
        for i in range(self.s):
            sx_list.append(sx[self.sim_schema[i][0]])
        return jnp.concatenate(sx_list, axis=None)

    def _unpack_sim(self, sx):
        """ Extract a simulation output from a m-dimensional vector.

        Args:
            sx (ndarray): A 1D array of length m containing the vectorized
                simulation outputs.

        Returns:
            dict: A dictionary with keys corresponding to simulation names
            and values corresponding to simulation outputs.

        """

        sx_out = {}
        istart = 0
        for i, mi in enumerate(self.m_list):
            iend = istart + mi
            sx_out[self.sim_schema[i][0]] = sx[istart:iend]
            istart = iend
        return sx_out

    def _vobj_funcs(self, x, sx):
        """ Jittable evaluation of all objectives from the feature space.

        Args:
            x (dict): A Python dictionary containing the design point to
                evaluate.

            sx (dict): A Python dictionary containing the simulation outputs
                at x.

        Returns:
            ndarray: A 1D array containing the result of the evaluation.

        """

        fx_list = [self.empty]
        for obj_func in self.obj_funcs:
            fx_list.append(obj_func(x, sx))
        return jnp.concatenate(fx_list, axis=None)

    def _vcon_funcs(self, x, sx):
        """ Jittable evaluation of all constraints from the feature space.

        Args:
            x (dict): A Python dictionary containing the design point to
                evaluate.

            sx (dict): A Python dictionary containing the simulation outputs
                at x.

        Returns:
            ndarray: A 1D array containing the list of constraint violations
            at x, where a negative or zero score implies feasibility.

        """

        cx_list = [self.empty]
        for con_func in self.con_funcs:
            cx_list.append(con_func(x, sx))
        return jnp.concatenate(cx_list, axis=None)

    def _vpen_funcs(self, x, sx, cx, lamx):
        """ Jittable evaluation of all penalties from the feature space.

        Args:
            x (dict): A Python dictionary containing the design point to
                evaluate.

            sx (dict): A Python dictionary containing the simulation outputs
                at x.

            cx (float): The aggregated constraint violations at x.

            lamx (float): The penalty parameter to apply.

        Returns:
            ndarray: A 1D array containing the result of the evaluation.

        """

        px = cx * lamx
        fx_list = [self.empty]
        for obj_func in self.obj_funcs:
            fx_list.append(obj_func(x, sx) + px)
        return jnp.concatenate(fx_list, axis=None)

    def _fit_surrogates(self):
        """ Fit the surrogate models using the current sim databases. """

        for i in range(self.s):
            n_new = self.sim_db[i]['n']
            self.surrogates[i].fit(self.sim_db[i]['x_vals'][:n_new, :],
                                   self.sim_db[i]['s_vals'][:n_new, :])
            self.sim_db[i]['old'] = self.sim_db[i]['n']
        return

    def _update_surrogates(self):
        """ Update the surrogate models using the current sim databases. """

        for i in range(self.s):
            n_old = self.sim_db[i]['old']
            n_new = self.sim_db[i]['n']
            self.surrogates[i].update(self.sim_db[i]['x_vals'][n_old:n_new, :],
                                      self.sim_db[i]['s_vals'][n_old:n_new, :])
            self.sim_db[i]['old'] = self.sim_db[i]['n']
        return

    def _set_surrogate_tr(self, center, radius):
        """ Alert the surrogate functions of a new trust region.

        Args:
            center (ndarray): A 1D array containing the (embedded) coordinates
                of the new trust region center.

            radius (ndarray or float): The trust region radius.

        """

        for surrogate in self.surrogates:
            surrogate.setTrustRegion(center, radius)
        # Compile and set the optimizer attributes to compiled functions
        self._compile()
        self.optimizer.setObjective(self.evaluate_objectives)
        self.optimizer.setPenalty(self.evaluate_penalty)
        self.optimizer.setConstraints(self.evaluate_constraints)
        return

    def _evaluate_surrogates(self, x):
        """ Evaluate all simulation surrogates.

        Args:
            x (ndarray): A 1D array containing the (embedded) design point to
                evaluate.

        Returns:
            ndarray: A 1D array containing the (packed) result of the
            surrogate model evaluations.

        """

        sx_list = [self.empty]
        for surrogate in self.surrogates:
            sx_list.append(surrogate.evaluate(x))
        return jnp.concatenate(sx_list, axis=None)

    def _surrogate_uncertainty(self, x):
        """ Evaluate the standard deviation of the possible surrogate outputs.

        Args:
            x (ndarray): A 1D array containing the (embedded) design point to
                evaluate the surrogate uncertainties at.

        Returns:
            ndarray: A 1D array containing the standard deviation of the
            surrogate prediction at x.

        """

        sdx_list = [self.empty]
        for surrogate in self.surrogates:
            sdx_list.append(surrogate.stdDev(x))
        return jnp.concatenate(sdx_list, axis=None)

    def _evaluate_objectives(self, x, sx):
        """ Evaluate all objectives from the latent space.

        Args:
            x (ndarray): A 1D array containing the (embedded) design point to
                evaluate.

            sx (ndarray): A 1D array containing the (packed) simulation vector
                at x.

        Returns:
            ndarray: A 1D array containing the result of the evaluation.

        """

        xx = self.extract(x)
        ssx = self.unpack_sim(sx)
        return self.vobj_funcs(xx, ssx)

    def _obj_fwd(self, x, sx):
        """ Evaluate a forward pass over the objective functions.
    
        Args:
            x (ndarray): A 1D array containing the (embedded) design point to
                evaluate.
    
            sx (ndarray): A 1D array containing the (packed) simulation vector
                at x.
    
        Returns:
            (ndarray, (ndarray, ndarray)): The first entry is a 1D array
            containing the result of the evaluation, and the second entry
            contains the extracted pair (xx, ssx).
    
        """
    
        xx = self.extract(x)
        ssx = self.unpack_sim(sx)
        return self.vobj_funcs(xx, ssx), (xx, ssx)

    def _obj_bwd(self, res, w):
        """ Evaluate a backward pass over the objective functions.
    
        Args:
            res (tuple of ndarrays): Contains extracted value of x and the
                unpacked value of sx computed during the forward pass.
    
            w (ndarray): Contains the adjoint vector for the computation
                succeeding the objective evaluation in the compute graph.
    
        Returns:
            (ndarray, ndarray): A pair of 1D arrays containing the products
            w * jac(f wrt x) and w * jac(f wrt s), respectively.
    
        """
    
        xx, ssx = res
        return self.vobj_grads(xx, ssx, w)

    def _evaluate_constraints(self, x, sx):
        """ Evaluate the constraints from the latent space.

        Args:
            x (ndarray): A 1D array containing the (embedded) design point to
                evaluate.

            sx (ndarray): A 1D array containing the (packed) simulation vector
                at x.

        Returns:
            ndarray: A 1D array containing the list of constraint violations
            at x, where a negative or zero score implies feasibility.

        """

        xx = self.extract(x)
        ssx = self.unpack_sim(sx)
        return self.vcon_funcs(xx, ssx)

    def _con_fwd(self, x, sx):
        """ Evaluate a forward pass over the constraint functions.

        Args:
            x (ndarray): A 1D array containing the (embedded) design point to
                evaluate.

            sx (ndarray): A 1D array containing the (packed) simulation vector
                at x.

        Returns:
            (ndarray, (ndarray, ndarray)): The first entry is a 1D array
            containing the constraint violations at x, and the second entry
            contains the extracted pair (xx, ssx).

        """

        xx = self.extract(x)
        ssx = self.unpack_sim(sx)
        return self.vcon_funcs(xx, ssx), (xx, ssx)

    def _con_bwd(self, res, w):
        """ Evaluate a backward pass over the constraint functions.

        Args:
            res (tuple of ndarrays): Contains extracted value of x and the
                unpacked value of sx computed during the forward pass.

            w (ndarray): Contains the adjoint vector for the computation
                succeeding the constraint evaluation in the compute graph.

        Returns:
            (ndarray, ndarray): A pair of 1D arrays containing the products
            w * jac(c wrt x) and w * jac(c wrt s), respectively.

        """

        xx, ssx = res
        dcdx, dcds = jnp.zeros(self.n_latent), jnp.zeros(self.m)
        for i, con_grad in enumerate(self.con_grads):
            x_grad, s_grad = con_grad(xx, ssx)
            dcdx += self.embed(x_grad) * w[i]
            dcds += self.pack_sim(s_grad) * w[i]
        return dcdx, dcds

    def _evaluate_penalty(self, x, sx):
        """ Evaluate the penalized objective from the latent space.

        Args:
            x (ndarray): A 1D array containing the (embedded) design point to
                evaluate.

            sx (ndarray): A 1d array containing the (packed) simulation vector
                at x.

        Returns:
            ndarray: A 1D array containing the result of the objective
            evaluation with a penalty added for violated constraints.

        """

        xx = self.extract(x)
        ssx = self.unpack_sim(sx)
        cx = jnp.sum(jnp.maximum(self.vcon_funcs(xx, ssx), 0.0))
        return self.vpen_funcs(xx, ssx, cx, self.lam)

    def _pen_fwd(self, x, sx):
        """ Evaluate a forward pass over the penalized objective functions.
    
        Args:
            x (ndarray): A 1D array containing the (embedded) design point
                to evaluate.
    
            sx (ndarray): A 1D array containing the (packed) simulation
                vector at x.
    
        Returns:
            (ndarray, tuple): The first entry is a 1D array containing the
            result of the evaluation, and the second entry contains the tuple
            (xx, ssx, activities) where xx and ssx are the extracted values of
            x and sx, and "activities" gives the active constraint penalties.
    
        """
    
        xx = self.extract(x)
        ssx = self.unpack_sim(sx)
        cx = jnp.maximum(self.vcon_funcs(xx, ssx), 0.0)
        act = (jnp.isclose(cx, jnp.zeros(cx.shape)) - 1) * -self.lam
        return self.vpen_funcs(xx, ssx, jnp.sum(cx), self.lam), (xx, ssx, act)

    def _pen_bwd(self, res, w):
        """ Evaluate a backward pass over the penalized objective functions.
    
        Args:
            res (tuple of ndarrays): Contains extracted value of x and the
                unpacked value of sx computed during the forward pass followed
                by a vector encoding the indicies/penalties for the active
                constraints.
    
            w (ndarray): Contains the adjoint vector for the computation
                succeeding the penalty evaluation in the compute graph.
    
        Returns:
            (ndarray, ndarray): A pair of 1D arrays containing the products
            w * jac(c wrt x) and w * jac(c wrt s), respectively.
    
        """

        xx, ssx, act = res
        dcdx, dcds = self._con_bwd((xx, ssx), act)
        dfdx = dcdx * jnp.sum(w)
        dfds = dcds * jnp.sum(w)
        for i, obj_grad in enumerate(self.obj_grads):
            x_grad, s_grad = obj_grad(xx, ssx)
            dfdx += self.embed(x_grad) * w[i]
            dfds += self.pack_sim(s_grad) * w[i]
        return dfdx, dfds

    def _compile(self):
        """ Compile the helper functions and link the fwd/bwd pass functions """

        try:
            self.evaluate_surrogates = jax.jit(self._evaluate_surrogates)
        except BaseException:
            self.evaluate_surrogates = self._evaluate_surrogates
            logging.info("      WARNING: MOOP._evaluate_surrogates"
                         "failed to jit...")

        try:
            self.surrogate_uncertainty = jax.jit(self._surrogate_uncertainty)
        except BaseException:
            self.surrogate_uncertainty = self._surrogate_uncertainty
            logging.info("      WARNING: MOOP._surrogate_uncertainty"
                         "failed to jit...")

        @jax.custom_vjp
        def eval_obj(x, sx): return self._evaluate_objectives(x, sx)
        def obj_fwd(x, sx): return self._obj_fwd(x, sx)
        eval_obj.defvjp(obj_fwd, self.obj_bwd)
        self.evaluate_objectives = eval_obj
        @jax.custom_vjp
        def eval_con(x, sx): return self._evaluate_constraints(x, sx)
        def con_fwd(x, sx): return self._con_fwd(x, sx)
        eval_con.defvjp(con_fwd, self.con_bwd)
        self.evaluate_constraints = eval_con
        @jax.custom_vjp
        def eval_pen(x, sx): return self._evaluate_penalty(x, sx)
        def pen_fwd(x, sx): return self._pen_fwd(x, sx)
        eval_pen.defvjp(pen_fwd, self.pen_bwd)
        self.evaluate_penalty = eval_pen
