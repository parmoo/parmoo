
""" Contains the MOOP_base class defining multiobjective optimization problems.

``parmoo.moop.MOOP_base`` is the base class for defining and solving
multiobjective optimization problems (MOOPs). Each MOOP object may contain
several simulations, specified using dictionaries.  This class defines all
private utilities and several setter/getter methods.  The methods used for
actually solving the MOOP must be defined in another instantiation of this
class.

"""

from abc import ABC, abstractmethod
import codecs
from importlib import import_module
import json
import logging
from os.path import exists as file_exists
import pickle
import shutil
import warnings

import jax
from jax import numpy as jnp
import numpy as np

from parmoo.embeddings.embedder import Embedder
from parmoo.databases import NumpyDatabase
from parmoo.optimizers.surrogate_optimizer import SurrogateOptimizer
from parmoo.utilities.error_checks import gradient_error


class MOOP_base(ABC):
    """ ABC for defining a multiobjective optimization problem (MOOP).

    The following public methods are defined herein, but may need to be
    extended by an implementation class:

    Setters and problem definition:
     * ``MOOP.addDesign(*args)``
     * ``MOOP.addSimulation(*args)``
     * ``MOOP.addObjective(*args)``
     * ``MOOP.addConstraint(*args)``
     * ``MOOP.addAcquisition(*args)``
     * ``MOOP.compile()``

    Getters:
     * ``MOOP.getDesignType()``
     * ``MOOP.getSimulationType()``
     * ``MOOP.getObjectiveType()``
     * ``MOOP.getConstraintType()``

    Database lookups and evaluator utilities:
     * ``MOOP.updateSimDb(x, sx, s_name)``
     * ``MOOP.checkSimDb(x, s_name)``
     * ``MOOP.getPF(format='ndarray')``
     * ``MOOP.getSimulationData(format='ndarray')``
     * ``MOOP.getObjectiveData(format='ndarray')``
     * ``MOOP.evaluateSimulation(x, s_name)``
     * ``MOOP.addObjData(x, sx)``

    Checkpointing methods:
     * ``MOOP.setCheckpoint(checkpoint, [filename="parmoo"])``
     * ``MOOP.save([filename="parmoo"])``
     * ``MOOP.load([filename="parmoo"])``

   The following solver steps must be defined in another implementation class
   (subclass) of this:
     * ``MOOP.iterate(k, ib=None)``
     * ``MOOP.filterBatch(*args)``
     * ``MOOP.updateAll(k, batch)``
     * ``MOOP.solve(iter_max=None, sim_max=None)``

    The following private methods are also implemented herein:
     * ``MOOP._embed(x)``
     * ``MOOP._extract(x)``
     * ``MOOP._embed_grads(x)``
     * ``MOOP._pack_sim(sx)``
     * ``MOOP._unpack_sim(sx)``
     * ``MOOP._vobj_funcs(x, sx)``
     * ``MOOP._vcon_funcs(x, sx)``
     * ``MOOP._vpen_funcs(x, sx, cx)``
     * ``MOOP._fit_surrogates()``
     * ``MOOP._update_surrogates()``
     * ``MOOP._set_surrogate_tr(center, radius)``
     * ``MOOP._evaluate_surrogates(x)``
     * ``MOOP._surrogate_uncertainty(x)``
     * ``MOOP._evaluate_objectives(x, sx)``
     * ``MOOP._obj_fwd(x, sx)``
     * ``MOOP._obj_bwd(res, w)``
     * ``MOOP._evaluate_constraints(x, sx)``
     * ``MOOP._con_fwd(x, sx)``
     * ``MOOP._con_bwd(res, w)``
     * ``MOOP._evaluate_penalty(x, sx)``
     * ``MOOP._pen_fwd(x, sx)``
     * ``MOOP._pen_bwd(res, w)``

    """

    __slots__ = [
        # Problem dimensions
        'm', 'm_list', 'n_embed', 'n_feature', 'n_latent', 'o', 'p', 's',
        # Tolerances and bounds
        'feature_des_tols', 'latent_des_tols', 'cont_var_inds', 'latent_lb',
        'latent_ub',
        # Schemas
        'des_schema', 'sim_schema', 'obj_schema', 'con_schema',
        # Constants, counters, and adaptive parameters
        'compiled', 'empty', 'epsilon', 'iteration', 'penalty_param',
        # Checkpointing markers
        'checkpoint', 'checkpoint_file', 'new_checkpoint', 'new_data',
        # Design variables, simulations, objectives, and constraints
        'embedders', 'emb_hp', 'sim_funcs', 'obj_funcs', 'obj_grads',
        'con_funcs', 'con_grads',
        # Solver components
        'acquisitions', 'searches', 'surrogates', 'optimizer',
        # Database information
        'database',
        # Temporary solver components and metadata used during setup
        'acq_tmp', 'opt_tmp', 'search_tmp', 'sur_tmp', 'acq_hp', 'opt_hp',
        'sim_hp',
        # Random generator object with state information
        'np_random_gen',
        # Compiled function definitions -- These are only defined after calling
        # the MOOP.compile() method
        'obj_bwd', 'con_bwd', 'pen_bwd'
    ]

    def __init__(self, opt_func, hyperparams=None):
        """ Initializer for the MOOP class.

        Args:
            opt_func (SurrogateOptimizer): A solver for the surrogate problems.

            hyperparams (dict, optional): A dictionary of hyperparameters for
                the opt_func, and any other procedures that will be used.

        """

        # Configure jax to use only CPUs
        jax.config.update('jax_platform_name', 'cpu')
        # Initialize the problem dimensions
        self.m = 0
        self.m_list, self.n_embed = [], []
        self.n_feature, self.n_latent = 0, 0
        self.o, self.p, self.s = 0, 0, 0
        # Initialize the bounds and tolerances
        self.feature_des_tols, self.latent_des_tols = [], []
        self.cont_var_inds = []
        self.latent_lb, self.latent_ub = [], []
        # Initialize the schemas
        self.des_schema, self.sim_schema = [], []
        self.obj_schema, self.con_schema = [], []
        # Initialize the constants, counters, and adaptive parameters
        self.compiled = False
        self.empty = jnp.zeros(0)
        self.epsilon = jnp.sqrt(jnp.finfo(jnp.ones(1).dtype).eps)
        self.iteration = 0
        self.penalty_param = 1.0
        # Initialize checkpointing markers
        self.checkpoint = False
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
        self.database = NumpyDatabase(hyperparams)
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
        if not isinstance(self.optimizer, SurrogateOptimizer):
            raise TypeError("opt_func must be a derivative of the "
                            "SurrogateOptimizer abstract class")
        self.opt_tmp = opt_func

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
                 * 'embedder' (parmoo.embeddings.embedder.Embedder): When
                   des_type is 'custom', this is a custom Embedder class, which
                   maps the input to a point in the unit hypercube and reports
                   the embedded dimension.

        """

        for arg in args:
            arg1 = {}
            for key in arg:
                if key != 'embedder':
                    arg1[key] = arg[key]
            arg1['np_random_gen'] = self.np_random_gen
            embedder = arg['embedder']
            if not isinstance(embedder, Embedder):
                raise TypeError(
                    "The 'embedder' key must contain an instance of a"
                    " parmoo.embeddings.embedder.Embedder class."
                )
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
            if self.feature_des_tols[-1] >= 1.0e-16:
                self.cont_var_inds.append(self.n_feature - 1)
            des_tols = embedder.getLatentDesTols()
            for des_tol in des_tols:
                self.latent_des_tols.append(float(des_tol))
            # Update the schema and add the embedder to list
            dtype = embedder.getInputType()
            self.des_schema.append((arg['name'], dtype))
            self.embedders.append(embedder)
            self.emb_hp.append(arg1)  # This is saved for re-loading
            self.database.addDesign(
                arg['name'], dtype, self.feature_des_tols[-1]
            )

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

        for arg in args:
            # Update the schema and track the simulation output dimensions
            if arg['m'] > 1:
                self.sim_schema.append((arg['name'], 'f8', arg['m']))
            else:
                self.sim_schema.append((arg['name'], 'f8'))
            self.m_list.append(arg['m'])
            self.m += arg['m']
            self.s += 1
            # Initialize the hyperparameter dictionary
            hps = arg['hyperparams']
            hps["np_random_gen"] = self.np_random_gen
            # Add the simulation's search and surrogate techniques
            self.search_tmp.append(arg['search'])
            self.sur_tmp.append(arg['surrogate'])
            self.sim_hp.append(hps)
            # Add the simulation function
            self.sim_funcs.append(arg['sim_func'])
            self.database.addSimulation(arg['name'], arg['m'])

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
            self.obj_schema.append((arg['name'], 'f8'))
            self.obj_funcs.append(arg['obj_func'])
            if 'obj_grad' in arg:
                self.obj_grads.append(arg['obj_grad'])
            self.o += 1
            self.database.addObjective(arg['name'])

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
                   indicates that x is feasible, while
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
            self.con_schema.append((arg['name'], 'f8'))
            if 'con_func' in arg:
                self.con_funcs.append(arg['con_func'])
            else:
                self.con_funcs.append(arg['constraint'])
            if 'con_grad' in arg:
                self.con_grads.append(arg['con_grad'])
            self.p += 1
            self.database.addConstraint(arg['name'])

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
            hps = arg['hyperparams']
            hps["np_random_gen"] = self.np_random_gen
            self.acq_tmp.append(arg['acquisition'])
            self.acq_hp.append(hps)

    def compile(self):
        """ Compile the MOOP object and initialize its components.

        This locks the MOOP definition and jits all jit-able methods.

        This must be done *before* adding any simulation or objective data to
        the internal database.

        This cannot be done *after* simulation or objective data has been added
        to the internal database.

        """

        logging.info(" Compiling the MOOP object...")
        # Verify that the MOOP is in a valid state before compiling
        if self.n_feature <= 0:
            raise RuntimeError("Cannot compile a MOOP with no design "
                               "variables.")
        if self.o <= 0:
            raise RuntimeError("Cannot compile a MOOP with no objectives.")
        if len(self.acq_tmp) == 0:
            warnings.warn(
                "You are compiling a MOOP with no acquisition functions."
                " I'll let you do it for analysis purposes, but the solve()"
                " command won't work correctly until you recompile with"
                " one or more acquisition functions..."
            )
        logging.info("   Initializing MOOP solver's component objects...")
        # Reset the internal lists
        self.searches, self.surrogates = [], []
        self.acquisitions = []
        self.optimizer = None
        # Pre-create numpy arrays for initialization
        lbs = np.asarray(self.latent_lb)
        ubs = np.asarray(self.latent_ub)
        des_tols = np.asarray(self.latent_des_tols)
        # Try jitting ParMOO embedders and extractors
        self._jit_all((lbs + ubs) / 2, np.zeros(self.m))
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
        # Start the Database
        self.database.startDatabase()
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

    def getDesignType(self):
        """ Get the numpy dtype of all design points for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's design points.
            If no design variables have yet been added, returns None.

        """

        return self.database.getDesignType()

    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's simulation outputs.
            If no simulations have been given, returns None.

        """

        return self.database.getSimulationType()

    def getObjectiveType(self):
        """ Get the numpy dtype of an objective point for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's objective points.
            If no objectives have yet been added, returns None.

        """

        return self.database.getObjectiveType()

    def getConstraintType(self):
        """ Get the numpy dtype of the constraint violations for this MOOP.

        Returns:
            dtype: The numpy dtype of this MOOP's constraint violation
            outputs. If no constraint functions have been given, returns None.

        """

        return self.database.getConstraintType()

    def checkSimDb(self, x, s_name):
        """ Check self.sim_db[s_name] to see if the design x was evaluated.

        Args:
            x (dict): A Python dictionary specifying the keys/names and
                corresponding values of a design point to search for.

            s_name (str): The name of the simulation whose database will be
                searched.

        Returns:
            None or numpy.ndarray: returns None if x is not in
            self.sim_db[s_name] (up to the design tolerance). Otherwise,
            returns the corresponding value of sx.

        """

        return self.database.checkSimDb(x, s_name)

    def updateSimDb(self, x, sx, s_name):
        """ Update sim_db[s_name] by adding a design/simulation output pair.

        Args:
            x (dict): A Python dictionary specifying the keys/names and
                corresponding values of a design point to add.

            sx (ndarray): A 1D array containing the corresponding
                simulation output(s).

            s_name (str): The name of the simulation to whose database the
                pair (x, sx) will be added into.

        """

        self.database.updateSimDb(x, sx, s_name)

    def evaluateSimulation(self, x, s_name):
        """ Evaluate sim_func[s_name] and store the result in the database.

        Args:
            x (dict): A Python dictionary with keys/names corresponding
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
            self.database.updateSimDb(x, sx, s_name)
            if self.checkpoint:
                self.save(filename=self.checkpoint_file)
        return sx

    def addObjData(self, x, sx):
        """ Update the internal objective database by truly evaluating x.

        Args:
            x (dict): A Python dictionary containing the value of the design
                variable to add to ParMOO's database.

            sx (dict): A Python dictionary containing the values of the
                corresponding simulation outputs for ALL simulations involved
                in this MOOP -- sx['s_name'][:] contains the output(s)
                for sim_func['s_name'].

        """

        if self.database.checkObjDb(x) is not None:
            return
        fx = {}
        for i, obj_func in enumerate(self.obj_funcs):
            fx[self.obj_schema[i][0]] = obj_func(x, sx)
        cx = {}
        for i, constraint_func in enumerate(self.con_funcs):
            cx[self.con_schema[i][0]] = constraint_func(x, sx)
        self.database.updateObjDb(x, fx, cx)

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

        return self.database.getPF(format)

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

        return self.database.getSimulationData(format)

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

        return self.database.getObjectiveData(format)

    @abstractmethod
    def iterate(self, k, ib=None):
        """ Perform an iteration of ParMOO's solver to generate candidates. """

    @abstractmethod
    def filterBatch(self, *args):
        """ Filter a batch produced by ParMOO's MOOP.iterate method. """

    @abstractmethod
    def updateAll(self, k, batch):
        """ Update all surrogates given a batch of freshly evaluated data. """

    @abstractmethod
    def solve(self, iter_max=None, sim_max=None):
        """ Solve a MOOP using ParMOO. """

    def setCheckpoint(self, checkpoint=True, filename="parmoo"):
        """ Activate ParMOO's checkpointing feature.

        Note that for checkpointing to work, all simulation, objective,
        and constraint functions must be defined in the global scope.
        ParMOO also cannot save lambda functions.

        Args:
            checkpoint (bool): Turn checkpointing on (True) or off (False).

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
        self.checkpoint_file = filename
        self.database.setCheckpoint(checkpoint, filename)

    def save(self, filename="parmoo"):
        """ Serialize and save the MOOP object and all of its dependencies.

        Args:
            filename (str, optional): The filepath to serialized
                checkpointing file(s). Do not include file extensions,
                they will be appended automatically. This method may create
                several additional save files with this same name, but
                different file extensions, in order to recursively save
                dependency objects (such as surrogate models). Defaults to
                the value "parmoo" (filename will be "parmoo.moop").

        """

        # Check whether the file exists first
        exists = file_exists(filename + ".moop")
        if exists and self.new_checkpoint:
            raise OSError(
                f"Creating a new checkpoint file, but {filename}.moop"
                " already exists! Move the existing file to a new location"
                " or delete it, so that ParMOO doesn't accidentally"
                " overwrite your data..."
            )
        # Create a serializable ParMOO dictionary
        parmoo_state = {
            'm': self.m,
            'm_list': self.m_list,
            'n_embed': self.n_embed,
            'n_feature': self.n_feature,
            'n_latent': self.n_latent,
            'o': self.o,
            'p': self.p,
            's': self.s,
            'feature_des_tols': self.feature_des_tols,
            'latent_des_tols': self.latent_des_tols,
            'cont_var_inds': self.cont_var_inds,
            'latent_lb': self.latent_lb,
            'latent_ub': self.latent_ub,
            'des_schema': self.des_schema,
            'sim_schema': self.sim_schema,
            'obj_schema': self.obj_schema,
            'con_schema': self.con_schema,
            'iteration': self.iteration,
            'penalty_param': self.penalty_param,
            'checkpoint': self.checkpoint,
            'checkpoint_file': self.checkpoint_file,
            'np_random_state':
            self.np_random_gen.bit_generator.state,
        }
        # Pickle and add a list of the model and solver hyperparameters
        parmoo_state['hyperparams'] = []
        for hpi in [self.emb_hp, self.acq_hp, self.opt_hp, self.sim_hp]:
            parmoo_state['hyperparams'].append(
                codecs.encode(pickle.dumps(hpi), "base64").decode()
            )
        # Add the names/modules for all components of the MOOP definition
        parmoo_state['embedders'] = [
            (embedder.__class__.__name__, embedder.__class__.__module__)
            for embedder in self.embedders
        ]
        parmoo_state['sim_funcs'] = []
        parmoo_state['sim_funcs_info'] = []
        for si in self.sim_funcs:
            if type(si).__name__ == "function":
                parmoo_state['sim_funcs'].append((si.__name__, si.__module__))
                parmoo_state['sim_funcs_info'].append("function")
            else:
                parmoo_state['sim_funcs'].append(
                    (si.__class__.__name__, si.__class__.__module__)
                )
                parmoo_state['sim_funcs_info'].append(
                    codecs.encode(pickle.dumps(si), "base64").decode()
                )
        parmoo_state['obj_funcs'] = []
        parmoo_state['obj_funcs_info'] = []
        for fi in self.obj_funcs:
            if type(fi).__name__ == "function":
                parmoo_state['obj_funcs'].append((fi.__name__, fi.__module__))
                parmoo_state['obj_funcs_info'].append("function")
            else:
                parmoo_state['obj_funcs'].append(
                    (fi.__class__.__name__, fi.__class__.__module__)
                )
                parmoo_state['obj_funcs_info'].append(
                        codecs.encode(pickle.dumps(fi), "base64").decode()
                )
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
        parmoo_state['optimizer'] = (
            self.optimizer.__class__.__name__,
            self.optimizer.__class__.__module__
        )
        parmoo_state['searches'] = [
            (search.__class__.__name__, search.__class__.__module__)
            for search in self.searches
        ]
        parmoo_state['surrogates'] = [
            (sur.__class__.__name__, sur.__class__.__module__)
            for sur in self.surrogates
        ]
        parmoo_state['acquisitions'] = [
            (acq.__class__.__name__, acq.__class__.__module__)
            for acq in self.acquisitions
        ]
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
        # Save the serialized ParMOO dictionary
        fname = filename + ".moop"
        fname_tmp = "." + fname + ".swap"
        with open(fname_tmp, 'w') as fp:
            json.dump(parmoo_state, fp)
        shutil.move(fname_tmp, fname)
        self.new_checkpoint = False

    def load(self, filename="parmoo"):
        """ Load a serialized MOOP object and all of its dependencies.

        Args:
            filename (str, optional): The filepath to the serialized
                checkpointing file(s). Do not include file extensions,
                they will be appended automatically. This method may also
                load from other save files with the same name, but different
                file extensions, in order to recursively load dependency
                objects (such as surrogate models) as needed.
                Defaults to the value "parmoo" (filename will be
                "parmoo.moop").

        """

        PYDOCS = (
            "https://docs.python.org/3/tutorial/modules.html"
            "#the-module-search-path"
        )

        # Load the serialized dictionary object
        fname = f"{filename}.moop"
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
        self.cont_var_inds = parmoo_state['cont_var_inds']
        self.latent_des_tols = parmoo_state['latent_des_tols']
        self.latent_lb = parmoo_state['latent_lb']
        self.latent_ub = parmoo_state['latent_ub']
        self.des_schema = parmoo_state['des_schema']
        self.sim_schema = parmoo_state['sim_schema']
        self.obj_schema = parmoo_state['obj_schema']
        self.con_schema = parmoo_state['con_schema']
        self.iteration = parmoo_state['iteration']
        self.penalty_param = parmoo_state['penalty_param']
        self.checkpoint = parmoo_state['checkpoint']
        self.checkpoint_file = parmoo_state['checkpoint_file']
        self.np_random_gen = np.random.default_rng()
        self.np_random_gen.bit_generator.state = \
            parmoo_state['np_random_state']
        # Recover the pickled hyperparameter dictionaries
        hps = []
        for i, hpi in enumerate(parmoo_state['hyperparams']):
            hps.append(pickle.loads(codecs.decode(hpi.encode(), "base64")))
            if i != 2:
                for j in range(len(hps[i])):
                    hps[i][j]['np_random_gen'] = self.np_random_gen
            else:
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
                raise ModuleNotFoundError(
                    f"module: {emb_mod} could not be loaded. Please make"
                    f" sure that {emb_mod} exists on this machine and"
                    f" is part of the module search path: {PYDOCS}"
                )
            try:
                new_emb = getattr(mod, emb_name)
            except KeyError:
                raise KeyError(
                    f"function: {emb_name} defined in {emb_mod} could not be"
                    f" loaded. Please make sure that {emb_name} is defined"
                    f" in {emb_mod} with global scope."
                )
            toadd = new_emb(self.emb_hp[i])
            self.embedders.append(toadd)
        self.sim_funcs = []
        for (sim_name, sim_mod), info in zip(parmoo_state['sim_funcs'],
                                             parmoo_state['sim_funcs_info']):
            try:
                mod = import_module(sim_mod)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"module: {sim_mod} could not be loaded. Please make"
                    f" sure that {sim_mod} exists on this machine and"
                    f" is part of the module search path: {PYDOCS}"
                 )
            try:
                sim_ptr = getattr(mod, sim_name)
            except KeyError:
                raise KeyError(
                    f"function: {sim_name} defined in {sim_mod} could not be"
                    f" loaded. Please make sure that {sim_name} is defined"
                    f" in {emb_mod} with global scope."
                )
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
        # Re-compile the MOOP and re-load the database
        self.compile()
        self.database.loadCheckpoint(filename)
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
        self.new_checkpoint = False

    def _embed(self, x):
        """ Embed a design input as a n-dimensional vector for ParMOO.

        Args:
            x (dict): A Python dictionary whose keys match the design
                variable names, and whose values contain design variable
                values.

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

    def _embed_grads(self, dx):
        """ Embed a design input as a n-dimensional vector for ParMOO.

        Args:
            dx (dict): A Python dictionary whose keys match the design
                variable names, and whose values contain the partials
                with respect to each of the design variables.

        Returns:
            ndarray: A 1D array of length n_latent containing the embedded
            design vector.

        """

        dxx = jnp.zeros(sum(self.n_embed))
        for i in self.cont_var_inds:
            istart = sum(self.n_embed[:i])
            iend = istart + self.n_embed[i]
            dxx = dxx.at[istart:iend].set(self.embedders[i].embed_grad(
                                          dx[self.des_schema[i][0]]))
        return dxx

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

        new_sim_db = self.database.getNewSimulationData()
        for i, dti in enumerate(self.sim_schema):
            sim_namei = dti[0]
            n = len(new_sim_db[sim_namei])
            x_vals = np.zeros((n, self.n_latent))
            for j, xj in enumerate(new_sim_db[sim_namei]):
                x_vals[j, :] = self._embed(xj)
            s_vals = np.reshape(
                new_sim_db[sim_namei]['out'], (n, self.m_list[i])
            )
            self.surrogates[i].fit(x_vals, s_vals)

    def _update_surrogates(self):
        """ Update the surrogate models using the current sim databases. """

        new_sim_db = self.database.getNewSimulationData()
        for i, dti in enumerate(self.sim_schema):
            sim_namei = dti[0]
            n = len(new_sim_db[sim_namei])
            x_vals = np.zeros((n, self.n_latent))
            for j, xj in enumerate(new_sim_db[sim_namei]):
                x_vals[j, :] = self._embed(xj)
            s_vals = np.reshape(
                new_sim_db[sim_namei]['out'], (n, self.m_list[i])
            )
            self.surrogates[i].update(x_vals, s_vals)

    def _set_surrogate_tr(self, center, radius):
        """ Alert the surrogate functions of a new trust region.

        Args:
            center (ndarray): A 1D array containing the (embedded) coordinates
                of the new trust region center.

            radius (ndarray or float): The trust region radius.

        """

        for surrogate in self.surrogates:
            surrogate.setTrustRegion(center, radius)
        eval_obj, eval_con, eval_pen = self._link()
        self.optimizer.setObjective(eval_obj)
        self.optimizer.setConstraints(eval_con)
        self.optimizer.setPenalty(eval_pen)

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

        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        return self._vobj_funcs(xx, ssx)

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

        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        return self._vobj_funcs(xx, ssx), (x, sx)

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

        x, sx = res
        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        dfdx, dfds = jnp.zeros(self.n_latent), jnp.zeros(self.m)
        for i, obj_grad in enumerate(self.obj_grads):
            x_grad, s_grad = obj_grad(xx, ssx)
            dfdx += self._embed_grads(x_grad) * w[i]
            dfds += self._pack_sim(s_grad) * w[i]
        return dfdx, dfds

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

        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        return self._vcon_funcs(xx, ssx)

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

        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        return self._vcon_funcs(xx, ssx), (x, sx)

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

        x, sx = res
        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        dcdx, dcds = jnp.zeros(self.n_latent), jnp.zeros(self.m)
        for i, con_grad in enumerate(self.con_grads):
            x_grad, s_grad = con_grad(xx, ssx)
            dcdx += self._embed_grads(x_grad) * w[i]
            dcds += self._pack_sim(s_grad) * w[i]
        return dcdx, dcds

    def _evaluate_penalty(self, x, sx):
        """ Evaluate the penalized objective from the latent space.

        Args:
            x (ndarray): A 1D array containing the (embedded) design point to
                evaluate.

            sx (ndarray): A 1D array containing the (packed) simulation vector
                at x.

        Returns:
            ndarray: A 1D array containing the result of the objective
            evaluation with a penalty added for violated constraints.

        """

        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        cx = jnp.sum(jnp.maximum(self._vcon_funcs(xx, ssx), 0.0))
        return self._vpen_funcs(xx, ssx, cx, self.penalty_param)

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

        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        cx = jnp.maximum(self._vcon_funcs(xx, ssx), 0.0)
        act = (jnp.isclose(cx, jnp.zeros(cx.shape)) - 1) * -self.penalty_param
        return (
            self._vpen_funcs(xx, ssx, jnp.sum(cx), self.penalty_param),
            (x, sx, act)
        )

    def _pen_bwd(self, res, w):
        """ Evaluate a backward pass over the penalized objective functions.

        Args:
            res (tuple of ndarrays): Contains extracted value of x and the
                unpacked value of sx computed during the forward pass followed
                by a vector encoding the indices/penalties for the active
                constraints.

            w (ndarray): Contains the adjoint vector for the computation
                succeeding the penalty evaluation in the compute graph.

        Returns:
            (ndarray, ndarray): A pair of 1D arrays containing the products
            w * jac(c wrt x) and w * jac(c wrt s), respectively.

        """

        x, sx, act = res
        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        dcdx, dcds = self._con_bwd((x, sx), act)
        dfdx = dcdx * jnp.sum(w)
        dfds = dcds * jnp.sum(w)
        for i, obj_grad in enumerate(self.obj_grads):
            x_grad, s_grad = obj_grad(xx, ssx)
            dfdx += self._embed_grads(x_grad) * w[i]
            dfds += self._pack_sim(s_grad) * w[i]
        return dfdx, dfds

    def _jit_all(self, xx, sx):
        """ Compile the MOOP object and initialize its components.

        Attempts to JIT all internal functions and logs any failures.

        Args:
            xx (numpy.ndarray): A sample design point to evaluate to trigger
                the JIT compile.
            xx (numpy.ndarray): A sample simulation to evaluate to trigger the
                JIT compile.

        """

        logging.info("   jitting and testing ParMOO's embedders...")
        try:
            x = jax.jit(self._extract)(xx)
            for key in self.des_schema:
                assert (key[0] in x)
        except BaseException:
            logging.info("     WARNING: 1 or more extractors failed to jit...")
        try:
            xx2 = jax.jit(self._embed)(x)
            assert (xx2.shape == xx.shape)
        except BaseException:
            logging.info("     WARNING: 1 or more embedders failed to jit...")
        try:
            xx2 = jax.jit(self._embed_grads)(x)
            assert (xx2.shape == xx.shape)
        except BaseException:
            logging.info(
                "     WARNING: 1 or more grad embedders failed to jit..."
            )
        try:
            sx = jax.jit(self._unpack_sim)(sx)
            for key in self.sim_schema:
                assert (key[0] in sx)
        except BaseException:
            logging.info("     WARNING: MOOP._unpack_sim failed to jit...")
        try:
            sx2 = jax.jit(self._pack_sim)(sx)
            assert (sx2.shape == sx.shape)
        except BaseException:
            logging.info("     WARNING: MOOP._pack_sim failed to jit...")
        logging.info("   Done.")
        # Jitting ParMOO objectives and constraints
        logging.info("   jitting ParMOO's objective and constraints...")
        try:
            _ = jax.jit(self._vobj_funcs)(x, sx)
        except BaseException:
            logging.info("     WARNING: 1 or more obj_funcs failed to jit...")
        try:
            _ = jax.jit(self._vcon_funcs)(x, sx)
        except BaseException:
            logging.info("     WARNING: 1 or more con_funcs failed to jit...")
        try:
            _ = jax.jit(self._vpen_funcs)(x, sx, 0., 1.)
        except BaseException:
            logging.info("     WARNING: MOOP._vpen_funcs failed to jit...")
        if len(self.obj_grads) == self.o:
            try:
                _, _ = jax.jit(self._obj_bwd)((xx, sx), jnp.zeros(self.o))
            except BaseException:
                logging.info(
                    "     WARNING: 1 or more obj_grads failed to jit..."
                )
            self.obj_bwd = self._obj_bwd
        else:
            self.obj_bwd = gradient_error
        if len(self.con_grads) == self.p:
            try:
                _, _ = jax.jit(self._con_bwd)((xx, sx), jnp.zeros(self.p))
            except BaseException:
                logging.info(
                    "     WARNING: 1 or more con_grads failed to jit..."
                )
            self.con_bwd = self._con_bwd
        else:
            self.con_bwd = gradient_error
        if len(self.obj_grads) == self.o and len(self.con_grads) == self.p:
            try:
                _, _ = jax.jit(self._pen_bwd)(
                    (xx, sx, jnp.zeros(self.p)), jnp.zeros(self.o)
                )
            except BaseException:
                logging.info("     WARNING: MOOP._pen_grads failed to jit...")
            self.pen_bwd = self._pen_bwd
        else:
            self.pen_bwd = gradient_error
        logging.info("   Done.")

    def _link(self):
        """ Link the forward/backward pass functions """

        @jax.custom_vjp
        def eval_obj(x, sx): return self._evaluate_objectives(x, sx)
        def obj_fwd(x, sx): return self._obj_fwd(x, sx)
        eval_obj.defvjp(obj_fwd, self.obj_bwd)
        @jax.custom_vjp
        def eval_con(x, sx): return self._evaluate_constraints(x, sx)
        def con_fwd(x, sx): return self._con_fwd(x, sx)
        eval_con.defvjp(con_fwd, self.con_bwd)
        @jax.custom_vjp
        def eval_pen(x, sx): return self._evaluate_penalty(x, sx)
        def pen_fwd(x, sx): return self._pen_fwd(x, sx)
        eval_pen.defvjp(pen_fwd, self.pen_bwd)
        return eval_obj, eval_con, eval_pen
