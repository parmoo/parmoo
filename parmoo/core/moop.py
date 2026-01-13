
""" Contains the MOOP class for defining multiobjective optimization problems.

``parmoo.moop.MOOP`` is the serial implementation of the MOOP_base class for
defining and solving multiobjective optimization problems (MOOPs). Each MOOP
object may contain several simulations, specified using dictionaries.

"""

import inspect
from jax import numpy as jnp
import logging
import numpy as np
import warnings

from parmoo.acquisitions.acquisition_function import AcquisitionFunction
from parmoo.core.moop_base import MOOP_base
from parmoo.core.moop_checks import check_sims
from parmoo.databases import NumpyDatabase
from parmoo.embeddings.default_embedders import ContinuousEmbedder,  \
                                                IntegerEmbedder,     \
                                                CategoricalEmbedder, \
                                                IdentityEmbedder
from parmoo.optimizers.surrogate_optimizer import SurrogateOptimizer
from parmoo.utilities.error_checks import check_names


class MOOP(MOOP_base):
    """ Class for defining a multiobjective optimization problem (MOOP).

    Upon initialization, supply a scalar optimization procedure and
    dictionary of hyperparameters using the default constructor:
     * ``MOOP.__init__(ScalarOpt, [hyperparams={}])``

    New: To fix the random seed, use the hyperparameter key "np_random_gen"
    and set either an int or ``numpy.random.Generator`` instance
    as the corresponding value.

    Class methods are summarized below.  Several of these methods may be fully
    or partially inherited from the ``parmoo.moop_base.MOOP_base`` super class.

    To define the MOOP, add each design variable, simulation, objective, and
    constraint by using the following functions:
     * ``MOOP.addDesign(*args)``
     * ``MOOP.addSimulation(*args)``
     * ``MOOP.addObjective(*args)``
     * ``MOOP.addConstraint(*args)``

    Next, define your solver.

    Acquisition functions (used for scalarizing problems/setting targets) are
    added using:
     * ``MOOP.addAcquisition(*args)``

    When you are done defining a MOOP, it can be "compiled" to finalize
    the definition:
     * ``MOOP.compile()``

    After creating a MOOP, the following methods may be useful for getting
    the numpy.dtype of the input/output arrays:
     * ``MOOP.getDesignType()``
     * ``MOOP.getSimulationType()``
     * ``MOOP.getObjectiveType()``
     * ``MOOP.getConstraintType()``

    To turn on checkpointing use:
     * ``MOOP.setCheckpoint(checkpoint, [filename="parmoo"])``

    ParMOO also offers logging. To turn on logging, activate INFO-level
    logging by importing Python's built-in logging module.

    If there is any pre-existing simulation data, it can be added by
    calling the following method, where (x, sx) are the design, output
    pair for the simulation "s_name":
     * ``MOOP.updateSimDb(x, sx, s_name)``

    After defining the MOOP and setting up checkpointing and logging info,
    use the following method to solve the MOOP (serially):
     * ``MOOP.solve(iter_max=None, sim_max=None)``

    The following methods are used for solving the MOOP and managing the
    internal simulation/objective databases:
     * ``MOOP.checkSimDb(x, s_name)``
     * ``MOOP.evaluateSimulation(x, s_name)``
     * ``MOOP.addObjData(x, sx)``
     * ``MOOP.iterate(k, ib=None)``
     * ``MOOP.filterBatch(*args)``
     * ``MOOP.updateAll(k, batch)``

    Finally, the following methods are used to retrieve data after the
    problem has been solved:
     * ``MOOP.getPF(format='ndarray')``
     * ``MOOP.getSimulationData(format='ndarray')``
     * ``MOOP.getObjectiveData(format='ndarray')``

    The following methods are used to save/load the current checkpoint (state):
     * ``MOOP.save([filename="parmoo"])``
     * ``MOOP.load([filename="parmoo"])``

    """

    __slots__ = [
                 # Problem dimensions
                 'm', 'm_list', 'n_embed', 'n_feature', 'n_latent',
                 'o', 'p', 's',
                 # Tolerances and bounds
                 'feature_des_tols', 'latent_des_tols', 'cont_var_inds',
                 'latent_lb', 'latent_ub',
                 # Schemas
                 'des_schema', 'sim_schema', 'obj_schema', 'con_schema',
                 # Constants, counters, and adaptive parameters
                 'compiled', 'empty', 'epsilon', 'iteration', 'penalty_param',
                 # Checkpointing markers
                 'checkpoint', 'checkpoint_file',
                 'new_checkpoint', 'new_data',
                 # Design variables, simulations, objectives, and constraints
                 'embedders', 'emb_hp', 'sim_funcs',
                 'obj_funcs', 'obj_grads', 'con_funcs', 'con_grads',
                 # Solver components
                 'acquisitions', 'searches', 'surrogates', 'optimizer',
                 # Database information
                 'database',
                 # Temporary solver components and metadata used during setup
                 'acq_tmp', 'opt_tmp', 'search_tmp', 'sur_tmp',
                 'acq_hp', 'opt_hp', 'sim_hp',
                 # Random generator object with state information
                 'np_random_gen',
                 # Compiled function definitions -- These are only defined
                 # after calling the MOOP.compile() method
                 'obj_bwd', 'con_bwd', 'pen_bwd'
                ]

    def __init__(self, opt_func, hyperparams=None):
        """ Initializer for the MOOP class.

        Args:
            opt_func (SurrogateOptimizer): A solver for the surrogate problems.

            hyperparams (dict, optional): A dictionary of hyperparameters for
                the opt_func, and any other procedures that will be used.

        """

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
                 * 'embedder' (parmoo.embeddings.embedder.Embedder): When
                   des_type is 'custom', this is a custom Embedder class, which
                   maps the input to a point in the unit hypercube and reports
                   the embedded dimension.

        """

        for arg in args:
            # Check arg and optional inputs for correct types
            if not isinstance(arg, dict):
                raise TypeError("Each argument must be a Python dict")
            if 'name' in arg:
                name = arg['name']
            else:
                name = f"x{len(self.des_schema) + 1}"
            check_names(name, self.des_schema, self.sim_schema,
                        self.obj_schema, self.con_schema)
            arg['name'] = name
            # Append each design variable (default) to the schema
            if 'des_type' in arg:
                if not isinstance(arg['des_type'], str):
                    raise TypeError("args['des_type'] must be a str")
            if (
                'des_type' not in arg or
                arg['des_type'] in ["continuous", "cont", "real"]
            ):
                arg['embedder'] = ContinuousEmbedder(arg)
            elif arg['des_type'] in ["integer", "int"]:
                arg['embedder'] = IntegerEmbedder(arg)
            elif arg['des_type'] in ["categorical", "cat"]:
                arg['embedder'] = CategoricalEmbedder(arg)
            elif arg['des_type'] in ["custom"]:
                if 'embedder' not in arg:
                    raise AttributeError(
                        "For a custom embedder, the 'embedder' key must be"
                        " present."
                    )
                arg1 = {}
                for key in arg:
                    if key != 'embedder':
                        arg1[key] = arg[key]
                arg1['np_random_gen'] = self.np_random_gen
                try:
                    arg['embedder'] = arg['embedder'](arg1)
                except BaseException:
                    raise TypeError(
                        "When present, the 'embedder' key must contain a"
                        " parmoo.embeddings.embedder.Embedder class."
                    )
            elif arg['des_type'] in ["raw"]:
                arg['embedder'] = IdentityEmbedder(arg)
            else:
                raise ValueError(
                    f"des_type={arg['des_type']} is not a recognized value"
                )
            # Add the design variable
            super().addDesign(arg)

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
            if 'name' in arg:
                name = arg['name']
            else:
                name = f"sim{self.s + 1}"
            check_names(name, self.des_schema, self.sim_schema,
                        self.obj_schema, self.con_schema)
            arg['name'] = name
            if 'hyperparams' not in arg:
                arg['hyperparams'] = {}
            super().addSimulation(arg)

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
            arg['name'] = name
            # Finally, if all else passed, add the objective
            super().addObjective(arg)

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
            arg['name'] = name
            # Finally, if all else passed, add the constraint
            super().addConstraint(arg)

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
            else:
                arg['hyperparams'] = {}
            try:
                acq = arg['acquisition'](1, np.zeros(1), np.ones(1), {})
                if not isinstance(acq, AcquisitionFunction):
                    raise TypeError(
                        "'acquisition' must specify a child of the "
                        "AcquisitionFunction class"
                    )
            except BaseException:
                raise TypeError("'acquisition' must specify a child of the"
                                + " AcquisitionFunction class")
            # If all checks passed, add the acquisition to the list
            super().addAcquisition(arg)

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
             * The first entry in each tuple is a Python dictionary
               specifying the design point to evaluate.
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
            # Compile the MOOP and start the database if needed
            if not self.compiled:
                self.compile()
            # Generate search data
            for j, search in enumerate(self.searches):
                des = search.startSearch(np.asarray(self.latent_lb),
                                         np.asarray(self.latent_ub))
                for xi in des:
                    xbatch.append((self._extract(xi),
                                   self.sim_schema[j][0]))
        # General case for k>0 iterations
        else:
            # TODO(thchang): Remove this expensive query
            obj_data = self.database.getObjectiveData()
            xx = np.zeros((len(obj_data), self.n_latent))
            fxx = np.zeros((len(obj_data), self.o))
            cxx = np.zeros((len(obj_data), self.p))
            for i, datum in enumerate(obj_data):
                xx[i, :] = self._embed(datum)
                for j, dt in enumerate(self.obj_schema):
                    fxx[i, j] = datum[dt[0]]
                for j, dt in enumerate(self.con_schema):
                    cxx[i, j] = datum[dt[0]]
            # Set acquisition function targets
            x0 = np.zeros((len(self.acquisitions), self.n_latent))
            for i, acqi in enumerate(self.acquisitions):
                x0[i, :] = acqi.setTarget(
                    {'x_vals': xx, 'f_vals': fxx, 'c_vals': cxx},
                    self._evaluate_penalty
                )
            # Solve the surrogate problem
            x_candidates = self.optimizer.solve(x0)
            # Create a batch for filtering methods
            for i, acqi in enumerate(self.acquisitions):
                if self.s > 0:
                    for sn in self.sim_schema:
                        xbatch.append((self._extract(x_candidates[i, :]),
                                       sn[0]))
                else:
                    xbatch.append(self._extract(x_candidates[i, :]))
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
                    xxi = self._embed(xi)
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
                            ibatch = [self._embed(self._extract(xk))
                                      for xk in x_improv]
                            while (any([any([np.all(np.abs(xj - xk) < des_tols)
                                             and namei == j for (xj, j)
                                             in ebatch])
                                        for xk in ibatch]) or
                                   any([self.checkSimDb(self._extract(xk),
                                                        namei)
                                        is not None for xk in ibatch])):
                                x_improv = self.surrogates[i].improve(xxi,
                                                                      True)
                                ibatch = [self._embed(self._extract(xk))
                                          for xk in x_improv]
                            # Add improvement points to the fbatch
                            for xj in ibatch:
                                fbatch.append((self._extract(xj), namei))
                                ebatch.append((xj, namei))
            else:
                # If there were no simulations, just add all points to fbatch
                des_tols = np.asarray(self.latent_des_tols)
                for xi in xbatch:
                    # This 2nd extract/embed, while redundant, is necessary
                    # for categorical variables to be processed correctly
                    xxi = self._embed(xi)
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
                for x, sx in self.database.browseCompleteSimulations():
                    fx = {}
                    for i, obj_func in enumerate(self.obj_funcs):
                        fx[self.obj_schema[i][0]] = obj_func(x, sx)
                    cx = {}
                    for i, con_func in enumerate(self.con_funcs):
                        cx[self.con_schema[i][0]] = con_func(x, sx)
                    if self.database.checkObjDb(x) is None:
                        self.database.updateObjDb(x, fx, cx)
        else:
            # If any constraints are violated, increase the penalty parameter
            # toward the limit
            for (xi, i) in batch:
                xxi = self._embed(xi)
                sxi = self._evaluate_surrogates(xxi)
                eps = np.sqrt(self.epsilon)
                if np.any(self._evaluate_constraints(xxi, sxi) > eps):
                    self.penalty_param = min(1e4, self.penalty_param * 2.0)
                    break
            # Update the surrogate models and objective database
            self._update_surrogates()
            for xi in batch:
                (x, i) = xi
                if self.database.checkObjDb(x) is None:
                    sx = {}
                    xx = self._embed(x)
                    sxx = np.zeros(self.m)
                    m_count = 0
                    # Check for xi in every sim_db
                    for j in range(self.s):
                        sim_namej = self.sim_schema[j][0]
                        sx[sim_namej] = self.database.checkSimDb(x, sim_namej)
                        if sx[sim_namej] is None:
                            sx = None
                            break
                        else:
                            sxx[m_count:m_count + self.m_list[j]] = \
                                sx[sim_namej][:]
                            m_count = m_count + self.m_list[j]
                    # If xi was in every sim_db, add it to the database and
                    # report to the optimizer
                    if sx is not None:
                        fx = {}
                        fxx = np.zeros(self.o)
                        for i, obj_func in enumerate(self.obj_funcs):
                            fxx[i:i+1] = obj_func(x, sx)
                            fx[self.obj_schema[i][0]] = fxx[i]
                        cx = {}
                        cxx = np.zeros(self.p)
                        for i, con_func in enumerate(self.con_funcs):
                            cxx[i:i+1] = con_func(x, sx)
                            cx[self.con_schema[i][0]] = cxx[i]
                        self.database.updateObjDb(x, fx, cx)
                        self.optimizer.returnResults(
                            xx, fxx, sxx, np.zeros(self.m)
                        )
        # If checkpointing is on, save the moop before continuing
        if self.checkpoint:
            self.save(filename=self.checkpoint_file)
        return

    def solve(self, iter_max=None, sim_max=None):
        """ Solve a MOOP using ParMOO.

        If desired, be sure to turn on checkpointing before starting the
        solve, using:

        ``MOOP.setCheckpoint(checkpoint, [filename="parmoo"])``

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
