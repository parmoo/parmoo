
""" Contains the MOOP class for defining multiobjective optimization problems.

parmoo.moop.MOOP is the base class for defining and solving multiobjective
optimization problems (MOOPs). Each MOOP object may contain several
simulations, specified using dictionaries.

"""

import inspect
from jax import jacfwd, lax
from jax import numpy as jnp
import json
import numpy as np
import pandas as pd
from parmoo import structs
from parmoo.embeddings.default_embedders import *
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

    The following methods are used to save/load a ParMOO state to/from disk:
     * ``MOOP.save([filename="parmoo"])``
     * ``MOOP.load([filename="parmoo"])``

    To turn on checkpointing use:
     * ``MOOP.setCheckpoint(checkpoint, [checkpoint_data, filename])``

    ParMOO also offers logging. To turn on logging, activate INFO-level
    logging by importing Python's built-in logging module.

    After defining the MOOP and setting up checkpointing and logging info,
    use the following method to solve the MOOP (serially):
     * ``MOOP.solve(iter_max=None, sim_max=None)``

    The following methods are used for solving the MOOP and managing the
    internal simulation/objective databases:
     * ``MOOP.check_sim_db(x, s_name)``
     * ``MOOP.update_sim_db(x, sx, s_name)``
     * ``MOOP.evaluateSimulation(x, s_name)``
     * ``MOOP.addData(x, sx)``
     * ``MOOP.iterate(k, ib=None)``
     * ``MOOP.updateAll(k, batch)``

    Finally, the following methods are used to retrieve data after the
    problem has been solved:
     * ``MOOP.getPF(format='ndarray')``
     * ``MOOP.getSimulationData(format='ndarray')``
     * ``MOOP.getObjectiveData(format='ndarray')``

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

    """

    __slots__ = [
                 # Problem dimensions
                 'n_feature', 'n_latent',
                 'm', 'm_total', 'o', 'p', 's',
                 # Embedding dimensions, bounds, and tolerances
                 'embedders', 'embedding_size',
                 'latent_lb', 'latent_ub',
                 'latent_des_tols',
                 # Schemas and databases
                 'des_schema', 'sim_schema', 'obj_schema', 'con_schema',
                 'data', 'sim_db', 'n_dat', 'new_data',
                 # Constants, counters, and adaptive parameters
                 'empty', 'epsilon', 'iteration', 'lam',
                 # Checkpointing markers
                 'checkpoint', 'checkpoint_data', 'checkpoint_file',
                 'new_checkpoint',
                 # Simulations, objectives, constraints, and their metadata
                 'sim_funcs', 'obj_funcs', 'con_funcs',
                 # Solver components and their metadata
                 'acquisitions', 'searches', 'surrogates',
                 'optimizer', 'optimizer_obj',
                 'hyperparams', 'history',
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
        self.n_feature, self.n_latent = 0, 0
        self.m = []
        self.m_total, self.o, self.p, self.s = 0, 0, 0, 0
        # Initialize the embedding dimensions, bounds, and tolerances
        self.embedders, self.embedding_size = [], []
        self.latent_lb, self.latent_ub = [], []
        self.latent_des_tols = []
        # Initialize the schemas and databases
        self.des_schema, self.sim_schema = [], []
        self.obj_schema, self.con_schema = [], []
        self.data, self.sim_db = {}, []
        self.n_dat, self.new_data = 0, True
        # Initialize the constants, counters, and adaptive parameters
        self.empty = jnp.zeros(0)
        self.epsilon = jnp.sqrt(jnp.finfo(jnp.ones(1)).eps)
        self.iteration = 0
        self.lam = 1.0
        # Initialize checkpointing markers
        self.checkpoint, self.checkpoint_data = False, False
        self.checkpoint_file = "parmoo"
        self.new_checkpoint = True
        # Initialize simulations, objectives, constraints, and their metadata
        self.sim_funcs, self.obj_funcs, self.con_funcs = [], [], []
        # Initialize solver components and their metadata
        self.acquisitions, self.searches, self.surrogates = [], [], []
        self.optimizer, self.optimizer_obj = None, None
        self.hyperparams, self.history = {}, {}
        # Set up the surrogate optimizer and its hyperparameters
        try:
            self.optimizer_obj = opt_func(1,
                                          np.zeros(1),
                                          np.ones(1),
                                          self.hyperparams)
        except BaseException:
            raise TypeError("opt_func must be a derivative of the "
                            + "SurrogateOptimizer abstract class")
        if not isinstance(self.optimizer_obj, structs.SurrogateOptimizer):
            raise TypeError("opt_func must be a derivative of the "
                            + "SurrogateOptimizer abstract class")
        self.optimizer = opt_func
        if hyperparams is not None:
            if isinstance(hyperparams, dict):
                self.hyperparams = hyperparams
            else:
                raise TypeError("hyperparams must be a Python dict")
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
                 * 'name' (str, optional): The unique name of this design
                   variable, which ultimately serves as its primary key in
                   all of ParMOO's databases. This is also how users should
                   index this variable in all user-defined functions passed
                   to ParMOO.
                   If left blank, it is assigned an integer index by default
                   corresponding to the order added.
                 * 'des_type' (str): The type for this design variable.
                   Currently supported options are:
                    * 'continuous' (or 'cont' or 'real')
                    * 'categorical' (or 'cat')
                    * 'integer' (or 'int')
                    * 'raw' -- for advanced use only, not recommended
                    * 'custom'
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
                   default value is epsilon * max(ub - lb, 1.0e-4).
                 * 'levels' (int or list): When des_type is 'categorical', this
                   specifies the number of levels for the variable (when int)
                   or the names of each valid category (when a list).
                 * 'embedder': When des_type is 'custom', this is a custom
                   embedding function, which maps the input to a point in the
                   unit hypercube of dimension 'embedding_size'.

        """

        from parmoo.util import check_names

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
            if 'des_type' in arg:
                if not isinstance(arg['des_type'], str):
                    raise TypeError("args['des_type'] must be a str")
            # Append each design variable (default) to the schema
            if 'des_type' not in arg or \
               arg['des_type'] in ["continuous", "cont", "real"]:
                embedder = ContinuousEmbedder(arg)
            elif arg['des_type'] in ["integer", "int"]:
                embedder = IntegerEmbedder(arg)
            elif arg['des_type'] in ["categorical", "cat"]:
                embedder = CategoricalEmbedder(arg)
            elif arg['des_type'] in ["custom"]:
                embedder = arg['embedder'](arg)
            elif arg['des_type'] in ["raw"]:
                embedder = IdentityEmbedder(arg)
            else:
                raise ValueError("des_type=" + arg['des_type'] +
                                 " is not a recognized value")
            # Collect the metadata for this embedding
            self.n_feature += 1
            self.embedding_size.append(embedder.getEmbeddingSize())
            self.n_latent += self.embedding_size[-1]
            lbs = embedder.getLowerBounds()
            for lb in lbs:
                self.latent_lb.append(lb)
            ubs = embedder.getUpperBounds()
            for ub in ubs:
                self.latent_ub.append(ub)
            des_tols = embedder.getDesTols()
            for des_tol in des_tols:
                self.latent_des_tols.append(float(des_tol))
            dtype = embedder.getInputType()
            self.embedders.append(embedder)
            # Update the schema
            if 'name' in arg:
                name = arg['name']
            else:
                name = len(self.des_schema)
            check_names(name, self.des_schema, self.sim_schema,
                        self.obj_schema, self.con_schema)
            self.des_schema.append((name, dtype))
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

        from parmoo.util import check_sims, check_names

        # Assert proper order of problem definition
        if len(self.obj_funcs) > 0 or len(self.con_funcs) > 0:
            raise RuntimeError("Cannot add more simulations after"
                               " adding objectives and/or constraints")
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
            self.m.append(m)
            self.m_total += m
            self.s += 1
            # Initialize the hyperparameter dictionary
            if 'hyperparams' in arg:
                hyperparams = arg['hyperparams']
            else:
                hyperparams = {}
            hyperparams['des_tols'] = np.asarray(self.latent_des_tols)
            # Add the simulation's search and surrogate techniques
            self.searches.append(arg['search'](m,
                                               np.asarray(self.latent_lb),
                                               np.asarray(self.latent_ub),
                                               hyperparams))
            self.surrogates.append(arg['surrogate'](m,
                                                    np.asarray(self.latent_lb),
                                                    np.asarray(self.latent_ub),
                                                    hyperparams))
            # Add the simulation function and initialize its database
            self.sim_funcs.append(arg['sim_func'])
            self.sim_db.append({'x_vals': np.zeros((1, self.n_latent)),
                                's_vals': np.zeros((1, m)),
                                'n': 0,
                                'old': 0})
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

        from parmoo.util import check_names

        # Assert proper order of problem definition
        if len(self.acquisitions) > 0:
            raise RuntimeError("Cannot add more objectives after"
                               + " adding acquisition functions")
        for arg in args:
            # Check that the objective dictionary is a legal format
            if not isinstance(arg, dict):
                raise TypeError("Each arg must be a Python dict")
            if 'obj_func' in arg:
                if callable(arg['obj_func']):
                    if not (len(inspect.signature(arg['obj_func']).parameters)
                            in [2, 3]):
                        raise ValueError("The 'obj_func' must take 2 "
                                         + "(no derivatives) or 3 "
                                         + "(derivative option) arguments")
                else:
                    raise TypeError("The 'obj_func' must be callable")
            else:
                raise AttributeError("Each arg must conatain an 'obj_func'")
            # Add the objective name to the schema
            if 'name' in arg:
                name = arg['name']
            else:
                name = f"f{self.o + 1}"
            check_names(name, self.des_schema, self.sim_schema,
                        self.obj_schema, self.con_schema)
            self.obj_schema.append((name, 'f8'))
            # Finally, if all else passed, add the objective
            self.obj_funcs.append(arg['obj_func'])
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
                 * 'constraint' (function): An algebraic constraint function
                   that maps from R^n X R^m --> R and evaluates to zero or a
                   negative number when feasible and positive otherwise.
                   Interface should match:
                   `violation = constraint(x, sim_func(x), der=0)`,
                   where `der` is an optional argument specifying whether to
                   take the derivative of the constraint function
                    * 0 -- no derivative taken, return c(x, sim_func(x))
                    * 1 -- return derivative wrt x, or
                    * 2 -- return derivative wrt  sim(x).
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

        from parmoo.util import check_names

        # Assert proper order of problem definition
        if len(self.acquisitions) > 0:
            raise RuntimeError("Cannot add more constraints after"
                               + " adding acquisition functions")
        for arg in args:
            # Check that the constraint dictionary is a legal format
            if not isinstance(arg, dict):
                raise TypeError("Each arg must be a Python dict")
            if 'constraint' in arg:
                if callable(arg['constraint']):
                    if not (len(inspect.signature(arg['constraint']).
                                parameters) in [2, 3]):
                        raise ValueError("The 'constraint' must take 2 "
                                         + "(no derivatives) or 3 "
                                         + "(derivative option) arguments")
                else:
                    raise TypeError("The 'constraint' must be callable")
            else:
                raise AttributeError("Each arg must contain a 'constraint'")
            # Add the constraint name
            if 'name' in arg:
                name = arg['name']
            else:
                name = f"c{self.p + 1}"
            check_names(name, self.des_schema, self.sim_schema,
                        self.obj_schema, self.con_schema)
            self.con_schema.append((name, 'f8'))
            # Finally, if all else passed, add the constraint
            self.con_funcs.append(arg['constraint'])
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
                raise AttributeError("'acquisition' field must be present in "
                                     + "every arg")
            if self.n_latent < 1:
                raise RuntimeError("Cannot add acquisition function without"
                                   " any design variables")
            if self.o < 1:
                raise RuntimeError("Cannot add acquisition function without"
                                   " any objectives")
            if 'hyperparams' in arg:
                if not isinstance(arg['hyperparams'], dict):
                    raise TypeError("When present, 'hyperparams' must be a "
                                    + "Python dict")
                hyperparams = arg['hyperparams']
            else:
                hyperparams = {}
            try:
                acquisition = arg['acquisition'](self.o,
                                                 np.asarray(self.latent_lb),
                                                 np.asarray(self.latent_ub),
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
        self.checkpoint = checkpoint
        self.checkpoint_data = checkpoint_data
        self.checkpoint_file = filename
        return

    def getDesignType(self):
        """ Get the numpy dtype of all design points for this MOOP.

        Returns:
            np.dtype: The numpy dtype of this MOOP's design points.
            If no design variables have yet been added, returns None.

        """

        if self.n_feature < 1:
            return None
        else:
            return np.dtype(self.des_schema)

    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP.

        Returns:
            np.dtype: The numpy dtype of this MOOP's simulation outputs.
            If no simulations have been given, returns None.

        """

        if self.m_total < 1:
            return None
        else:
            return np.dtype(self.sim_schema)

    def getObjectiveType(self):
        """ Get the numpy dtype of an objective point for this MOOP.

        Returns:
            np.dtype: The numpy dtype of this MOOP's objective points.
            If no objectives have yet been added, returns None.

        """

        if self.o < 1:
            return None
        else:
            return np.dtype(self.obj_schema)

    def getConstraintType(self):
        """ Get the numpy dtype of the constraint violations for this MOOP.

        Returns:
            np.dtype: The numpy dtype of this MOOP's constraint violation
            outputs. If no constraint functions have been given, returns None.

        """

        if self.p < 1:
            return None
        else:
            return np.dtype(self.con_schema)

    def check_sim_db(self, x, s_name):
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
        xx = self._embed(x)
        des_tols = np.asarray(self.latent_des_tols)
        for j in range(self.sim_db[i]['n']):
            if np.all(np.abs(self.sim_db[i]['x_vals'][j, :] - xx) < des_tols):
                # If found, return the sim value
                return self.sim_db[i]['s_vals'][j, :]
        # Nothing found, return None
        return None

    def update_sim_db(self, x, sx, s_name):
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

        # Extract the simulation name
        i = -1
        for j, sj in enumerate(self.sim_schema):
            if sj[0] == s_name:
                i = j
                break
        if i < 0 or i > self.s - 1:
            raise ValueError("s_name did not contain a legal name/index")
        xx = self._embed(x)
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
            x (dict or numpy structured array): A numpy structured array or a
                Python dictionary specifying the keys/names and corresponding
                values of a design point to search for.

        s_name (str): The name of the simulation whose database will be
            searched.

            x (dict or numpy structured array): Either a numpy structured
                array or a Python dictionary with keys/names corresponding
                to the design variable names given and values containing
                the corresponding values of the design point to evaluate.

            s_name (str): The name of the simulation to evaluate.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the output from the
            sx = simulation[s_name](x).

        """

        sx = self.check_sim_db(x, s_name)
        if sx is None:
            sx = np.asarray(self.sim_funcs[i](x))
            self.update_sim_db(x, sx, s_name)
        return sx

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

        xx = self._embed(x)
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
            (list): A list of design points (numpy structured or 1D arrays) or
            tuples (design points, simulation name) specifying the unfiltered
            list of candidates that ParMOO recommends for true simulation
            evaluations. Specifically:
             * Each item or the first entry in tuple is either a numpy
               structured array (when operating with named variables) or a
               1D numpy.ndarray (in unnamed mode). When operating with
               unnamed variables, the indices were assigned in the order
               that the design variables were added to the MOOP using
               `MOOP.addDesign(*args)`.
             * If the item is a tuple, then the second entry in the tuple
               is either the (str) name of the simulation to
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
        # Check that ib is a list of legal integers or None
        if isinstance(ib, list) and all([isinstance(ibj, int) for ibj in ib]):
            for ibj in ib:
                if ibj < 0 or ibj >= len(self.acquisitions):
                    raise ValueError(f"invalid index found in ib: {ibj}")
        elif ib is not None:
            raise TypeError("when present, ib must be a list of int types")
        else:
            ib = [i for i in range(len(self.acquisitions))]
        # Check that there are design variables for this problem
        if self.n_latent == 0:
            raise AttributeError("there are no design vars for this problem")
        # Check that there are objectives
        if self.o == 0:
            raise AttributeError("there are no objectives for this problem")

        # Prepare a batch to return
        xbatch = []
        # Special rule for the k=0 iteration
        if k == 0:
            # Initialize the database
            self.n_dat = 0
            self.data = {'x_vals': np.zeros((1, self.n_latent)),
                         'f_vals': np.zeros((1, self.o)),
                         'c_vals': np.zeros((1, 1))}
            # Initialize the surrogate optimizer
            self.hyperparams['des_tols'] = self.latent_des_tols
            self.optimizer_obj = self.optimizer(self.o,
                                                np.asarray(self.latent_lb),
                                                np.asarray(self.latent_ub),
                                                self.hyperparams)
            self.optimizer_obj.setObjective(self._evaluate_objectives)
            self.optimizer_obj.setSimulation(self._evaluate_surrogates,
                                             self._surrogate_uncertainty)
            self.optimizer_obj.setPenalty(self._evaluate_penalty)
            self.optimizer_obj.setConstraints(self._evaluate_constraints)
            for i, acquisition in enumerate(self.acquisitions):
                self.optimizer_obj.addAcquisition(acquisition)
            self.optimizer_obj.setTrFunc(self._set_surrogate_tr)
            # Generate search data
            for j, search in enumerate(self.searches):
                des = search.startSearch(np.asarray(self.latent_lb),
                                         np.asarray(self.latent_ub))
                for xi in des:
                    xbatch.append((self._extract(xi),
                                   self.sim_schema[j][0]))
        # Now the main loop
        else:
            x0 = np.zeros((len(self.acquisitions), self.n_latent))
            # Set acquisition functions
            for i, acqi in enumerate(self.acquisitions):
                x0[i, :] = acqi.setTarget(self.data,
                                          self._evaluate_penalty,
                                          self.history)
            # Solve the surrogate problem
            x_candidates = self.optimizer_obj.solve(x0)
            # Create a batch for filter method
            for i, acqi in enumerate(self.acquisitions):
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
            *args (list of numpy.ndarrays or tuples): The list of
            unfiltered candidates returned by the MOOP.iterate() method.
            A list of design points (numpy structured or 1D arrays) or
            tuples (design points, simulation name) specifying the unfiltered
            list of candidates that ParMOO recommends for true simulation
            evaluations. Specifically:
             * Each item or the first entry in tuple is either a numpy
               structured array (when operating with named variables) or a
               1D numpy.ndarray (in unnamed mode). When operating with
               unnamed variables, the indices were assigned in the order
               that the design variables were added to the MOOP using
               `MOOP.addDesign(*args)`.
             * If the item is a tuple, then the second entry in the tuple
               is either the (str) name of the simulation to
               evaluate (when operating with named variables) or the (int)
               index of the simulation to evaluate (when operating in
               unnamed mode). Note, in unnamed mode, simulation indices
               were assigned in the order that they were added using
               `MOOP.addSimulation(*args)`.

        Returns:
            (list): A filtered list of ordered pairs (tuples), specifying
            the (design points, simulation name) that ParMOO suggests for
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
                           and self.check_sim_db(xi, namei) is None:
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
                                   any([self.check_sim_db(self._extract(xk),
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
            self._fit_surrogates()
            if self.s > 0:
                # Check every point in sim_db[0]
                des_tols = np.asarray(self.latent_des_tols)
                for xi, si in zip(self.sim_db[0]['x_vals'],
                                  self.sim_db[0]['s_vals']):
                    sim = np.zeros(self.m_total)
                    sim[0:self.m[0]] = si[:]
                    m_count = self.m[0]
                    is_shared = True
                    # Check for xi in sim_db[1:s]
                    for j in range(1, self.s):
                        is_shared = False
                        for xj, sj in zip(self.sim_db[j]['x_vals'],
                                          self.sim_db[j]['s_vals']):
                            # If found, update sim value and break loop
                            if np.all(np.abs(xi - xj) < des_tols):
                                sim[m_count:m_count + self.m[j]] = sj[:]
                                m_count = m_count + self.m[j]
                                is_shared = True
                                break
                        if not is_shared:
                            break
                    # If xi was in every sim_db, add it to the database
                    if is_shared:
                        self.addData(self._extract(xi), self._unpack_sim(sim))
        else:
            # If any constraints are violated, increase lam toward the limit
            for (xi, i) in batch:
                xxi = self._embed(xi)
                sxi = self._evaluate_surrogates(xxi)
                eps = np.sqrt(self.epsilon)
                if np.any(self._evaluate_constraints(xxi, sxi) > eps):
                    self.lam = min(1e4, self.lam * 2.0)
                    break
            # Update the models and objective database
            self._update_surrogates()
            for xi in batch:
                (x, i) = xi
                xx = self._embed(x)
                is_shared = True
                sim = np.zeros(self.m_total)
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
                                sim[m_count:m_count + self.m[j]] = sj[:]
                                m_count = m_count + self.m[j]
                                is_shared = True
                                break
                        # If not found, stop checking
                        if not is_shared:
                            break
                # If xi was in every sim_db, add it to the database and report
                # to the optimizer
                if is_shared:
                    fx = np.zeros(self.o)
                    sx = self._unpack_sim(sim)
                    sdx = self._unpack_sim(np.zeros(self.m_total))
                    for i, obj_func in enumerate(self.obj_funcs):
                        fx[i] = obj_func(x, sx)
                    self.addData(x, sx)
                    self.optimizer_obj.returnResults(xx, fx, sim, np.zeros(self.m_total))
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

        """

        import logging

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

        # Print logging info summary of problem setup
        logging.info(" Beginning new run of ParMOO...")
        logging.info(" summary of settings:")
        logging.info(f"   {self.n_feature} design dimensions")
        logging.info(f"   {self.n_latent} embedded design dimensions")
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
        # Build the data type
        dt = []
        for dname in self.des_schema:
            dt.append((str(dname[0]), dname[1]))
        for fname in self.obj_schema:
            dt.append((str(fname[0]), fname[1]))
        for cname in self.con_schema:
            dt.append((str(cname[0], cname[1])))
        # Initialize result array
        result = np.zeros(pf['x_vals'].shape[0], dtype=dt)
        # Extract all results
        if self.n_dat > 0:
            for i, xi in enumerate(pf['x_vals']):
                xxi = self._extract(xi)
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

        # Initialize result dictionary
        result = {}
        # For each simulation
        for i, sname in enumerate(self.sim_schema):
            # Extract all results
            x_vals = np.asarray([self._extract(xi)
                                 for xi in self.sim_db[i]['x_vals']])
            # Build the datatype
            dt = []
            for dname in self.des_schema:
                dt.append((str(dname[0]), dname[1]))
            if len(sname) == 2:
                dt.append(('out', sname[1]))
            else:
                dt.append(('out', sname[1], sname[2]))
            # Initialize result array for sname[i]
            result[sname[0]] = np.zeros(self.sim_db[i]['n'], dtype=dt)
            if self.sim_db[i]['n'] > 0:
                # Copy results
                for (name, t) in self.des_schema:
                    result[sname[0]][name][:] = x_vals[name][:]
                if len(sname) == 2:
                    result[sname[0]]['out'] = self.sim_db[i]['s_vals'][:,
                                                                       0]
                else:
                    result[sname[0]]['out'] = self.sim_db[i]['s_vals']
        if format == 'pandas':
            # For simulation data, converting to pandas is a little more
            # complicated...
            result_pd = {}
            for i, snamei in enumerate(result.keys()):
                rtempi = {}
                for (name, t) in self.des_schema:
                    rtempi[name] = result[snamei][name]
                # Need to break apart the output column manually
                if self.m[i] > 1:
                    for i in range(self.m[i]):
                        rtempi[f'out_{i}'] = result[snamei]['out'][:, i]
                else:
                    rtempi['out'] = result[snamei]['out'][:, 0]
                # Create dictionary of dataframes, indexed by sim names
                result_pd[snamei] = pd.DataFrame(rtempi)
            return result_pd
        elif format == 'ndarray':
            return result
        else:
            raise ValueError(str(format) + "is an invalid value for "
                             + "'format'")

    def getObjectiveData(self, format='ndarray'):
        """ Extract all computed objective scores from this MOOP's database.

        Args:
            format (str, optional): Either 'ndarray' (default) or 'pandas',
                in order to produce output as a numpy structured array or
                pandas dataframe. Note: format='pandas' is only valid for
                named inputs.

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

        # Build the data type
        dt = []
        for dname in self.des_schema:
            dt.append((str(dname[0]), dname[1]))
        for fname in self.obj_schema:
            dt.append((str(fname[0]), fname[1]))
        for cname in self.con_schema:
            dt.append((str(cname[0], cname[1])))
        # Initialize result array
        if self.n_dat > 0:
            result = np.zeros(self.data['x_vals'].shape[0], dtype=dt)
        else:
            result = np.zeros(0, dtype=dt)
        # Extract all results
        if self.n_dat > 0:
            x_vals = np.asarray([self._extract(xi)
                                 for xi in self.data['x_vals']])
            for (name, t) in self.des_schema:
                result[name][:] = x_vals[name][:]
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
        parmoo_state = {'n_latent': self.n_latent,
                        'm': self.m,
                        'm_total': self.m_total,
                        'o': self.o,
                        'p': self.p,
                        's': self.s,
                        'n_dat': self.n_dat,
                        'sim_schema': self.sim_schema,
                        'des_schema': self.des_schema,
                        'obj_schema': self.obj_schema,
                        'con_schema': self.con_schema,
                        'lam': self.lam,
                        'epsilon': self.epsilon,
                        'hyperparams': self.hyperparams,
                        'history': self.history,
                        'iteration': self.iteration,
                        'checkpoint': self.checkpoint,
                        'checkpoint_data': self.checkpoint_data,
                        'checkpoint_file': self.checkpoint_file}
        # Serialize numpy arrays
        if isinstance(self.latent_lb, np.ndarray):
            parmoo_state['latent_lb'] = self.latent_lb.tolist()
        else:
            parmoo_state['latent_lb'] = self.latent_lb
        if isinstance(self.latent_ub, np.ndarray):
            parmoo_state['latent_ub'] = self.latent_ub.tolist()
        else:
            parmoo_state['latent_ub'] = self.latent_ub
        if isinstance(self.latent_des_tols, np.ndarray):
            parmoo_state['latent_des_tols'] = self.latent_des_tols.tolist()
        else:
            parmoo_state['latent_des_tols'] = self.latent_des_tols
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
        # Store names/modules of object classes
        parmoo_state['optimizer'] = (self.optimizer.__name__,
                                     self.optimizer.__module__)
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
            self.optimizer_obj.save(fname_tmp)
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
        self.n_latent = parmoo_state['n_latent']
        self.m = parmoo_state['m']
        self.m_total = parmoo_state['m_total']
        self.o = parmoo_state['o']
        self.p = parmoo_state['p']
        self.s = parmoo_state['s']
        self.n_dat = parmoo_state['n_dat']
        self.sim_schema = [tuple(item) for item in parmoo_state['sim_schema']]
        self.des_schema = [tuple(item) for item in parmoo_state['des_schema']]
        self.obj_schema = [tuple(item) for item in parmoo_state['obj_schema']]
        self.con_schema = [tuple(item)
                           for item in parmoo_state['con_schema']]
        self.lam = parmoo_state['lam']
        self.epsilon = parmoo_state['epsilon']
        self.hyperparams = parmoo_state['hyperparams']
        self.history = parmoo_state['history']
        self.iteration = parmoo_state['iteration']
        self.checkpoint = parmoo_state['checkpoint']
        self.checkpoint_data = parmoo_state['checkpoint_data']
        self.checkpoint_file = parmoo_state['checkpoint_file']
        # Reload serialize numpy arrays
        self.latent_lb = np.array(parmoo_state['latent_lb'])
        self.latent_ub = np.array(parmoo_state['latent_ub'])
        self.latent_des_tols = np.array(parmoo_state['latent_des_tols'])
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
        self.obj_funcs = []
        for (obj_name, obj_mod), info in zip(parmoo_state['obj_funcs'],
                                             parmoo_state['obj_funcs_info']):
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
            self.obj_funcs.append(toadd)
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
        self.con_funcs = []
        for (const_name, const_mod), info in \
                zip(parmoo_state['con_funcs'],
                    parmoo_state['con_funcs_info']):
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
            self.con_funcs.append(toadd)
        # Recover object classes and instances
        mod = import_module(parmoo_state['optimizer'][1])
        self.optimizer = getattr(mod, parmoo_state['optimizer'][0])
        self.searches = []
        for i, (search_name, search_mod) in enumerate(
                                                parmoo_state['searches']):
            mod = import_module(search_mod)
            new_search = getattr(mod, search_name)
            toadd = new_search(self.m[i], self.latent_lb, self.latent_ub, {})
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
            toadd = new_sur(self.m[i], self.latent_lb, self.latent_ub, {})
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
            toadd = new_acq(self.o, self.latent_lb, self.latent_ub, {})
            try:
                fname = filename + ".acquisition." + str(i + 1)
                toadd.load(fname)
            except NotImplementedError:
                pass
            self.acquisitions.append(toadd)
        # Rebuild the optimizer object
        self.optimizer_obj = self.optimizer(self.o, self.latent_lb, self.latent_ub, {})
        self.optimizer_obj.setObjective(self._evaluate_objectives)
        self.optimizer_obj.setSimulation(self._evaluate_surrogates,
                                         self._surrogate_uncertainty)
        self.optimizer_obj.setPenalty(self._evaluate_penalty)
        self.optimizer_obj.setConstraints(self._evaluate_constraints)
        for i, acquisition in enumerate(self.acquisitions):
            self.optimizer_obj.addAcquisition(acquisition)
        self.optimizer_obj.setTrFunc(self._set_surrogate_tr)
        try:
            fname = filename + ".optimizer"
            self.optimizer_obj.load(fname)
        except NotImplementedError:
            pass
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
        toadd = {'sim_id': s_name}
        for key in x.dtype.names:
            toadd[key] = x[key]
        if isinstance(sx, np.ndarray):
            toadd['out'] = sx.tolist()
        else:
            toadd['out'] = sx
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
            xx.append(jnp.array(ei.embed(x[self.des_schema[i][0]])).flatten())
        return jnp.concatenate(xx)

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
            iend = istart + self.embedding_size[i]
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

        sx_list = []
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
        for i, mi in enumerate(self.m):
            iend = istart + mi
            sx_out[self.sim_schema[i][0]] = sx[istart:iend]
            istart = iend
        return sx_out

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
            center (numpy.ndarray): A 1d numpy.ndarray containing the
                (embedded) coordinates of the new center in the rescaled
                design space.

            radius (np.ndarray or float): The trust region radius.

        """

        for si in self.surrogates:
            si.setTrustRegion(center, radius)
        return

    def _evaluate_surrogates(self, x):
        """ Evaluate all simulation surrogates.

        Args:
            x (numpy.ndarray): A 1d numpy.ndarray containing the (embedded)
                design point to evaluate.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the (embedded) result
            of the surrogate model evaluations.

        """

        sx_list = [self.empty]
        for si in self.surrogates:
            sx_list.append(si.evaluate(x))
        return jnp.concatenate(sx_list)

    def _surrogate_uncertainty(self, x):
        """ Evaluate the standard deviation of the possible surrogate outputs.

        Args:
            x (numpy.ndarray): A 1d numpy.ndarray containing the (embedded)
                design point to evaluate uncertainties at.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the standard
            deviation of the surrogates at x.

        """

        sdx_list = [self.empty]
        for si in self.surrogates:
            sdx_list.append(si.stdDev(x))
        return jnp.concatenate(sdx_list)

    def _evaluate_objectives(self, x, sx):
        """ Evaluate all objectives using the simulation surrogates as needed.

        Args:
            x (numpy.ndarray): A 1d numpy.ndarray containing the (embedded)
                design point to evaluate.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the result of the
            evaluation.

        """

        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        fx = []
        for i, obj_func in enumerate(self.obj_funcs):
            fx.append(jnp.array(obj_func(xx, ssx)).flatten())
        return jnp.concatenate(fx)

    def _evaluate_constraints(self, x, sx):
        """ Evaluate the constraints using the simulation surrogates as needed.

        Args:
            x (numpy.ndarray): A 1d numpy.ndarray containing the (embedded)
                design point to evaluate.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the list of constraint
            violations at x (zero if no violation).

        """

        if self.p == 0:
            return self.empty
        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        cx = []
        for i, constraint_func in enumerate(self.con_funcs):
            cx.append(jnp.array(constraint_func(xx, ssx)).flatten())
        return jnp.concatenate(cx)

    def _evaluate_penalty(self, x, sx):
        """ Evaluate the penalized objective using the surrogates as needed.

        Args:
            x (numpy.ndarray): A 1d numpy.ndarray containing the (embedded)
                design point to evaluate.

        Returns:
            numpy.ndarray: A 1d numpy.ndarray containing the result of the
            evaluation.

        """

        xx = self._extract(x)
        ssx = self._unpack_sim(sx)
        fx = []
        for i, obj_func in enumerate(self.obj_funcs):
            fx.append(jnp.array(obj_func(xx, ssx)).flatten())
        cx = 0.0
        for i, constraint_func in enumerate(self.con_funcs):
            cx += jnp.maximum(constraint_func(xx, ssx), 0)
        return jnp.concatenate(fx) + (self.lam * cx)
