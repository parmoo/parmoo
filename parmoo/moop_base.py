
""" Contains the MOOP class for defining multiobjective optimization problems.

``parmoo.moop.MOOP`` is the base class for defining and solving multiobjective
optimization problems (MOOPs). Each MOOP object may contain several
simulations, specified using dictionaries.

"""

from abc import ABC, abstractmethod
import logging

import jax
from jax import numpy as jnp

from parmoo.simulation_database import SimulationDatabase
from parmoo.util import gradient_error


class MOOP_base(ABC):
    """ ABC for defining a multiobjective optimization problem (MOOP).

    The following public methods must be defined by the implementation:

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

    Database accessors:
     * ``MOOP.updateSimDb(x, sx, s_name)``
     * ``MOOP.checkSimDb(x, s_name)``
     * ``MOOP.getPF(format='ndarray')``
     * ``MOOP.getSimulationData(format='ndarray')``
     * ``MOOP.getObjectiveData(format='ndarray')``
   
   Solver steps:
     * ``MOOP.evaluateSimulation(x, s_name)``
     * ``MOOP.addObjData(x, sx)``
     * ``MOOP.iterate(k, ib=None)``
     * ``MOOP.filterBatch(*args)``
     * ``MOOP.updateAll(k, batch)``
     * ``MOOP.solve(iter_max=None, sim_max=None)``

    Checkpointing methods:
     * ``MOOP.setCheckpoint(checkpoint, [checkpoint_data, filename])``
     * ``MOOP.save([filename="parmoo"])``
     * ``MOOP.load([filename="parmoo"])``
     * ``MOOP.savedata(x, sx, s_name, [filename="parmoo"])``

    The following private methods are implemented herein:
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
                 'm', 'm_list', 'n_embed', 'n_latent',
                 'o', 'p', 's',
                 # Tolerances and bounds
                 'feature_des_tols', 'latent_des_tols', 'cont_var_inds',
                 # Schemas
                 'des_schema', 'sim_schema', 'obj_schema', 'con_schema',
                 # Constants, counters, and adaptive parameters
                 'compiled', 'empty', 'epsilon', 'iteration', 'lam',
                 # Design variables, simulations, objectives, and constraints
                 'embedders',
                 'obj_funcs', 'obj_grads', 'con_funcs', 'con_grads',
                 # Database information
                 'database',
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

        Returns:
            MOOP: A new MOOP object with no design variables, objectives, or
            constraints.

        """

        # Configure jax to use only CPUs
        jax.config.update('jax_platform_name', 'cpu')
        # Initialize the problem dimensions
        self.m = 0
        self.m_list, self.n_embed = [], []
        self.n_latent = 0
        self.o, self.p, self.s = 0, 0, 0
        # Initialize the bounds and tolerances
        self.feature_des_tols, self.latent_des_tols = [], []
        self.cont_var_inds = []
        # Initialize the schemas
        self.des_schema, self.sim_schema = [], []
        self.obj_schema, self.con_schema = [], []
        # Initialize the constants, counters, and adaptive parameters
        self.compiled = False
        self.empty = jnp.zeros(0)
        self.epsilon = jnp.sqrt(jnp.finfo(jnp.ones(1)).eps)
        self.iteration = 0
        self.lam = 1.0
        # Initialize design variable embeddings
        self.embedders = []
        # Initialize objectives, constraints
        self.obj_funcs, self.obj_grads = [], []
        self.con_funcs, self.con_grads = [], []
        # Initialize the database
        self.database = SimulationDatabase()
        # Initialize backward pass functions
        self.obj_bwd, self.con_bwd, self.pen_bwd = None, None, None
        return

    @abstractmethod
    def addDesign(self, *args):
        """ Add a new design variables to the MOOP. """

    @abstractmethod
    def addSimulation(self, *args):
        """ Add new simulations to the MOOP. """

    @abstractmethod
    def addObjective(self, *args):
        """ Add a new objective to the MOOP. """

    @abstractmethod
    def addConstraint(self, *args):
        """ Add a new constraint to the MOOP. """

    @abstractmethod
    def addAcquisition(self, *args):
        """ Add an acquisition function to the MOOP. """

    @abstractmethod
    def compile(self):
        """ Compile the MOOP object and initialize its components. """

    @abstractmethod
    def setCheckpoint(
            self, checkpoint, checkpoint_data=True, filename="parmoo"
    ):
        """ Set ParMOO's checkpointing feature. """

    @abstractmethod
    def getDesignType(self):
        """ Get the numpy dtype of all design points for this MOOP. """

    @abstractmethod
    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP. """

    @abstractmethod
    def getObjectiveType(self):
        """ Get the numpy dtype of an objective point for this MOOP. """

    @abstractmethod
    def getConstraintType(self):
        """ Get the numpy dtype of the constraint violations for this MOOP. """

    @abstractmethod
    def checkSimDb(self, x, s_name):
        """ Check self.sim_db[s_name] to see if the design x was evaluated. """

    @abstractmethod
    def updateSimDb(self, x, sx, s_name):
        """ Update a sim_db by adding a design/simulation output pair. """

    @abstractmethod
    def evaluateSimulation(self, x, s_name):
        """ Evaluate sim_func[s_name] and store the result in the database. """

    @abstractmethod
    def addObjData(self, x, sx):
        """ Update the internal objective database by truly evaluating x. """

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

    @abstractmethod
    def getPF(self, format='ndarray'):
        """ Extract nondominated and efficient sets from internal database. """

    @abstractmethod
    def getSimulationData(self, format='ndarray'):
        """ Extract raw simulation outputs from the MOOP's database. """

    @abstractmethod
    def getObjectiveData(self, format='ndarray'):
        """ Extract all computed objective scores from the MOOP's database. """

    @abstractmethod
    def save(self, filename="parmoo"):
        """ Serialize and save the MOOP object and all of its dependencies. """

    @abstractmethod
    def load(self, filename="parmoo"):
        """ Load a serialized MOOP object and all of its dependencies. """

    @abstractmethod
    def savedata(self, x, sx, s_name, filename="parmoo"):
        """ Save the current simulation database for this MOOP. """

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

        sim_db = self.database.getSimulationData()
        for i, dti in enumerate(self.sim_schema):
            sim_namei = dti[0]
            n = len(sim_db[sim_namei])
            x_vals = np.zeros((n, self.n_latent))
            for j, xj in enumerate(sim_db[sim_namei]):
                x_vals[j, :] = self._embed(xj)
            self.surrogates[i].fit(x_vals, sim_db[sim_namei]['out'])
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
        eval_obj, eval_con, eval_pen = self._link()
        self.optimizer.setObjective(eval_obj)
        self.optimizer.setConstraints(eval_con)
        self.optimizer.setPenalty(eval_pen)
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
        return self._vpen_funcs(xx, ssx, cx, self.lam)

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
        act = (jnp.isclose(cx, jnp.zeros(cx.shape)) - 1) * -self.lam
        return self._vpen_funcs(xx, ssx, jnp.sum(cx), self.lam), (x, sx, act)

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

    def _jit_all(self, xx, sx):
        """ Attempt to JIT all internal functions and log any failures.

        Args:
            xx (jnp.ndarray):  A sample design point used to trigger JIT.
            sx (jnp.ndarray):  A sample simulation output used to trigger JIT.

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
            logging.info("     WARNING: 1 or more grad embedders failed to "
                         "jit...")
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
                logging.info("     WARNING: 1 or more obj_grads failed to "
                             "jit...")
            self.obj_bwd = self._obj_bwd
        else:
            self.obj_bwd = gradient_error
        if len(self.con_grads) == self.p:
            try:
                _, _ = jax.jit(self._con_bwd)((xx, sx), jnp.zeros(self.p))
            except BaseException:
                logging.info("     WARNING: 1 or more con_grads failed to "
                             "jit...")
            self.con_bwd = self._con_bwd
        else:
            self.con_bwd = gradient_error
        if len(self.obj_grads) == self.o and len(self.con_grads) == self.p:
            try:
                _, _ = jax.jit(self._pen_bwd)((xx, sx, jnp.zeros(self.p)),
                                              jnp.zeros(self.o))
            except BaseException:
                logging.info("     WARNING: MOOP._pen_grads failed to jit...")
            self.pen_bwd = self._pen_bwd
        else:
            self.pen_bwd = gradient_error
        logging.info("   Done.")
