
""" Contains the libE_MOOP class and parmoo_persis_gen function.

Use the libE_MOOP class to define and solve multiobjective optimization
problems (MOOPs) with parallel simulation evaluations. The libE_MOOP class
extends the base class parmoo.moop.MOOP for defining and solving MOOPs.

The parmoo_persis_gen function can be used as a generator function in
libEnsemble. To do so, create a regular `parmoo.MOOP` object and add it to
the `gen_specs` dict, then import and use `parmoo_persis_gen` as the libE
gen func.

"""

import numpy as np
from parmoo import MOOP
import warnings


def parmoo_persis_gen(H, persis_info, gen_specs, libE_info):
    """ A persistent ParMOO generator function for libEnsemble.

    This generator function is meant to be called from within libEnsemble.

    Args:
        H (numpy structured array): The current libE history array.

        persis_info (dict): Any information that should persist after this
            generator has exited. Must contain the following field:
             * 'moop' (parmoo.MOOP)

        gen_specs (dict): A list of specifications for the generator function.

        libE_info (dict): Other information that will be used by libEnsemble.

    Returns:
        dict: The final simulation history.

        dict: The persistent information after completion of the generator.

        int: The stop tag.

    """

    from libensemble.message_numbers import STOP_TAG, PERSIS_STOP, EVAL_GEN_TAG
    from libensemble.message_numbers import FINISHED_PERSISTENT_GEN_TAG
    from libensemble.tools.persistent_support import PersistentSupport

    # Get moop from pers_info
    if 'moop' in persis_info:
        moop = persis_info['moop']
        if not isinstance(moop, MOOP):
            raise TypeError("persis_info['moop'] must be an instance of " +
                            "parmoo.MOOP class")
    else:
        raise KeyError("'moop' key is required in persis_info dict")
    # Setup persistent support
    ps = PersistentSupport(libE_info, EVAL_GEN_TAG)
    # Send batches until manager sends stop tag
    tag = None
    k = 0
    sim_count = 0
    # Iterate until the termination condition is reached
    while tag not in [STOP_TAG, PERSIS_STOP]:
        # Generate a batch by running one iteration
        x_out = moop.iterate(k)
        # Check for duplicates in simulation databases
        xbatch = []
        ibatch = []
        for (xi, i) in x_out:
            if moop.check_sim_db(xi, i) is None:
                xbatch.append(xi)
                ibatch.append(i)
        # Get the batch size and allocate the H_o structured array
        b = len(xbatch)
        H_o = np.zeros(b, dtype=gen_specs['out'])
        # Populate the H_o structured array 'x' values as appropriate
        if moop.use_names:
            for name in moop.des_names:
                for i in range(b):
                    H_o[name[0]][i] = xbatch[i][name[0]]
        else:
            H_o['x'] = np.asarray(xbatch)
        for i, namei in enumerate(ibatch):
            H_o['sim_name'][i] = namei
        # Evaluate H_o and add to the simulation database
        batch = []
        if isinstance(x_out[0][-1], str) or x_out[0][-1] >= 0:
            tag, Work, calc_in = ps.send_recv(H_o)
            if calc_in is not None:
                for s_out in calc_in:
                    sim_name = s_out['sim_name']
                    # Check whether design variables are all named
                    if moop.use_names:
                        xx = np.zeros(1, dtype=moop.des_names)[0]
                        for name in moop.des_names:
                            xx[name[0]] = s_out[name[0]]
                        sim_num = -1
                        for j, sj in enumerate(moop.sim_names):
                            if sj[0] == sim_name:
                                sim_num = j
                                break
                        sx = np.zeros(moop.m[sim_num])
                        sx[:] = s_out[moop.sim_names[sim_num][0]]
                        sname = sim_name.decode('utf-8')
                    else:
                        xx = np.zeros(moop.n)
                        xx[:] = s_out['x'][:]
                        sx = np.zeros(moop.m[sim_name])
                        sx[:] = s_out['f'][:]
                        sname = int(sim_name)
                    # Copy sim results into ParMOO databases
                    moop.update_sim_db(xx, sx, sname)
                    batch.append((xx, sname))
                    sim_count += 1
            else:
                new_count = 0
                for s_out in Work[sim_count:]:
                    sim_name = s_out['sim_name']
                    # Check whether design variables are all named
                    if moop.use_names:
                        xx = np.zeros(1, dtype=moop.des_names)[0]
                        for name in moop.des_names:
                            xx[name[0]] = s_out[name[0]]
                        sim_num = -1
                        for j, sj in enumerate(moop.sim_names):
                            if sj[0] == sim_name:
                                sim_num = j
                                break
                        sx = np.zeros(moop.m[sim_num])
                        sx[:] = s_out[moop.sim_names[sim_num][0]]
                        sname = sim_name.decode('utf-8')
                    else:
                        xx = np.zeros(moop.n)
                        xx[:] = s_out['x'][:]
                        sx = np.zeros(moop.m[sim_name])
                        sx[:] = s_out['f'][:]
                        sname = int(sim_name)
                    # Copy sim results into ParMOO databases
                    moop.update_sim_db(xx, sx, sname)
                    batch.append((xx, sname))
                    new_count += 1
                sim_count += new_count
        # Update the ParMOO databases
        moop.updateAll(k, batch)
        k += 1
    # Return the results
    persis_info['moop'] = moop
    return H_o, persis_info, FINISHED_PERSISTENT_GEN_TAG


class libE_MOOP(MOOP):
    """ Class for solving a MOOP using libEnsemble to manage parallelism.

    Upon initialization, supply a scalar optimization procedure and
    dictionary of hyperparameters using the default constructor:
     * ``moop = libE_MOOP.__init__(ScalarOpt, [hyperparams={}])``

    Class methods are summarized below.

    To define the MOOP, add each design variable, simulation, objective, and
    constraint (in that order) by using the following functions:
     * ``libE_MOOP.addDesign(*args)``
     * ``libE_MOOP.addSimulation(*args)``
     * ``libE_MOOP.addObjective(*args)``
     * ``libE_MOOP.addConstraint(*args)``

    Next, define your solver.

    Acquisition functions (used for scalarizing problems/setting targets) are
    added using:
     * ``libE_MOOP.addAcquisition(*args)``

    After creating a MOOP, the following methods may be useful for getting
    the numpy.dtype of the input/output arrays:
     * ``libE_MOOP.getDesignType()``
     * ``libE_MOOP.getSimulationType()``
     * ``libE_MOOP.getObjectiveType()``
     * ``libE_MOOP.getConstraintType()``

    The following methods are used to save/load ParMOO objects from memory:
     * ``libE_MOOP.save([filename="parmoo"])``
     * ``libE_MOOP.load([filename="parmoo"])``

    To turn on checkpointing (recommended), use:
     * ``libE_MOOP.setCheckpoint(checkpoint, [checkpoint_data, filename])``

    ParMOO's logging feature is not active for the `libE_MOOP` class
    since libEnsemble already provides this feature.

    After defining the MOOP and setting up checkpointing and logging,
    use the following method to solve the MOOP (using libEnsemble
    to distribute simulation evaluations):
     * ``libE_MOOP.solve(iter_max=None, sim_max=None, wt_max=864000,
                         profile=False)``

    The following methods are used for managing ParMOO's internal
    simulation/objective databases. Note that these databases are
    maintained separately from libEnsemble's simulation database:
     * ``libE_MOOP.check_sim_db(x, s_name)``
     * ``libE_MOOP.update_sim_db(x, sx, s_name)``
     * ``libE_MOOP.evaluateSimulation(x, s_name)``
     * ``libE_MOOP.addData(x, sx)``
     * ``libE_MOOP.iterate(k)``
     * ``libE_MOOP.updateAll(k, batch)``

    Finally, the following methods are used to retrieve data after the
    problem has been solved:
     * ``libE_MOOP.getPF(format='ndarray')``
     * ``libE_MOOP.getSimulationData(format='ndarray')``
     * ``libE_MOOP.getObjectiveData(format='ndarray')``

    Other private methods from the MOOP class do not work for a libE_MOOP.

    """

    __slots__ = ['moop']

    def __init__(self, opt_func, hyperparams=None):
        """ Initializer for the libE interface to the MOOP class.

        Args:
            opt_func (SurrogateOptimizer): A solver for the surrogate problems.

            hyperparams (dict, optional): A dictionary of hyperparameters for
                the opt_func, and any other procedures that will be used.

        Returns:
            libE_MOOP: A new libE_MOOP object with no design variables,
                objectives, or constraints.

        """

        if hyperparams is None:
            hp = {}
        else:
            hp = hyperparams
        # Create a MOOP
        self.moop = MOOP(opt_func, hyperparams=hp)
        return

    def addDesign(self, *args):
        """ Add a new design variables to the libE_MOOP.

        Append new design variables to the problem. Note that every design
        variable must be added before any simulations or acquisition functions
        can be added since the number of design variables is used to infer
        the size of simulation databases and acquisition function policies.

        Args:
            args (dict): Each argument is a dictionary representing one design
                variable. The dictionary contains information about that
                design variable, including:
                 * 'name' (str, optional): The name of this design
                   if any are left blank, then ALL names are considered
                   unspecified.
                 * 'des_type' (str): The type for this design variable.
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
                   dimension, before two design values are considered
                   to have equal values in this dimension. If not specified,
                   the default value is 1.0e-8.
                 * 'levels' (int): When des_type is 'categorical', this
                   specifies the number of levels for the variable.

        """

        self.moop.addDesign(*args)
        return

    def addSimulation(self, *args):
        """ Add new simulations to the libE_MOOP.

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
                   Most notably, 'search_budget': (int) can be specified
                   here.

        """

        self.moop.addSimulation(*args)
        return

    def addObjective(self, *args):
        """ Add a new objective to the libE_MOOP.

        Append a new objective to the problem. The objective must be an
        algebraic function of the design variables and simulation outputs.
        Note that all objectives must be specified before any acquisition
        functions can be added.

        Args:
            args (dict): Python dictionary containing objective function
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

        self.moop.addObjective(*args)
        return

    def addConstraint(self, *args):
        """ Add a new constraint to the libE_MOOP.

        Append a new constraint to the problem. The constraint can be
        a linear or nonlinear inequality constraint, and may depend on
        the design variables and/or the simulation outputs.

        Args:
            *args (dict): Python dictionary containing constraint function
                information, including:
                 * 'name' (str, optional): The name of this constraint
                   (defaults to "const" + str(i), where i = 1, 2, 3, ... for
                   the first, second, third, ... constraint added to the
                   MOOP).
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

        self.moop.addConstraint(*args)
        return

    def addAcquisition(self, *args):
        """ Add an acquisition function to the libE_MOOP.

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

        self.moop.addAcquisition(*args)
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

        self.moop.setCheckpoint(checkpoint, checkpoint_data=checkpoint_data,
                                filename=filename)
        return

    def getDesignType(self):
        """ Get the numpy dtype of all design points for this MOOP.

        Use this type when allocating a numpy array to store the design
        points for this MOOP object.

        Returns:
            np.dtype: The numpy dtype of this MOOP's design points.
            If no design variables have yet been added, returns None.

        """

        return self.moop.getDesignType()

    def getSimulationType(self):
        """ Get the numpy dtypes of the simulation outputs for this MOOP.

        Use this type if allocating a numpy array to store the simulation
        outputs of this MOOP object.

        Returns:
            np.dtype: The numpy dtype of this MOOP's simulation outputs.
            If no simulations have been given, returns None.

        """

        return self.moop.getSimulationType()

    def getObjectiveType(self):
        """ Get the numpy dtype of an objective point for this MOOP.

        Use this type if allocating a numpy array to store the objective
        values of this MOOP object.

        Returns:
            np.dtype: The numpy dtype of this MOOP's objective points.
            If no objectives have yet been added, returns None.

        """

        return self.moop.getObjectiveType()

    def getConstraintType(self):
        """ Get the numpy dtype of the constraint violations for this MOOP.

        Use this type if allocating a numpy array to store the constraint
        scores output of this MOOP object.

        Returns:
            np.dtype: The numpy dtype of this MOOP's constraint violation
            output. If no constraints have been given, returns None.

        """

        return self.moop.getConstraintType()

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

        return self.moop.check_sim_db(x, s_name)

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

        self.moop.update_sim_db(x, sx, s_name)
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

        return self.moop.evaluateSimulation(x, s_name)

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

        self.moop.addData(x, sx)
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

        return self.moop.iterate(k)

    def updateAll(self, k, batch):
        """ Update all surrogates given a batch of freshly evaluated data.

        Args:
            k (int): The iteration counter (corresponding to the value in
                libE_MOOP.moop.iteration).

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

        return self.moop.updateAll(k, batch)

    def moop_sim(self, H, persis_info, sim_specs, _):
        """ Evaluates the sim function for a collection of points given in
        ``H['x']``.

        """

        batch = len(H)
        sim_names = H['sim_name']
        H_o = np.zeros(batch, dtype=sim_specs['out'])
        for i in range(batch):
            namei = sim_names[i]
            if self.moop.use_names:
                j = -1
                for jj, jname in enumerate(self.moop.sim_names):
                    if jname[0] == sim_names[i]:
                        j = jj
                        break
            else:
                j = namei
            if self.moop.use_names:
                xx = np.zeros(1, dtype=self.moop.des_names)[0]
                for name in self.moop.des_names:
                    xx[name[0]] = H[name[0]][i]
                H_o[self.moop.sim_names[j][0]][i] = \
                    self.moop.sim_funcs[j](xx)
            else:
                H_o['f'][i, :self.moop.m[j]] = \
                    self.moop.sim_funcs[j](H['x'][i])
        return H_o, persis_info

    def solve(self, iter_max=None, sim_max=None, wt_max=864000, profile=False):
        """ Solve a MOOP using ParMOO + libEnsemble.

        If desired, be sure to turn on checkpointing before starting the
        solve, using:

        ``MOOP.setCheckpoint(checkpoint, [checkpoint_data, filename])``

        ParMOO will solve the MOOP and use libEnsemble to distribute
        simulations over available resources.

        Args:
            iter_max (int): The max number of ParMOO iterations to be
                performed by libEnsemble (default is unlimited).

            sim_max (int): The max number of simulation to be performed by
                libEnsemble (default is unlimited).

            wt_max (int): The max number of seconds that the simulation may
                run for (the default is 864000 secs, i.e., 10 days).

            profile (bool): Specifies whether to run libE with the profiler.

        """

        import sys
        from libensemble.libE import libE
        from libensemble.alloc_funcs.start_only_persistent \
            import only_persistent_gens as alloc_f
        from libensemble.tools import parse_args

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
            if self.moop.s == 0:
                raise ValueError("If 0 simulations are given, then iter_max" +
                                 "must be provided")
            iter_max = sim_max
        # Count the total search budget
        total_search_budget = 0
        for search in self.moop.searches:
            total_search_budget += search.budget
        total_sims_per_iter = len(self.moop.acquisitions) * self.moop.s
        # Count the total sims to exhaust iter_max if sim_max is None
        if sim_max is None:
            sim_max = total_search_budget + iter_max * total_sims_per_iter
        # libE only uses sim_max, so set it appropriately
        sim_max = min(sim_max,
                      total_search_budget + iter_max * total_sims_per_iter)
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

        # Create libEnsemble dictionaries
        nworkers, is_manager, libE_specs, _ = parse_args()
        if self.moop.use_names:
            libE_specs['final_fields'] = []
            for name in self.moop.des_names:
                libE_specs['final_fields'].append(name[0])
            for name in self.moop.sim_names:
                libE_specs['final_fields'].append(name[0])
            libE_specs['final_fields'].append('sim_name')
        else:
            libE_specs['final_fields'] = ['x', 'f', 'sim_name']
        # Set optional libE specs
        libE_specs['profile'] = profile

        if nworkers < 2:
            raise ValueError("Cannot run ParMOO + libE with less than 2 " +
                             "workers -- aborting...\n\n" +
                             "Note: this error could be caused by a " +
                             "failure to specify the communication mode " +
                             " (e.g., local comms or MPI)")

        # Get the max m for all SimGroups
        max_m = max(self.moop.m)

        # Set the input dictionaries
        if self.moop.use_names:
            x_type = self.moop.des_names.copy()
            x_type.append(('sim_name', 'a10'))
            f_type = self.moop.sim_names.copy()
            all_types = x_type.copy()
            for name in f_type:
                all_types.append(name)
            sim_specs = {'sim_f': self.moop_sim,
                         'in': [name[0] for name in x_type],
                         'out': f_type}

            gen_specs = {'gen_f': parmoo_persis_gen,
                         'persis_in': [name[0] for name in all_types],
                         'out': x_type,
                         'user': {}}
        else:
            sim_specs = {'sim_f': self.moop_sim,
                         'in': ['x', 'sim_name'],
                         'out': [('f', float, max_m)]}

            gen_specs = {'gen_f': parmoo_persis_gen,
                         'persis_in': ['x', 'sim_name', 'f'],
                         'out': [('x', float, self.moop.n),
                                 ('sim_name', int)],
                         'user': {}}

        alloc_specs = {'alloc_f': alloc_f, 'out': [('gen_informed', bool)]}

        persis_info = {}
        for i in range(nworkers + 1):
            persis_info[i] = {}
        persis_info[1]['moop'] = self.moop

        exit_criteria = {'sim_max': sim_max, 'wallclock_max': wt_max}

        # Perform the run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria,
                                    persis_info, alloc_specs, libE_specs)

        # When running with MPI, only the manager returns results
        if is_manager:
            self.moop = persis_info[1]['moop']
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

        return self.moop.getPF(format=format)

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

        return self.moop.getSimulationData(format=format)

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

        return self.moop.getObjectiveData(format=format)

    def save(self, filename="parmoo"):
        """ Serialize and save the MOOP object and all of its dependencies.

        Args:
            filename (str, optional): The filepath to serialized
                checkpointing file(s). Do not include file extensions,
                they will be appended automaically. May create
                several save files with extensions of this name, in order
                to recursively save dependencies objects. Defaults to
                the value "parmoo" (filename will be "parmoo.moop").

        """

        self.moop.save(filename=filename)
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

        self.moop.load(filename=filename)
        return

    def savedata(self, x, sx, s_name, filename="parmoo"):
        """ Save the current simulation database for this MOOP.

        Args:
            filename (str, optional): The filepath to the checkpointing
                file(s). Do not include file extensions, they will be
                appended automaically. Defaults to the value "parmoo"
                (filename will be "parmoo.simdb.json").

        """

        self.moop.savedata(x, sx, s_name, filename=filename)
        return
