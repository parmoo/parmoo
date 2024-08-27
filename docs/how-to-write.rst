Write a ParMOO Script
=====================

The MOOP class
--------------

The :mod:`MOOP <moop.MOOP>` class is the fundamental data structure in
ParMOO.

..
    _The following has been commented out as the corresponding UML is obsolete:
    |
    |Below is a UML diagram showing the key public methods and
    |dependencies.
    |
    |.. only:: html
    |
    |    .. figure:: img/moop-uml.svg
    |        :alt: ParMOO UML Diagram
    |        :align: center
    |
    |    |
    |
    |.. only:: latex
    |
    |    .. figure:: img/moop-uml.png
    |        :alt: ParMOO UML Diagram
    |        :align: center
    |
    |    |

To create an instance of the :mod:`MOOP <moop.MOOP>` class,
use the :meth:`constructor <moop.MOOP.__init__>`.

.. code-block:: python

    from parmoo import MOOP
    moop = MOOP(optimizer, hyperparams=hp)

In the above code snippet, ``optimizer`` should be an implementation
of the :mod:`SurrogateOptimizer <structs.SurrogateOptimizer>`
Abstract-Base-Class (ABC),
and the optional input ``hp`` is a dictionary of hyperparameters for the
``optimizer`` object.
The ``optimizer`` is the surrogate optimization problem solver that will be used
to generate candidate solutions for the MOOP.
The choice of surrogate optimizer determines what information
will be required when defining each objective and constraint.

 * If you use a derivative-free technique, such as
   :meth:`GlobalSurrogate_PS <optimizers.pattern_search.GlobalSurrogate_PS>`,
   then you do not need to provide derivative information for your
   objective or constraint functions.
 * If you use a derivative-based technique, such as
   :meth:`GlobalSurrogate_BFGS <optimizers.lbfgsb.GlobalSurrogate_BFGS>`,
   then you need to provide an additional input
   to your objectives and constraint functions, which can be
   set to evaluate their derivatives with respect to design inputs
   and simulation outputs.

**As of version 0.4.0:** to fix the random seed, ParMOO no longer uses
the global numpy random seed.
Instead, pass an integer or ``numpy.random.Generator`` object using the key
``hp["np_random_gen"]`` when creating the MOOP.
It will automatically be passed on to all subclasses and components.

To avoid issues, it is best to define your MOOP in the following order,
but as of version 0.4.0, this is no longer a requirement.

 1. Add design variables using
    :meth:`MOOP.addDesign(*args) <moop.MOOP.addDesign>`.
 2. Add simulations using
    :meth:`MOOP.addSimulation(*args) <moop.MOOP.addSimulation>`.
 3. Add objectives using
    :meth:`MOOP.addObjective(*args) <moop.MOOP.addObjective>`.
 4. Add constraints using
    :meth:`MOOP.addConstraint(*args) <moop.MOOP.addConstraint>`.
 5. Add acquisitions using
    :meth:`MOOP.addAcquisition(*args) <moop.MOOP.addAcquisition>`.

All of these methods accept one or more ``args``, each of which is a
dictionary, as detailed in the corresponding sections below.

.. _naming:

The name Key and ParMOO Input/Output Types
------------------------------------------

Each of the design, simulation, objective, and constraint dictionaries
may contain an optional ``name`` key.
When omitted, the ``name`` of the design variables,
simulations, objectives, and constraints
default to ``{x|sim|f|c}i``,
where ``x`` is a design variable,
``sim`` is for a simulation, ``f`` is for an objective,
``c`` is for a constraint, and ``i=1,2,...``
is determined by the order in which each was added.

For example, if you add 3 simulations, then they will automatically
be named ``sim1``, ``sim2``, and ``sim3`` unless a different name,
was specified for one or more by including the ``name`` key.
Similarly, design variables are named ``x1``, ``x2``, ...;
objectives are named ``f1``, ``f2``, ...; and
constraints are named ``c1``, ``c2``, ....

Use of a repeated ``name`` will result in an error.

The inputs to any user-defined functions will be passed as Python dictionaries
with keys corresponding to the ``name`` keys used above.

After solving, ParMOO formats its output in a numpy structured array, using the
given or automatically assigned ``name`` keys to specify the name for
each field.

As of version 0.4.0, this operation mode is required and all other naming
conventions are no longer supported.

After adding all design variables, simulations, objectives, and constraints
to the MOOP, you can check the numpy dtype for each of these by using

 * :meth:`MOOP.getDesignType() <moop.MOOP.getDesignType>`,
 * :meth:`MOOP.getSimulationType() <moop.MOOP.getSimulationType>`,
 * :meth:`MOOP.getObjectiveType() <moop.MOOP.getObjectiveType>`, and
 * :meth:`MOOP.getConstraintType() <moop.MOOP.getConstraintType>`.

.. literalinclude:: ../examples/named_var_ex.py
    :language: python

The result is the following.

.. literalinclude:: ../examples/named_var_ex.out

Working with Unnamed Outputs [obsolete]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In older versions of ParMOO, it was possible to omit the ``name`` keys
and use ``numpy.ndarray`` structures as inputs/outputs to ParMOO.

In an effort to simplify workflow, improve maintainability, and optimize
iteration times when working with ``jax``, this feature has been removed
and all ParMOO scripts **must** use named inputs/outputs as of
version 0.4.0.

Adding Design Variables
-----------------------

Design variables are added to your :mod:`MOOP <moop.MOOP>` object
using the :meth:`addDesign(*args) <moop.MOOP.addDesign>` method.
ParMOO currently supports
several types of design variables:

 * ``continuous`` (or ``real`` or ``cont``),
 * ``integer`` (or ``int``),
 * ``categorical`` (or ``cat``),
 * ``custom``,
 * ``raw`` -- not recommended, for advanced users only.

To add a continuous variable, use the following format.

.. code-block:: python

    # Add a continuous design variable
    moop.addDesign({'name': "MyContVar", # optional
                    'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0,
                    'des_tol': 1.0e-8})

|

 * Note that when the ``des_type`` key is omitted, its value defaults to
   ``continuous.``
 * For continuous design variables, both a lower (``lb``) and upper (``ub``)
   bound must be specified.
   These bounds are *hard constraints*, meaning that no simulations
   or objectives will be evaluated outside of these bounds.
 * The optional key ``des_tol`` specifies a minimum step size between
   values for this design variable (default value is :math:`10^{-8}`).
   For this design variable, any two values that are closer than ``des_tol``
   will be treated as exactly equal.

To add an integer design variable, use the following format.

.. code-block:: python

    # Add an integer design variable
    moop.addDesign({'name': "MyIntVar", # optional
                    'des_type': "integer",
                    'lb': 0,
                    'ub': 100})

|

 * The ``lb`` and ``ub`` keys must be integer-valued, and serve
   the same purpose as with continuous design variables.

To add a categorical design variable, use the following format.

.. code-block:: python

    # Add a categorical design variable
    moop.addDesign({'name': "MyCatVar", # optional
                    'des_type': "categorical",
                    'levels': 3})

|

 * The ``levels`` key is either an integer specifying the
   number of categories taken on by this design variable
   (ParMOO will index these levels by :math:`0, 1, \ldots, \text{levels}-1`)
   or a list of level IDs specifying the ID for each category
   (ParMOO will use these names for the levels, e.g.,
   ``["first cat", "second cat", ... ]`` or ``[-1, 0, 1]``).

**Note** because ``jax`` cannot ``jit`` functions with
string-valued inputs and outputs, as of version 0.4.0 it is
strongly recommended to only use integer-valued level names when specifying
the ``levels`` key using the list syntax.
While it is still possible to specify string-valued category IDs, doing
so will cause ``jax.jit(...)`` to fail in several places, which may ultimately
increase iteration times by up to 10x.

To add a custom design variable, use the following format.

.. code-block:: python

    # Add a custom design variable
    moop.addDesign({'name': "MyCustomVar", # optional
                    'des_type': "custom",
                    'embedder': my_embedding_func,
                    })

|

 * The ``embedder`` key should be an instance of the
   :class:`Embedder <structs.Embedder>` class,
   defining a user-provided embedding.
   **Warning:** if jax cannot jit the ``embed`` and ``extract`` methods
   for this class, ParMOO's iterations may become extremely slow.

To add a raw design variable, use the following format. Please note that
raw design variables are not recommended, and one will typically need to
write custom ``search``, ``surrogate``, ``optimizer``, and ``acquisition``
functions/classes to accommodate a raw variable.
This feature is only included to allow flexibility for expert users.

.. code-block:: python

    # Add a raw design variable
    moop.addDesign({'name': "MyRawVar", # optional
                    'des_type': "raw"})

|

Note that for every MOOP, at least one design variable is required
before solving.

Adding Simulations
------------------

Before you can add a simulation to your :mod:`MOOP <moop.MOOP>`, you must
define the simulation function.

*The simulation function can be either a Python function or a callable object
whose __call__ method matches the signature below.*

The simulation should take a single
Python dictionary as input, whose keys match the design variable
names.
The simulation function returns a ``numpy.ndarray`` (or array-like object)
containing the simulation output(s).

For example, with three design variables named ``x1``, ``x2``, and
``x3``, you might define the quadratic
:math:`{\bf S}({\bf x}) = \|{\bf x}\|^2` as follows.

.. code-block:: python

    def quadratic_sim(x):
        return np.array([x["x1"] ** 2 + x["x2"] ** 2 + x["x3"] ** 2])

To add your simulation to the :mod:`MOOP <moop.MOOP>` object,
use the :meth:`addSimulation(*args) <moop.MOOP.addSimulation>` method.

.. code-block:: python

    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF

    moop.addSimulation({'name': "MySim", # optional
                        'm': 1, # number of outputs
                        'sim_func': quadratic_sim, # simulation function
                        'search': LatinHypercube, # search technique
                        'surrogate': GaussRBF, # surrogate model
                        'hyperparams': {'search_budget': 20}})

In the above example,
 * ``name`` is used as described in the section on
   :ref:`name key <naming>`;
 * ``m`` specifies the number of outputs for this simulation;
 * ``sim_func`` is given a reference to the simulation function;
 * ``search`` specifies the :mod:`GlobalSearch <structs.GlobalSearch>`
   that you will use when generating data for this particular simulation;
 * ``surrogate`` specifies the class of
   :mod:`SurrogateFunction <structs.SurrogateFunction>`
   that you will use to model this particular simulation's output;
 * ``hyperparams`` is a dictionary of hyperparameter values that will be
   passed to the surrogate and search technique objects.
   One particularly important key in the ``hyperparams`` dictionary is the
   ``search_budget`` key, which specifies how many simulation evaluations
   should be used during the initial search phase.

If you wish, you may create a MOOP without any simulations.

Adding Objectives
-----------------

Objectives are algebraic functions of your design variables and
simulation outputs.
*ParMOO always minimizes objectives.*
If you would like to maximize instead, re-define the problem by minimizing
the negative-value of your objective.
We provide a :mod:`library of common built-in objectives <objectives.obj_lib>`,
which can do this automatically.

Just like with simulation functions, ParMOO accepts either a Python
function or a callable object for each objective.
Make sure you match the expected signature, which depends on your choice of
:ref:`name keys <naming>`.

In particular, your objective function should accept two
Python dictionaries and return a single scalar output.
The following objective minimizes the sum of all outputs of the
simulation output named ``MySim.``

.. code-block:: python

    def min_sim(x, sim):
        return sum(sim["MySim"])

Similarly, the following objective minimizes the squared value of the
design variable named ``MyDes``.

.. code-block:: python

    def min_des(x, sim):
        return x["MyDes"] ** 2

**Note:** The following has been changed as of version 0.4.0
in order to offer better support for ``jax.jit()`` compilation.

If you are using a gradient-based
:mod:`SurrogateOptimizer <structs.SurrogateOptimizer>`,
then you are required to supply an additional function for evaluating
the gradient of your objective with respect to both ``x`` and ``sim`` inputs.

Note that for categorical variables ParMOO does not use the partial
derivatives given here, and it is acceptable to fill these keys
with a garbage value or leave them uninitialized.

Modifying the above two objectives to support derivative-based solvers,
we get the following.

.. code-block:: python

    def min_sim_grad(x, sim):
        dx = {}
        ds = {}
        for key in x:
            dx[key] = 0.0
        for key in sim:
            ds[key] = np.zeros(sim[key].size)
        ds["MySim"] = 1.0
        return dx, ds

    def min_des_grad(x, sim):
        for key in x:
            dx[key] = 0.0
        for key in sim:
            ds[key] = np.zeros(sim[key].size)
        dx["MyDes"] = 2.0 * x["MyDes"]
        return dx, ds

For a full example showing how to solve a MOOP using a derivative-based
solver, see :ref:`Solving a MOOP with Derivative-Based Solvers <advanced_ex>`
in :doc:`Basic Tutorials <tutorials/basic-tutorials>`.

To add the objective(s), use the
:mod:`MOOP.addObjective(*args) <moop.MOOP.addObjective>` method.

.. code-block:: python

    moop.addObjective({'name': "Min MySim",
                       'obj_func': min_sim,
                       # below is only needed for gradient-based solvers
                       'obj_grad': min_sim_grad
                       })

    moop.addObjective({'name': "Min MyDes",
                       'obj_func': min_des,
                       # below is only needed for gradient-based solvers
                       'obj_grad': min_des_grad
                       })

Note that for every MOOP, at least one objective is required before solving.

Adding Constraints
------------------

Adding constraints is similar to adding objectives.
The main difference is in how ParMOO treats constraint functions.
Although ParMOO may evaluate infeasible design points along the way,
ParMOO will search for solutions where all constraints are less than
or equal to zero.

For example, to add the constraint that the simulation ``MySim`` must
have output greater than or equal to 0 and that the design variable
``MyDes`` must be less than or equal to 0.9, you would define the
following constraint functions.

.. code-block:: python

    def sim_constraint(x, sim):
        return -1.0 * sim["MySim"]

    def des_constraint(x, sim):
        return x["MyDes"] - 0.9

As with objectives, if you want to use a gradient-based
:mod:`SurrogateOptimizer <structs.SurrogateOptimizer>`, you must
modify the above constraint functions as follows.

.. code-block:: python

    def sim_constraint_grad(x, sim):
        dx, ds = {}, {}
        for key in x:
            dx[key] = 0.0
        for key in sim:
            ds[key] = np.zeros(sim[key].size)
        ds["MySim"] = -1.0
        return dx, ds

    def des_constraint_grad(x, sim):
        dx, ds = {}, {}
        for key in x:
            dx[key] = 0.0
        for key in sim:
            ds[key] = np.zeros(sim[key].size)
        dx["MyDes"] = 1.0
        return dx, ds

To add the constraint(s), use the
:mod:`MOOP.addConstraint(*args) <moop.MOOP.addConstraint>` method.

.. code-block:: python

    moop.addConstraint({'name': "Constrain MySim",
                        'con_func': sim_constraint,
                        # below is only needed for gradient-based solvers
                        'con_grad': sim_constraint_grad
                        })

    moop.addConstraint({'name': "Constrain MyDes",
                        'con_func': des_constraint,
                        # below is only needed for gradient-based solvers
                        'con_grad': des_constraint_grad
                        })

You are not required to add any constraints of this form to your MOOP
before solving.

Adding Acquisitions
-------------------

After you have added all of the design variables, simulations, objectives,
and constraints to your MOOP, you must add one or more acquisitions
using the :meth:`MOOP.addAcquisition(*args) <moop.MOOP.addAcquisition>`
method.

.. code-block:: python

    from parmoo.acquisitions import RandomConstraint, FixedWeights

    moop.addAcquisition({'acquisition': RandomConstraint})
    moop.addAcquisition({'acquisition': FixedWeights,
                         'hyperparams': {'weights': np.array([0.5, 0.5])}})

The acquisition dictionary may contain two keys:
 * ``acquisition`` (required) specifies one
   :mod:`AcquisitionFunction <structs.AcquisitionFunction>`
   that you would like to use for this problem; and
 * ``hyperparams`` (optional) specifies a dictionary of hyperparameter
   values that are used by the specified
   :mod:`AcquisitionFunction <structs.AcquisitionFunction>`.

The number of acquisitions added determines the batch size for each of
ParMOO's batches of simulation evaluations (which could be done in parallel).
**In general, if there are q acquisition functions and s simulations,
then ParMOO will generate batches of q*s simulations**.
In other words, each simulation is evaluated once per acquisition function in
each iteration of ParMOO's algorithm.

Using a Precomputed Simulation Database
---------------------------------------

If you would like to specify a precomputed database, use the
:meth:`MOOP.updateSimDb(x, sx, s_name) <moop.MOOP.updateSimDb>`
method to add all simulation data into ParMOO's database after creating
your MOOP but before solving.
Be careful not to add duplicate points, because these could cause numerical
issues when fitting surrogate models.

Before doing so, you must first call the
:meth:`MOOP.compile() <moop.MOOP.compile>` method to "finalize" the
definition of your MOOP.
This can only be done once, so be sure that you are done defining the MOOP
before doing so.
If you turn on logging first (see below), ParMOO will attempt to jit all
user-defined functions at this stage and log a warning to alert users if
any methods failed to compile at this stage.
Failure to compile does not prevent ParMOO from running, but may increase
iteration times by 10x.

Note that the ``MOOP.compile()`` command is run automatically when calling
``MOOP.solve()`` (below), so you only needed to compile manually when
adding precomputed simulation evaluations.

.. literalinclude:: ../examples/precomputed_data.py
    :language: python

The output of the above code is shown below.

.. literalinclude:: ../examples/precomputed_data.out

Logging and Checkpointing
-------------------------

When solving large or expensive problems, it is often a good idea to
activate ParMOO's logging and/or checkpointing features.

Logging
~~~~~~~

For diagnostics, ParMOO logs its progress at the ``logging.INFO`` level.
To display these log messages, turn on Python's ``INFO``-level logging.

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.INFO)

If you would like to also print a formatted timestamp, use the Python
logger's built-in formatting options.

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.INFO,
                        [format='%(asctime)s %(levelname)-8s %(message)s',
                         datefmt='%Y-%m-%d %H:%M:%S'])

Be aware that when using ParMOO together with
`libEnsemble <https://github.com/Libensemble/libensemble>`_,
``libE`` already comes with its own logging tools, which are recommended,
and ParMOO's logging tools will not work.

Checkpointing
~~~~~~~~~~~~~

A ParMOO can be run with checkpointing turned on, so that your MOOP can be
paused and resumed later, and your simulation data can be recovered after
a crash.
Checkpointing is off by default.
To turn it on, use the method:

 * :meth:`setCheckpoint(checkpoint, [checkpoint_data, filename]) <moop.MOOP.setCheckpoint>`

.. code-block:: python

    moop.setCheckpoint(True, checkpoint_data=True, filename="parmoo")

The first argument tells ParMOO to save its internal class attributes and
databases, so that they can be reloaded in the future.
In the above example, this save data will be written to a file in the calling
directory, with the name ``parmoo.moop``.

In order to save the problem definition, ParMOO needs to store information
for reloading all of your functions.
For this to work:

 * All functions (such as simulation functions, objective functions, and
   constraint functions) are defined in the global scope;
 * All modules are reloaded before attempting to recover a previously-saved
   MOOP object (by calling the :meth:`load(filename) <moop.MOOP.load>` method);
 * ParMOO cannot reload ``lambda`` functions. Use only regular functions
   and callable objects when checkpointing.

If the option argument ``checkpoint_data`` is set to ``True`` (default),
the ParMOO will also save a second copy of all simulation evaluations in a
human-readable JSON file in the same directory, with the name
``parmoo.simdb.json``.
This file is not used by ParMOO, it is only provided for user-convenience.

Reloading After Crash or Early Stop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a crash or early termination, reload the saved ``.moop`` file to resume.
**Make sure that you first import any external modules and redefine any
functions that are needed by ParMOO (with the exact same signatures).**

.. code-block:: python

    from parmoo import MOOP
    from optimizers import [optimizer]

    # Create a new MOOP object
    moop = MOOP([optimizer])
    # Reload the old problem
    moop.load(filename="parmoo") # Use your saved file name, omitting ".moop"

Then resume your solve with an increased budget.

.. code-block:: python

    # Resume solve with increased budget
    moop.solve(6)

Example
~~~~~~~

The example below shows how the `Quickstart demo <quickstart_ex>`_ can be
modified to use logging and checkpointing, including an example of how
to load a MOOP from a saved checkpoint file and resume running.

.. literalinclude:: ../examples/checkpointing.py
    :language: python

The result is the following.

.. literalinclude:: ../examples/checkpointing.out


Methods for Solving
-------------------

Once you have finished creating your :mod:`MOOP <moop.MOOP>` object and
adding all design variables, simulations, objectives, constraints,
and acquisitions, you are ready to solve your problem.

The easiest way to solve is by using :meth:`MOOP.solve(k) <moop.MOOP.solve>`.
Here, ``k`` is the number of iterations of ParMOO's algorithm
that you would like to perform.
Note that a value of ``k=0`` is legal, and will result in ParMOO
generating and evaluating an experimental design and fitting its surrogates,
without ever attempting to solve a single scalarized surrogate problems.

.. code-block:: python

    # Evaluate an experimental design, then performing 5 iterations
    moop.solve(5)

Note that the above command will perform all simulation evaluations serially.
To generate a batch of simulations that you could evaluate in parallel,
use :meth:`MOOP.iterate(k) <moop.MOOP.iterate>`, where ``k`` is the iteration
index.
You can let ParMOO handle the simulation evaluations with
:meth:`MOOP.evaluateSimulation(x, s_name) <moop.MOOP.evaluateSimulation>`,
or you can evaluate the simulations yourself and add them to the simulation
database using
:meth:`MOOP.updateSimDb(x, sx, s_name) <moop.MOOP.updateSimDb>`.
Afterward, call :meth:`MOOP.updateAll(k, batch) <moop.MOOP.updateAll>` to
update the surrogate models and objective database.

.. code-block:: python

    # Do 5 iterations letting ParMOO handle simulation evaluation
    # Note that the i=0 iteration will just generate an experimental design
    for i in range(5):
        # Get batch
        batch = moop.iterate(i)
        # Let ParMOO evaluate design point x for simulation s_name
        for (x, s_name) in batch:
            moop.evaluateSimulation(x, s_name)
        # Update ParMOO models
        moop.updateAll(i, batch)

or

.. code-block:: python

    # Solve another MOOP, doing simulation evaluation manually
    for i in range(5):
        # Get batch
        batch = moop.iterate(i)
        # User evaluates design point x for simulation s_name
        for (x, s_name) in batch:
            ### User code to evaluate x with sim["s_name"] goes HERE ###
            ### Store results in variable sx ###
            moop.updateSimDb(x, sx, s_name)
        # Update ParMOO models
        moop.updateAll(i, batch)

Additional ParMOO solver execution paradigms (including those where ParMOO
will handle parallel execution on the user's behalf) are included under
:doc:`Additional ParMOO Plugins and Features <extras>`.

Viewing Your Results
--------------------

After solving the MOOP, you can view the results using
:meth:`MOOP.getPF() <moop.MOOP.getPF>`.

.. code-block:: python

    soln = moop.getPF()

The output format defaults to a numpy structured array.
However, you can change it to a pandas dataframe using the optional
``format`` argument.

.. code-block:: python

    soln = moop.getPF(format="pandas")

To get the full simulation and objective databases, you can also use
:meth:`MOOP.getSimulationData() <moop.MOOP.getSimulationData>`
and
:meth:`MOOP.getObjectiveData() <moop.MOOP.getObjectiveData>`.

.. code-block:: python

    sim_db = moop.getSimulationData()
    obj_db = moop.getObjectiveData()

To understand the format of these outputs, please revisit the section on
:ref:`The name Key and ParMOO Output Types <naming>`.

Finally, if you have installed ParMOO with its extra dependencies
(see the :ref:`Advanced Installation <install>`),
then you can visualize your results using any of the
:func:`viz.scatter() <viz.plot.scatter>`,
:func:`viz.parallel_coordinates() <viz.plot.parallel_coordinates>`, or
:func:`viz.radar() <viz.plot.radar>` functions.

.. code-block:: python

    from parmoo.viz import scatter

    scatter(moop)

Note that these plots are interactive and will render in a Dash app
hosted locally on your computer.
There are known issues when using the Chrome browser.

For more information, view the complete
:mod:`viz API page <viz.plot>`.

Built-in and Custom Components
------------------------------

By now you can see that the performance of ParMOO is determined
by your choices of

 * :mod:`AcquisitionFunction <structs.AcquisitionFunction>`,
 * :mod:`GlobalSearch <structs.GlobalSearch>`,
 * :mod:`SurrogateFunction <structs.SurrogateFunction>`, and
 * :mod:`SurrogateOptimizer <structs.SurrogateOptimizer>`.

You can find the current options for each of these in the following modules.

 * :doc:`parmoo.acquisitions <modules/acquisitions>`
 * :doc:`parmoo.searches <modules/searches>`
 * :doc:`parmoo.surrogates <modules/surrogates>`
 * :doc:`parmoo.optimizers <modules/optimizers>`

You can also create your own custom implementations for each of the above,
by implementing one of the abstract base classes in :mod:`structs`.

