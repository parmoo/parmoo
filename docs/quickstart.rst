
.. _Quickstart:

Quickstart
==========

Dependencies
------------

ParMOO has been tested on Unix/Linux and MacOS systems.

ParMOO's base has the following dependencies:
 * `Python 3.6+ <https://www.python.org/downloads/>`_
 * `numpy <https://numpy.org/>`_ -- for data structures and performant
   numerical linear algebra
 * `scipy <https://scipy.org>`_ -- for important scientific calculations
   (such as optimization solvers) needed by certain modules
 * `pyDOE <https://pythonhosted.org/pyDOE/>`_ -- for generating experimental
   designs

Additional dependencies are needed to use the additional features in
``parmoo.extras``.
 * `libEnsemble <https://github.com/Libensemble/libensemble>`_
   -- for managing parallel simulation evaluations

Installation
------------

The Python files in ParMOO are referenced relative to the base directory.
To install ParMOO, clone it and add the base to your PYTHONPATH environment
variable. On Debian-based systems with a bash shell, this looks like

.. code-block:: bash

   git clone [parmoo url]
   cd [parmoo base]
   export PYTHONPATH=$PYTHONPATH:`pwd`

For detailed instructions, see :doc:`install`.

Testing
-------

If you have `pytest <https://docs.pytest.org/en/7.0.x/>`_ with the
`pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ plugin
and `flake8 <https://flake8.pycqa.org/en/latest/>`_ installed,
then you can test your installation by running the
``run-tests.sh`` script in the ``tests`` directory.

.. code-block:: bash

   cd [parmoo base]/tests
   ./run-tests.sh

Basic Usage
-----------

ParMOO uses `numpy <https://numpy.org/>`_ in an object oriented design, based around the
:mod:`MOOP <moop.MOOP>` class.
To get started, create a :mod:`MOOP <moop.MOOP>` object, using the
:meth:`constructor <moop.MOOP.__init__>`.

.. code-block:: python

   from parmoo import MOOP
   from parmoo.optimizers import LocalGPS

   my_moop = MOOP(LocalGPS)

In the above example,
:mod:`optimizers.LocalGPS <optimizers.gps_search.LocalGPS>`
is the class of optimizers
that the ``my_moop`` will use to solve scalarized surrogate problems.

Next, add design variables to the problem as follows using the
:meth:`MOOP.addDesign(*args) <moop.MOOP.addDesign>` method.
In this example, we define one continuous and one categorical variable.

.. code-block:: python

   # Add a single continuous design variable in the range [0.0, 1.0]
   my_moop.addDesign({'name': "x1", # optional, name
                      'des_type': "continuous", # optional, type of variable
                      'lb': 0.0, # required, lower bound
                      'ub': 1.0, # required, upper bound
                      'tol': 1.0e-8 # optional tolerance
                     })
   # Add a second categorical design variable with 3 levels
   my_moop.addDesign({'name': "x2", # optional, name
                      'des_type': "categorical", # required, type of variable
                      'levels': 3 # required, number of categories
                     })

Next, add simulations to the problem as follows using the
:meth:`MOOP.addSimulation(*args) <moop.MOOP.addSimulation>` method.
In this example, we define a toy simulation ``sim_func(x)``.

.. code-block:: python

   from parmoo.searches import LatinHypercube
   from parmoo.surrogates import GaussRBF

   # Define a toy simulation for the problem, whose outputs are quadratic
   def sim_func(x):
      if x["x2"] == 0:
         return np.array([(x["x1"] - 0.2) ** 2, (x["x1"] - 0.8) ** 2])
      else:
         return np.array([99.9, 99.9])
   # Add the simulation to the problem
   my_moop.addSimulation({'name': "MySim", # Optional name for this simulation
                          'm': 2, # This simulation has 2 outputs
                          'sim_func': sim_func, # Our sample sim from above
                          'search': LatinHypercube, # Use a LH search
                          'surrogate': GaussRBF, # Use a Gaussian RBF surrogate
                          'hyperparams': {}, # Hyperparams passed to internals
                          'sim_db': { # Optional dict of precomputed points
                                     'search_budget': 10 # Set search budget
                                    },
                         })

Now we can add objectives and contraints using
:meth:`MOOP.addObjective(*args) <moop.MOOP.addObjective>` and
:meth:`MOOP.addConstraint(*args) <moop.MOOP.addConstraint>`.
In this example, there are 2 objectives (each mapping to a single
simulation output) and one constraint.

.. code-block:: python

   # First objective just returns the first simulation output
   my_moop.addObjective({'name': "f1", 'obj_func': lambda x, s: s["MySim"][0]})
   # Second objective just returns the second simulation output
   my_moop.addObjective({'name': "f2", 'obj_func': lambda x, s: s["MySim"][1]})
   # Add a single constraint, that x[0] >= 0.1
   my_moop.addConstraint({'name': "c1",
                          'constraint': lambda x, s: 0.1 - x["x1"]})

Finally, we must add one or more acquisition functions using
:meth:`MOOP.addAcquisition(*args) <moop.MOOP.addAcquisition>`.
These are used to scalarize the surrogate problems.
The number of acquisition functions
typically determines the number of parallel function evaluations.

.. code-block:: python

   from parmoo.acquisitions import UniformWeights

   # Add 3 acquisition functions
   for i in range(3):
      my_moop.addAcquisition({'acquisition': UniformWeights,
                              'hyperparams': {}})

Finally, the MOOP is solved using the
:meth:`MOOP.solve(k) <moop.MOOP.solve>` method, and the
results can be viewed using
:meth:`MOOP.getPF() <moop.MOOP.getPF>`.

.. code-block:: python

   my_moop.solve(5) # Solve with 5 iterations of ParMOO algorithm
   results = my_moop.getPF() # Extract the results

Congratulations, you now know enough to get started solving MOOPs!

Next steps:

 * If you want to take advantage of all that ParMOO has to offer, 
   please see :doc:`Writing a ParMOO Script <how-to-write>`.
 * If you would like more information on multiobjective optimization
   terminology and ParMOO's methodology, see the
   :doc:`Learn About MOOPs <about>` page.

Minimal Working Example
-----------------------

Putting it all together, we get the following minimal working example.

.. literalinclude:: ../examples/quickstart.py
    :language: python

The above code produces the output below.

.. literalinclude:: ../examples/quickstart.out

