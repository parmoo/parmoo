
.. image:: docs/img/logo-ParMOO.svg
    :align: center
    :alt: ParMOO

|

.. image:: https://img.shields.io/pypi/v/parmoo.svg?color=blue
    :target: https://pypi.org/project/parmoo

.. image:: https://github.com/parmoo/parmoo/actions/workflows/parmoo-ci.yml/badge.svg?/branch=main
    :target: https://github.com/parmoo/parmoo/actions

.. image:: https://readthedocs.org/projects/parmoo/badge/?maxAge=2592000
    :target: https://parmoo.readthedocs.org/en/latest/
    :alt: Documentation Status

|

ParMOO: Parallel MultiObjective Optimization in Python
======================================================

ParMOO is a parallel multiobjective optimization solver that seeks to
exploit simulation-based structure in objective and constraint functions.

To exploit structure, ParMOO models *simulations* separately from
*objectives* and *constraints*. In our language:

 * a **design variable** is an input to the problem, which we can directly
   control;
 * a **simulation** is an expensive or time-consuming process, including
   real-world experimentation, which is treated as a blackbox function
   of the design variables and evaluated sparingly;
 * an **objective** is an algebraic function of the design variables
   and/or simulation outputs, which we would like to optimize; and
 * a **constraint** is an algebraic function of the design variables
   and/or simulation outputs, which cannot exceed a specified bound.

To solve a multiobjective optimization problem (MOOP), we use surrogate
models of the simulation outputs, together with the algebraic definition of
the objectives and constraints.

.. figure:: docs/img/des-sim-obj-space.png
    :alt: Designs, simulations, and objectives
    :align: center

|

ParMOO is implemented in Python. In order to achieve scalable parallelism,
we use libEnsemble_ to distribute batches of simulation evaluations across
parallel resources.

Dependencies
------------

ParMOO has been tested on Unix/Linux and MacOS systems.

ParMOO's base has the following dependencies:

 * Python_ 3.6+
 * numpy_ -- for data structures and performant numerical linear algebra
 * scipy_ -- for scientific calculations needed for specific modules
 * pyDOE_ -- for generating experimental designs

Additional dependencies are needed to use the additional features in
``parmoo.extras``:

 * libEnsemble_ -- for managing parallel simulation evaluations

Installation
------------

The easiest way to install ParMOO, so that you can easily get the latest
version, is to clone it from our GitHub_ and pip install it in-place.
On Debian-based systems with a bash shell, this looks like

.. code-block:: bash

   git clone https://github.com/parmoo/parmoo
   cd parmoo
   pip install -e .

Testing
-------

If you have pytest_ with the pytest-cov_ plugin and flake8_ installed,
then you can test your installation.

.. code-block:: bash

   cd [parmoo base]
   parmoo/tests/run-tests.sh -cu[rl]

These tests are run regularly using GitHub Actions_.

Basic Usage
-----------

ParMOO uses numpy_ in an object-oriented design, based around the ``MOOP``
class. To get started, create a ``MOOP`` object.

.. code-block:: python

   from parmoo import MOOP
   from parmoo.optimizers import LocalGPS

   my_moop = MOOP(LocalGPS)

In the above example, ``LocalGPS`` is the class of optimizers that the
``my_moop`` will use to solve scalarized surrogate problems.

Next, add design variables to the problem as follows using the
``MOOP.addDesign(*args)`` method. In this example, we define one continuous
and one categorical design variable.

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
``MOOP.addSimulation`` method. In this example, we define a toy simulation
``sim_func(x)``.

.. code-block:: python

   import numpy as np
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

Now we can add objectives and constraints using ``MOOP.addObjective(*args)``
and ``MOOP.addConstraint(*args)``. In this example, there are 2 objectives
(each corresponding to a single simulation output) and one constraint.

.. code-block:: python

   # First objective just returns the first simulation output
   my_moop.addObjective({'name': "f1", 'obj_func': lambda x, s: s["MySim"][0]})
   # Second objective just returns the second simulation output
   my_moop.addObjective({'name': "f2", 'obj_func': lambda x, s: s["MySim"][1]})
   # Add a single constraint, that x[0] >= 0.1
   my_moop.addConstraint({'name': "c1",
                          'constraint': lambda x, s: 0.1 - x["x1"]})

Finally, we must add one or more acquisition functions using
``MOOP.addAcquisition(*args)``. These are used to scalarize the surrogate
problems. The number of acquisition functions typically determines the
number of simulation evaluations per batch. This is useful to know if you
are using a parallel solver.

.. code-block:: python

   from parmoo.acquisitions import UniformWeights

   # Add 3 acquisition functions
   for i in range(3):
      my_moop.addAcquisition({'acquisition': UniformWeights,
                              'hyperparams': {}})

Finally, the MOOP is solved using the ``MOOP.solve(budget)`` method, and the
results can be viewed using ``MOOP.getPF()`` method.

.. code-block:: python

   my_moop.solve(5) # Solve with 5 iterations of ParMOO algorithm
   results = my_moop.getPF() # Extract the results

Congratulations, you now know enough to get started solving MOOPs with
ParMOO!

Next steps:

 * Learn more about all that ParMOO has to offer (including saving and
   checkpointing, INFO-level logging, advanced problem definitions, and
   different surrogate and solver options) at our ReadTheDocs_ page.
 * Explore the advanced examples (including a ``libEnsemble`` example)
   in the ``examples`` directory.
 * Install libEnsemble_ and get started solving MOOPs in parallel.

Resources
---------

For more information, e-mail questions to:

 * ``parmoo@mcs.anl.gov``

Our full documentation is hosted on:

 * ReadTheDocs_

Please read our LICENSE_ and CONTRIBUTING_ files.


.. _Actions: https://github.com/parmoo/parmoo/actions
.. _CONTRIBUTING: https://github.com/parmoo/parmoo/blob/main/CONTRIBUTING.rst
.. _flake8: https://flake8.pycqa.org/en/latest
.. _GitHub: https://github.com/parmoo/parmoo
.. _libEnsemble: https://github.com/Libensemble/libensemble
.. _LICENSE: https://github.com/parmoo/parmoo/blob/main/LICENSE
.. _numpy: https://numpy.org
.. _pyDOE: https://pythonhosted.org/pyDOE
.. _pytest: https://docs.pytest.org/en/7.0.x
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest
.. _Python: https://www.python.org/downloads
.. _ReadTheDocs: https://parmoo.readthedocs.org
.. _scipy: https://scipy.org