Extras and Plugins
==================

Solving on Parallel Systems Using libEnsemble
---------------------------------------------

`libEnsemble <https://github.com/Libensemble/libensemble>`_ is
a Python library to "coordinate concurrent evaluation of dynamic
ensembles of calculations."
Read more about libEnsemble by visiting the
`libEnsemble documentation <https://libensemble.readthedocs.io/en/main/>`_.

The :mod:`libE_MOOP <extras.libe.libE_MOOP>` class is used to solve
MOOPs using libEnsemble.
Although it does not directly extend the :mod:`MOOP <moop.MOOP>` class
for technical reasons, the :mod:`libE_MOOP <extras.libe.libE_MOOP>` class functions
similarly to  a :mod:`MOOP <moop.MOOP>` extension.

To create an instance of the :mod:`libE_MOOP <extras.libe.libE_MOOP>` class,
import it from the :mod:`extras.libe` module and then create a MOOP, just as
you normally would.
The :meth:`solve() <extras.libe.libE_MOOP.solve>` method has been redefined
to create a libEnsemble
`Persistent Generator <https://libensemble.readthedocs.io/en/main/function_guides/generator.html>`_
function, which libEnsemble can call to generate batches of simulation
evaluations, which it will distribute over available resources.

Below we reproduce the example from the :doc:`Quickstart <quickstart>`
guide, using a :mod:`libE_MOOP <extras.libe.libE_MOOP>` object.
Note that for technical reasons, the libEnsemble interface does not
support the usage of Python ``lambda`` functions, so each of the objectives
and constraints is explicitly defined.

.. literalinclude:: ../examples/libe_basic_ex.py
    :language: python

To run a ParMOO/libEnsemble script, first make sure that libEnsemble
is installed.
You can find instructions on how to do so under libEnsemble's
`Advanced Installation <https://libensemble.readthedocs.io/en/main/advanced_installation.html>`_
documentation.

Next, run libEnsemble as described in the
`Running libEnsemble <https://libensemble.readthedocs.io/en/main/running_libE.html>`_
section.
Common methods of running a libEnsemble script are with MPI

.. code-block:: bash

    mpirun -np N python myscript.py

and with Python's built-in multiprocessing module.

.. code-block:: bash

    python myscript.py --comms local --nworkers N

The result from running the example is shown below.

.. literalinclude:: ../examples/libe_basic_ex.out
