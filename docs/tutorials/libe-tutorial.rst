libEnsemble Tutorial
====================

This is an example of basic ParMOO + libEnsemble usage from the
:doc:`Extras and Plugins <../extras>` section of the User Guide.

.. literalinclude:: ../../examples/libe_basic_ex.py
    :language: python

You can run the above script with MPI

.. code-block:: bash

    mpirun -np N python myscript.py

or with Python's built-in multiprocessing module.

.. code-block:: bash

    python myscript.py --comms local --nworkers N

The resulting output is shown below.

.. literalinclude:: ../../examples/libe_basic_ex.out
