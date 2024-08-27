libEnsemble Tutorial
====================

The following `libe_basic_ex.py <https://github.com/parmoo/parmoo/blob/main/examples/libe_basic_ex.py>`_ code is an example of basic ParMOO + libEnsemble usage from the
:doc:`Extras and Plugins <../extras>` section of the User Guide.

.. literalinclude:: ../../examples/libe_basic_ex.py
    :language: python

You can run the above script with MPI

.. code-block:: bash

    mpirun -np N python3 libe_basic_ex.py

or with Python's built-in multiprocessing module.

.. code-block:: bash

    python3 libe_basic_ex.py --comms local --nworkers {N+1}

The resulting output is shown below.

.. literalinclude:: ../../examples/libe_basic_ex.out
