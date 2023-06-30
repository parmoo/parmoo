Acquisition Functions
---------------------

Add one of these to your ``MOOP`` object to generate additional scalarizations
per iteration.
In general, ParMOO typically generates one candidate solution per simulation
per acquisition function, so the number of acquisition functions determines
the number of candidate simulations evaluated (in parallel) per
iteration/batch.

.. code-block:: python

    from parmoo import acquisitions

Current options are:

.. toctree::
    :maxdepth: 1
    :caption: Modules:

Weighted Sum Methods
~~~~~~~~~~~~~~~~~~~~

.. automodule:: acquisitions.weighted_sum
..    :members: acquisitions/weighted_sum

.. autoclass:: UniformWeights
   :member-order: bysource
   :members:

   .. automethod:: __init__

.. autoclass:: FixedWeights
   :member-order: bysource
   :members:

   .. automethod:: __init__

Epsilon Constraint Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: acquisitions.epsilon_constraint
..    :members: acquisitions/epsilon_constraint

.. autoclass:: RandomConstraint
   :member-order: bysource
   :members:

   .. automethod:: __init__

.. autoclass:: EI_RandomConstraint
   :member-order: bysource
   :members:

   .. automethod:: __init__
