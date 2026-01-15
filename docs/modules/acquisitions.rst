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

It is also possible to define your own custom acquisition function by importing
and extending the ABC

.. code-block:: python

    from parmoo.acquisitions.acquisition_function import AcquisitionFunction

The ``AcquisitionFunction`` ABC and current options from the existing
acquisition function library are defined below.

AcquisitionFunction (ABC)
~~~~~~~~~~~~~~~~~~~~~~~~~

The ABC for ``AcquisitionFunction`` base class can be extended by developers
looking to implement custom multiobjective acquisition functions.

.. automodule:: acquisitions.acquisition_function
..    :members: acquisitions/acquisition_function

.. autoclass:: AcquisitionFunction
   :member-order: bysource
   :members:

   .. automethod:: __init__

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
