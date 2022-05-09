Search Techniques
-----------------

A search technique is associated with each simulation when the
simulation dictionary is added to the ``MOOP`` object.
This technique is used for generating simulation data prior to the
first iteration of ParMOO, so that the initial surrogate models can
be fit.

For most search techniques, it is highly recommended that you supply
the following optional hyperparameter keys/values:
 * ``search_budget (int)``: specifies how many samples will be generated
   for this simulation.

.. code-block:: python

    from parmoo import searches

Available search techniques are as follows:

Latin Hypercube Sampling
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: searches.latin_hypercube
..    :members: searches/latin_hypercube

.. autoclass:: LatinHypercube
   :member-order: bysource
   :members:

   .. automethod:: __init__

