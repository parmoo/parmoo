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

It is also possible to import and extend the ``GlobalSearch`` ABC to implement
a custom global search technique.

.. code-block:: python

    from parmoo.searches.global_search import GlobalSearch

The ``GlobalSearch`` ABC and existing library of search techniques are
documented below.

GlobalSearch
~~~~~~~~~~~~

.. automodule:: searches.global_search
..    :members: searches/global_search

.. autoclass:: GlobalSearch
   :member-order: bysource
   :members:

   .. automethod:: __init__

Latin Hypercube Sampling
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: searches.latin_hypercube
..    :members: searches/latin_hypercube

.. autoclass:: LatinHypercube
   :member-order: bysource
   :members:

   .. automethod:: __init__

