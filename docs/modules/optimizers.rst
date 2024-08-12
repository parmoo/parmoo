Surrogate Optimizers
--------------------

When initializing a new ``MOOP`` object
(see :doc:`MOOP Classes <class_api>`),
you must provide a surrogate optimization problem solver, which will
be used to generate candidate solutions for each iteration.

.. code-block:: python

    from parmoo import optimizers

*Note that when using a gradient-based technique, you must provide
gradient evaluation options for all objective and constraint functions,
by adding code to handle the optional ``der`` input.*

.. code-block:: python

    def f(x, sx, der=0):
        # When using gradient-based solvers, define extra if-cases for
        # handling der=1 (calculate df/dx) and der=2 (caldculate df/dsx).

Pattern Search Techniques (gradient-free)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: optimizers.pattern_search
..    :members: optimizers/pattern_search

.. autoclass:: GlobalSurrogate_PS
   :member-order: bysource
   :members:

   .. automethod:: __init__

.. autoclass:: LocalSurrogate_PS
   :member-order: bysource
   :members:

   .. automethod:: __init__

Random Search Techniques (gradient-free)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. automodule:: optimizers.random_search
..    :members: optimizers/random_search

.. autoclass:: GlobalSurrogate_RS
   :member-order: bysource
   :members:

   .. automethod:: __init__

L-BFGS-B Variations (gradient-based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: optimizers.lbfgsb
..    :members: optimizers/lbfgsb

.. autoclass:: GlobalSurrogate_BFGS
   :member-order: bysource
   :members:

   .. automethod:: __init__

.. autoclass:: LocalSurrogate_BFGS
   :member-order: bysource
   :members:

   .. automethod:: __init__

