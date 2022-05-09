ParMOO Solver and Component Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement a new acquisition function, solver, surrogate, or
search technique for ParMOO, you must match its interface.
An interface definition for each of these methods is provided in the
``structs`` module in the corresponding Abstract Base Class (ABC).

.. code-block:: python

    from parmoo import structs

When implementing one of these techniques, you should extend the corresponding
ABC, defined below.

.. automodule:: structs
..    :members: structs

AcquisitionFunction
~~~~~~~~~~~~~~~~~~~

.. autoclass:: AcquisitionFunction
   :member-order: bysource
   :members:

   .. automethod:: __init__

GlobalSearch
~~~~~~~~~~~~

.. autoclass:: GlobalSearch
   :member-order: bysource
   :members:

   .. automethod:: __init__

SurrogateFunction
~~~~~~~~~~~~~~~~~

.. autoclass:: SurrogateFunction
   :member-order: bysource
   :members:

   .. automethod:: __init__

SurrogateOptimizer
~~~~~~~~~~~~~~~~~~

.. autoclass:: SurrogateOptimizer
   :member-order: bysource
   :members:

   .. automethod:: __init__
