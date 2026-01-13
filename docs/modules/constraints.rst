Constraint Functions
-------------------

This module provides a library of pre-defined ParMOO constraint function
implementations and templates to define your own constraint function.

ConstraintFunction Template (ABC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: constraints.const_func
..    :members: constraints/const_func

.. autoclass:: ConstraintFunction
   :member-order: bysource
   :members:

   .. automethod:: __init__

Constraint Function Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: constraints.const_lib
..    :members: constraints/const_lib

.. autoclass:: SingleSimBound
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SumOfSimSquaresBound
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SumOfSimsBound
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SingleSimBoundGradient
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SumOfSimSquaresBoundGradient
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SumOfSimsBoundGradient
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__
