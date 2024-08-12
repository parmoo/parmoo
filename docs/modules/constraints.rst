Constraint Function Templates (ABCs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For templates to define your own constraint function, see
the
:class:`CompositeFunction ABC in structs <structs.CompositeFunction>`.

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
