Objective Templates (ABCs)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For templates to define your own objective function, see
the
:class:`CompositeFunction ABC in structs <structs.CompositeFunction>`.

Objective Function Library
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: objectives.obj_lib
..    :members: objectives/obj_lib

.. autoclass:: SingleSimObjective
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SumOfSimSquaresObjective
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SumOfSimsObjective
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SingleSimGradient
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SumOfSimSquaresGradient
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

.. autoclass:: SumOfSimsGradient
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: __call__
