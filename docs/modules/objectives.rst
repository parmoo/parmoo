Objective Functions
-------------------

This module provides a library of pre-defined ParMOO objective function
implementations and templates to define your own objective function.

ObjectiveFunction Template (ABC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: objectives.obj_func
..    :members: objectives/obj_func

.. autoclass:: ObjectiveFunction
   :member-order: bysource
   :members:

   .. automethod:: __init__

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
