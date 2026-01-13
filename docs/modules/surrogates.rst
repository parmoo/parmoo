Surrogate Functions
-------------------

A surrogate model is associated with each simulation when its
simulation dictionary is added to the ``MOOP`` object.
This technique is used for generating an approximation to the simulation's
response surface, based on data gathered during the solve.

.. code-block:: python

    from parmoo import surrogates

To implement your own custom surrogate function in ParMOO, import and extend
the ``SurrogateFunction`` ABC.

.. code-block:: python

    from parmoo.surrogates.surrogate_function import SurrogateFunction

The ``SurrogateFunction`` ABC and the library of existing surrogate functions
are documented below.

SurrogateFunction
~~~~~~~~~~~~~~~~~

.. automodule:: surrogates.surrogate_function
..    :members: surrogates/surrogate_function

.. autoclass:: SurrogateFunction
   :member-order: bysource
   :members:

   .. automethod:: __init__

Gaussian Process (RBF) Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: surrogates.gaussian_proc
..    :members: surrogates/gaussian_proc

.. autoclass:: GaussRBF
   :member-order: bysource
   :members:

   .. automethod:: __init__

Polynomial Models
~~~~~~~~~~~~~~~~~

.. automodule:: surrogates.polynomial
..    :members: surrogates/polynomial

.. autoclass:: Linear
   :member-order: bysource
   :members:

   .. automethod:: __init__
