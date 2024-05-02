Surrogate Functions
-------------------

A surrogate model is associated with each simulation when its
simulation dictionary is added to the ``MOOP`` object.
This technique is used for generatng an approximation to the simulation's
response surface, based on data gathered during the solve.

.. code-block:: python

    from parmoo import surrogates

Available techniques are:

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
