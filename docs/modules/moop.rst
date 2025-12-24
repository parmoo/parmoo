ParMOO Core
------------------------

These classes and helper functions define the core of ParMOO's solver
infrastructure and algorithms.

Serial MOOP Class
~~~~~~~~~~~~~~~~~~~~~~~~

This is the main serial implementation for solving MOOPs with ParMOO.

.. code-block:: python

    from parmoo import MOOP

Use this class to define and solve a MOOP.  Several of the core methods needed
during setup and usage may be defined in the base class
``parmoo.core.moop_base.MOOP_base`` (defined below).
The ``MOOP.solve(...)`` method will perform simulations serially for this
class.

.. automodule:: core.moop
..    :members: core/moop

.. autoclass:: MOOP
   :member-order: bysource
   :members:

   .. automethod:: __init__

MOOP Base Class
~~~~~~~~~~~~~~~~~~~~~~~~

This is the abstract base class for defining MOOPs with ParMOO.

.. code-block:: python

    from parmoo.core.moop_base import MOOP_base

Extend this class to define a MOOP.

In order to solve a MOOP, you must provide an implementation including the
``MOOP.solve(...)`` and other undefined methods.

.. automodule:: core.moop_base
..    :members: core/moop_base

.. autoclass:: MOOP_base
   :member-order: bysource
   :members:

   .. automethod:: __init__
   .. automethod:: _extract
   .. automethod:: _embed
   .. automethod:: _embed_grads
   .. automethod:: _unpack_sim
   .. automethod:: _pack_sim
   .. automethod:: _vobj_funcs
   .. automethod:: _vcon_funcs
   .. automethod:: _vpen_funcs
   .. automethod:: _fit_surrogates
   .. automethod:: _update_surrogates
   .. automethod:: _set_surrogate_tr
   .. automethod:: _evaluate_surrogates
   .. automethod:: _surrogate_uncertainty
   .. automethod:: _evaluate_objectives
   .. automethod:: _obj_fwd
   .. automethod:: _obj_bwd
   .. automethod:: _evaluate_constraints
   .. automethod:: _con_fwd
   .. automethod:: _con_bwd
   .. automethod:: _evaluate_penalty
   .. automethod:: _pen_fwd
   .. automethod:: _pen_bwd
