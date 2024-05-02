Base (serial) MOOP Class
~~~~~~~~~~~~~~~~~~~~~~~~

This is the base class for solving MOOPs with ParMOO.

.. code-block:: python

    from parmoo import moop

Use this class to define and solve a MOOP.
The ``MOOP.solve(...)`` method will perform simulations serially for this
class.

.. automodule:: moop
..    :members: moop

.. autoclass:: MOOP
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

