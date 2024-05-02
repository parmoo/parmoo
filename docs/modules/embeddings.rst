Embedder Classes for Custom Variable Types
------------------------------------------

When defining your :class:`MOOP <moop.MOOP>` object as described in
:ref:`the name key section <naming>`, it is possible to provide a
custom variable by using the ``embedder`` key.

When used, this key must contain a value with the
:class:`Embedder <structs.Embedder>` type.
Embeddings for several common variable types are defined below, and
provided behind the scenes by ParMOO whenever a non custom design variable
is added to a problem.

They can also be added manually or studied to understand how a custom
design variable might be implemented.

.. code-block:: python

    from parmoo.embeddings import default_embedders

These pre-existing embedders are:

Default Embedder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: embeddings.default_embedders
..    :members: embeddings/default_embedders

.. autoclass:: ContinuousEmbedder
   :member-order: bysource
   :members:

   .. automethod:: __init__

.. autoclass:: IntegerEmbedder
   :member-order: bysource
   :members:

   .. automethod:: __init__

.. autoclass:: CategoricalEmbedder
   :member-order: bysource
   :members:

   .. automethod:: __init__

.. autoclass:: IdentityEmbedder
   :member-order: bysource
   :members:

   .. automethod:: __init__
