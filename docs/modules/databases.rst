SimulationDatabase Classes for ParMOO's Multiobjective Database
---------------------------------------------------------------

When defining your :class:`MOOP <core.moop.MOOP>` object as described in
:ref:`the name key section <naming>`, ParMOO will instantiate an internal
database for storing multiobjective simulation, function, and constraint
information.  This information is accessible via various methods of the
:class:`MOOP <core.moop.MOOP>` class, or from the ``MOOP.database`` attribute.

In the future, the type of database will be made a configurable parameter of
the MOOP definition.

For now, the abstract base class is accessible via the
``databases.simulation_database.SimulationDatabase`` ABC, and the only
implementation of this class is the ``databases.NumpyDatabase`` class.

.. code-block:: python

    from parmoo.databases.simulation_database import SimulationDatabase
    from parmoo.databases import NumpyDatabase

SimulationDatabase (ABC)
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: databases.simulation_database
..    :members: databases/simulation_database

.. autoclass:: SimulationDatabase
   :member-order: bysource
   :members:

   .. automethod:: __init__

NumpyDatabase (default implementation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: databases.numpy_database
..    :members: databases/numpy_database

.. autoclass:: NumpyDatabase
   :member-order: bysource
   :members:

   .. automethod:: __init__
