Basic Tutorials
===============

This is a collection of all tutorials demonstrating basic ParMOO functionality
(collected from throughout the ParMOO User Guide).

Quickstart Demo
~~~~~~~~~~~~~~~

This is a basic example of how to build and solve a MOOP with ParMOO,
taken from the :doc:`Quickstart <../quickstart>` guide.

.. literalinclude:: ../../examples/quickstart.py
    :language: python

The above code produces the following output.

.. literalinclude:: ../../examples/quickstart.out

Named Output Types
~~~~~~~~~~~~~~~~~~

This code snippet demonstrates ParMOO's output datatype when the
:mod:`MOOP <moop.MOOP>` object is defined using *named* variables.

.. literalinclude:: ../../examples/named_var_ex.py
    :language: python

The above code produces the following output.

.. literalinclude:: ../../examples/named_var_ex.out

Unnamed Output Types
~~~~~~~~~~~~~~~~~~~~

This code snippet demonstrates ParMOO's output datatype when the
:mod:`MOOP <moop.MOOP>` object is defined using *unnamed* variables.

.. literalinclude:: ../../examples/unnamed_var_ex.py
    :language: python

The above code produces the following output.

.. literalinclude:: ../../examples/unnamed_var_ex.out

Adding Precomputed Simulation Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code snippet demonstrates how to add a precomputed simulation output
to ParMOO's internal simulation databases.

.. literalinclude:: ../../examples/precomputed_data.py
    :language: python

The above code produces the following output.

.. literalinclude:: ../../examples/precomputed_data.out

.. _advanced_ex:

Solving a MOOP with Derivative-Based Solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to minimize two conflicting quadratic functions
of three variables (named ``x1``, ``x2``, and ``x3``),
under the constraint that an additional categorical variable (``x4``)
must be fixed in class ``0`` :math:`^*`,
using the derivative-based solver :mod:`LBFGSB <optimizers.lbfgsb.LBFGSB>`.

:math:`^*` No, this constraint does not really affect the solution;
it is just here to demonstrate how constraints/categorical variables
are handled by derivative-based solvers.
ParMOO does not use derivative information associated with
any categorical variables, so the derivative w.r.t. ``x4`` can be set
to any value, without affecting the outcome.

.. literalinclude:: ../../examples/advanced_ex.py
    :language: python

The above code produces the following output.

.. literalinclude:: ../../examples/advanced_ex.out

Note how in the full simulation database several of the design points
violate the constraint (``x4 != 0``).
But in the solution, the constraint is always satisfied.
