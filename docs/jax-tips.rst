Jax tips and tricks
===================

.. _jax_tips:

Starting in version 0.4.0, ParMOO uses jax_ for algorithmic differentiation
and just-in-time compilation.

We have carefully implemented core functionality so that ParMOO will try
to ``jit`` and ``jacrev`` with jax_, but easily fall-back to derivative free
techniques and uncompiled code if needed.
However, to get the most out of ParMOO, it is important to make sure that
the relevant bits of code can

 1. be compiled via ``jax.jit()``, and
 2. be differentiated via ``jax.jacrev()``.

In most cases, when everything in ParMOO's critical path can be compiled via
``jax.jit()``, you can expect over a 10x speedup in iteration times.

When the :method:`MOOP.compile() <moop.MOOP.compile>` method is called,
ParMOO attempts to jit many common items.
If infol-level logging is turned on (see the logging tutorial) then
ParMOO will print warnings for items that failed to ``jit``.

In many cases, it is worth taking time to figure out why these items won't
jit and it is often possible to adjust them so that they do.

Things that often fail to jit
-----------------------------

Common items that fail to jit include:

 1. The design variable embedders -- if you are using categorical variables
    and the level IDs contain string values (as opposed to integer level names)
    than jax's linear algebra compiler LAX will not be able compile the
    embedders.
 2. Your objective and constraint functions (and their gradients) --
    you are responsible for providing python implementations of the objective
    and constraint functions (and their gradients if needed).
    These functions will all be called many times on the critical path, so
    it is essential that they can jit.
    Common reasons why they cannot is if you are passing undetermined sized
    inputs/outputs in sub-functions or if you are using Python if-statements.
    For advanced control-flow, there are many tricks and jax alternatives
    to using if-statements, which can jit.
    See some of the examples in our tutorials_ or read the jax_sharp_bits_
    to get a feel.

In terms of differentiability, we do NOT require you to implement your
gradients in jax.
Instead, you only need to provide separate implementations of the gradient
and we will link them for you!
However, if you are writing a custom surrogate or acquisition function,
you should be aware that the ``surrogate.evaluate()`` and
``acquisition.scalarize()`` function must be differentiable and preferrably
jitt-able.

Again, see the jax_ docs or read the jax_sharp_bits_ to get a feel for how
this works.

Things that don't need to jit (but still could)
-----------------------------------------------

There are a few items that ParMOO does not need to jit or differentiate to
achieve optimal performance.
Most notably, we will not attempt to differentiate or jit your simulation
function, which is assumed to be complex.
However, if possible, you may be interested in jitting your simulation for
performance gains if you are not already implementing it in a compiled
language!

Other methods of the ``SurrogateFunction`` and ``AcquisitionFunction``
classes such as ``surrogate.fit()`` or ``acquisition.setTarget()`` also
do not need to jit as they are not on the critical path.

For users interested in implementing custom solvers, there are notes in the
:mod:`structs module <structs>`.


.. _jax: https://jax.readthedocs.io/en/latest/
.. _jax_sharp_bits: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
.. _tutorials: tutorials/basic-tutorials.html
