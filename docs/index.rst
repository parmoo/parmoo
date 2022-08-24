.. ParMOO documentation master file

.. image:: img/logo-ParMOO.svg
    :align: center
    :alt: ParMOO

|

ParMOO: Python library for parallel multiobjective simulation optimization
==========================================================================

ParMOO is a Python library for solving multiobjective simulation-based
optimization problems, while exploiting problem structure.

ParMOO stands for "parallel multiobjective optimization".
ParMOO can be used to solve multiobjective optimization problems (MOOPs)
or to generate batches of simulation inputs for parallel evaluation.

Target Audience
---------------

ParMOO is intended for computational scientists, optimization experts,
and machine learning practitioners, who are looking to build custom solvers
for computationally expensive problems.
ParMOO allows users with various levels of optimization expertise to easily
construct and deploy custom solvers for multiobjective simulation optimization
problems by mixing-and-matching search techniques, surrogate models,
acquisition functions, and solvers, while exploiting problem structure as
much as possible.

Either build your own custom solution from scratch, or leverage our
built-in libraries of solver components!

Getting Started
---------------

If you're new to ParMOO:

 * Check out the :ref:`Quickstart`
 * Try some of our :doc:`Basic Tutorials <tutorials/basic-tutorials>`
 * Try :doc:`Running in Parallel using libEnsemble <extras>`
 * Check us out on `GitHub <https://github.com/parmoo/parmoo>`_

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   quickstart
   install
   about
   how-to-write
   extras
   refs

.. toctree::
   :maxdepth: 2
   :caption: API:

   api

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/basic-tutorials
   tutorials/libe-tutorial

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide:

   dev-guide/contributing
   dev-guide/release-proc
   dev-guide/release-notes
   dev-guide/modules

Indices:
========

 * :ref:`genindex`
 * :ref:`modindex`
