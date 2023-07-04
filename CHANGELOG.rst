Release Notes
=============

Below are the release notes for ParMOO.

May reference issues on:
https://github.com/parmoo/parmoo/issues

Release 0.3.0
-------------

:Date: Jul 5, 2023

Significant structural changes for long-term support of future solvers,
bug-fixes, and significant improvements to documentation.

Major Changes:

 - ``surrogates.GaussRBF`` and ``surrogates.LocalGaussRBF`` now
   calculate model-form uncertainties
 - structural changes to ``MOOP`` class to support propagation of
   uncertainty information
 - added ``EI_RandomConstraint`` acquisition, which can be used to
   implement Bayesian optimization -- note that for large budgets,
   this is not currently recommended due to computational expense
   of numerical integration
 - updated ``LocalGPS`` to use trust regions, when provided, and
   perform multiple restarts
 - ``SurrogateOptimizer`` class now has access to more information about
   the objective, including raw simulation outputs, in order to support
   more diverse structure-exploiting solvers
 - Added additional stopping criteria to both ``MOOP.solve()`` and
   ``libE_MOOP.solve()`` -- all stopping criteria are now optional
   (although at least one must be specified) but they are ordered such
   that calling ``MOOP.solve(k)``, where ``k`` is a positional input,
   will pass to the ``iter_max`` criteria and produce the same behavior
   as before -- closes #18

API Changes:

 - In most cases, none. However, it is possible that if users were previously
   passing arguments to the ``MOOP.solve()`` method explicitly, then the
   name of the first positional argument has changed:
   ``budget`` -> ``max_iters``
 - For users implementing their own ``searches``, ``surrogates``,
   ``optimizers``, or ``acquisitions``, several classes in the ``structs``
   module have been updated to support the present restructuring of
   the ``MOOP`` class

Docs:

 - Updated Quickstart guide and README to demonstrate recommended inputs
   and settings for ParMOO -- this includes no more ``lambda`` functions,
   which closes #50
 - Added a FAQ page with additional usage details and responses to frequent
   questions -- the answers in which close #61
 - Added a new tutorial on how to perform high-dimensional multiobjective
   optimization on a limited budget with ParMOO
 - Changed examples and documentation to use and discuss pandas dataframes,
   which generally produce more legible outputs
 - Updated ``libE_MOOP`` example to demonstrate how to retrieve data in a
   way that is threadsafe for both Python MP and MPI usage

Requirements:

 - We now require scipy v1.10 or newer, due to usage of qmc integration tools
 - At the time of this release, libEnsemble is using a deprecated version of
   Pydantic -- for this release only we have fixed the requirement on
   libEnsemble to v0.9.2, but we will relax this requirement in the future
   once they have patched the issue

Bug-fixes:

 - Fixed an issue where in rare cases, problems with too many categorical
   variables could produce unexpected batch sizes
 - Errors in definition of test problems: DTLZ5, 6, and 7 (new implementations
   have been confirmed against ``pymoo``)
 - Fixed an issue which occasionally caused the ``libE_MOOP`` class to error
   out during post-run cleanup when used with MPI
 - Patched an issue with ``format="pandas"`` option for
   ``MOOP.getSimulationData()`` class and added a similar option to
   all ``libE_MOOP`` "getter" functions

Minor changes:

 - Fixed typos in docs/doc-strings
 - Updated styles to comply with new ``flake8`` recommendations
 - New unit tests added
 - Added warnings when ParMOO is run with bad budget settings

Release 0.2.2
-------------

:Date: Apr 25, 2023

Hot-fix for a minor issue in the plotting library without workaround.

 - Resolves #58

Release 0.2.1
-------------

:Date: Apr 10, 2023

Minor performance improvements, maintenance, and restructuring of test cases.

 - Both Gaussian RBF surrogates in ``parmoo/surrogates/gaussian_proc.py``
   now use the current mean of the response values as the prior instead
   of the zero function. This greatly improves convergence rates in practice,
   especially for our structure-exploiting methods.
 - Using an old version of ``plotly/dash`` for now because of a dash issue
   described in plotly/dash#2460
 - Added additional tests to check gradient calculations of ``GaussRBF``
   surrogates.
 - Added whitespace to pass new ``flake8`` standards.
 - Added year to JOSS publication in several places
 - Added "et al." to our docs configuration file after author names, to
   credit additional contributors in our documentation.

Release 0.2.0
-------------

:Date: Feb 2, 2023

Official release corresponding to accepted JOSS article.

 - Added support for a wider variety of design variables (including integer
   types), as well as support for "custom" design variables that use
   user-provided custom embedders/extractors
   Documentation on design variables has been expanded accordingly.
   Although design variables are still specified through dicts not classes,
   this addresses and therefore closes the primary issue raised in
   parmoo/parmoo#28
 - Updated ``extras/libe.py`` corresponding to interface changes made in
   libEnsemble Release 0.8.0. This also addresses the issues on MacOS,
   referenced in parmoo/parmoo#34
 - Added a post-run visualization library and corresponding
   documentation, closing issue parmoo/parmoo#27
 - Allow solvers to start from an initial point that is infeasible, so that
   problems with relaxable constraints and a very small feasible set can
   still be solved
 - Various style changes and additional usage environments requested by
   JOSS reviewers openjournals/joss-reviews#4468 including parmoo/parmoo#32
 - Added support for multistarting optimization solvers when solving
   surrogate problems. This is particularly important for the global
   ``GaussRBF`` surrogate
 - Fixed an issue in how model improvement points are calculated, as
   implemented in the ``surrogate.improve`` method for each GaussRBF variation
   in ``surrogates/gaussian_proc.py``, which was created when adding support
   for custom design variables
 - The default design tolerance for continuous variables now depends upon
   the value of ``ub - lb``

Note: 

 - Dropped support for Python 3.6, due to changes to GitHub Actions documented
   on actions/setup-python#544

Known issues:

 - The visualization library uses advanced plotly/dash features, which may
   not support the chrome browser, as described in parmoo/parmoo#37

Release 0.1.0
-------------

:Date: May 10, 2022

Initial release.

Known issues and desired features will be raised on GitHub post-release.

Known issues:

 - update unit tests to use sim/obj/const libraries
 - restructure test suite, unit tests are currently not usable as
   additional documentation
 - ``solve()`` method(s) should support additional stopping criteria
 - allow for maximizing objectives and constraint lower bounds without
   "hacky" solution (negating values)
 - missing functions from DTLZ libraries
 - ``README.md`` needs a code coverage badge

Desired features:

 - update, test, and merge-in MDML interface
 - allow user to choose whether or not to use named variables via ``useNames``
   method, or similar
 - add a funcx simulation interface, using libEnsemble release 0.9
 - add predicter interface and standalone module
 - a GUI interface for creating MOOPs
 - static visualization tools for plotting results
   (from ``MOOP.getPF()`` method)
 - a visualization dashboard for viewing progress interactively
 - design variable types should be a class, with embed/extract methods
   that can be called by ``MOOP.__embed__()`` and ``MOOP.__extract__()``
