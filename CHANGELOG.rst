Release Notes
=============

Below are the release notes for ParMOO.

May reference issues on:
https://github.com/parmoo/parmoo/issues

Release 0.4.0
-------------

:Date: Aug 27, 2024

Major interface-breaking refactor of core `MOOP` builder and libraries plus
minor bug-fixes.

Major changes:

 - Under the hood, ParMOO now uses `jax[cpu]` for all gradient evaluations,
   constraint evaluations, and function evaluations (but not simulation
   evaluations) on the critical path; this means that objective and
   constraint functions can be just-in-time (jit) compiled, which can give
   massive performance improvements.  However, not all Python features are
   supported by `jax.jit()`, so users must be careful to write their objective
   and constraint functions mindfully
 - In order for jax to be effective, we have updated ParMOO's interfaces to
   avoid using optional arguments -- this affects the objective and constraint
   function interfaces, and a separate gradient function must be provided by
   the user (as opposed to an optional `der` argument, as previously used)
 - In order to make it easier to support mixed variable types in jax and for
   future maintainability, all design variables have been replaced by a library
   of design variable `Embedder` classes (corresponding ABCs also added to the
   `structs.py`)
 - In order to make ParMOO more maintainable and for jax to work smoothly, we
   have dropped support for unnamed variables (Closes #31)
 - All `SurrogateFunction` and `AcquisitionFunction` libraries have been updated
   to be more jax-friendly
 - The `SurrogateOptimizer` class has been refactored to include a callback to
   observe simulation evaluation results.  It has also been given almost full
   control over when model improvement steps are called in order to make
   implementing many DFO methods easier
 - The pattern search family of optimizers has been greatly improved
 - `PyDOE` has been dropped since most relevant DOEs now appear in the newly
   added `scipy.stats.qmc` module
 - Switching to `PyDOE` required us to change how the numpy random seed is set.
   A random seed object is now passed as a hyperparameter to ParMOO and
   propagated to all libraries, which is the recommended way
 - Updated docs to reflect above changes

Style changes:

 - Overhauled some of the unit tests
 - Style fixes throughout
 - Renamed several methods and classes to have a consistent naming convention

Interface breaking:

 - The `Embedder` class is now used to define custom design variables (see
   Major changes)
 - jax is now used to evaluate gradients in the `SurrogateOptimizer` class (see
   Major changes) -- this alone shouldn't break the interface most use-cases, but
   may lead to decreased performance if not careful (see notes on achieving
   good performance in jax in the docs).  Additionally, jax defaults to single
   precision so double precision must be set manually using a `jax.config`
   command (see docs)
 - `SurrogateFunction`, `SurrogateOptimizer`, and `AcquisitionFunction`
   interfaces have changed (only affects users using custom methods)
 - The random seed must now be set using a numpy random seed object (see Major
   changes and examples in the docs)
 - We no longer support unnamed variables (see Major changes)
 - Gradient functions are now provided via an additional key in the dictionary,
   and cannot be set using an optional argument (see Major changes and examples
   in the updated docs)
 - Many library functions, methods, and classes have been renamed for
   consistency.  In some cases, the old names remain as aliases for backward
   compatibility.  See Style changes

New features:

 - ParMOO now supports jax for autograd (see Major changes above)
 - `SurrogateOptimizer` is now notified of the results of each simulation
   evaluation. (This allows checks for sufficient improvements)
 - Numerous new `AcquisitionFunction` types added
 - Added an option to create a private workdir for each libEnsemble thread
   (Closes #82)

Minor changes:

 - Updates to support `numpy 2.0`
 - Added a code coverage badge and updated the release process to reflect the
   extra steps needed to make this work (Closes #21 , Closes #93)

Requirements:

 - Added `jax[cpu]` to list of requirements
 - Removed `pyDOE` from list of requirements, in favor of `scipy.stats` (added
   in `scipy 1.10.0`)
 - Released the lock on `libensemble` version (from Release 0.3.1)
 - Updated all version requirements to be new enough to support `numpy 2.0`

Release 0.3.1
-------------

:Date: Sep 25, 2023

Bug-fixes and minor restructuring for future releases.

Fixed several serious bugs/limitations:

 - Introduced in v0.3.0: when generating batches, a bug was introduced into
   the lines of code that filter out duplicate candidates, resulting in
   significantly decreased performance but no errors being raised
 - Allow for ParMOO to still generate target points for the
   ``AcquisitionFunction``, even when there are no feasible points in the
   database
 - Increase the number of characters allowed in a name when working with
   libEnsemble from 10 to 40 characters
 - Broke the ``MOOP.iterate()`` method apart into 2 functions (``iterate``
   and ``filterBatch``), which makes the code more maintainable and allows
   for future improvements to the ``libE_MOOP`` parallelism
 - Updated deprecated keys in ``.readthedocs.yml`` config file

Release 0.3.0
-------------

:Date: Jul 6, 2023

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
