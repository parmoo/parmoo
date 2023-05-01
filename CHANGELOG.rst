Release Notes
=============

Below are the release notes for ParMOO.

May reference issues on:
https://github.com/parmoo/parmoo/issues

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
