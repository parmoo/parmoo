Release Notes
=============

Below are the release notes for ParMOO.

May reference issues on:
https://github.com/parmoo/parmoo/issues

Release 0.0.0
-------------

Initial release.

Known issues and desired features will be raised on GitHub post-release.

Known issues:

 - update unit tests to use sim/obj/const libraries
 - restructure test suite, unit tests are currently not usable as
   additional documentation
 - ``solve()`` method(s) should support additional stopping criteria
 - allow for maximizing objectives and constraint lower bounds without
   "hacky" solution (negating values)
 - note missing functions from DTLZ libraries

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