Contributing to ParMOO
======================

Contributions of source code, documentations, and fixes are happily
accepted via GitHub pull request to

    https://github.com/parmoo/parmoo

If you are planning a contribution, reporting bugs, or suggesting features, 
we encourage you to discuss the concept by opening a github issue at

  https://github.com/parmoo/parmoo/issues
  
or by emailing  ``parmoo@mcs.anl.gov``
and interacting with us to ensure that your effort is well-directed.

Contribution Process
--------------------

ParMOO uses the Gitflow model. Contributors should typically branch from, and
make pull requests to, the ``develop`` branch. The ``main`` branch is used only
for releases. Pull requests may be made from a fork, for those without
repository write access.

Issues can be raised at

    https://github.com/parmoo/parmoo/issues

Issues may include reporting bugs or suggested features.

By convention, user branch names should have a ``<type>/<name>`` format, where
example types are ``feature``, ``bugfix``, ``testing``, ``docs``, and
``experimental``.
Administrators may take a ``hotfix`` branch from the main, which will be
merged into ``main`` (as a patch) and ``develop``.
Administrators may also take a ``release`` branch off ``develop`` and then
merge this branch into ``main`` and ``develop`` for a release.

When a branch closes a related issue, the pull request message should include
the phrase "Closes #N," where N is the issue number.

New features should be accompanied by at least one test case.

All pull requests to ``develop`` or ``main`` must be reviewed by at least one
administrator.

Developer's Certificate
-----------------------

ParMOO is distributed under a 3-clause BSD license (see LICENSE_).  
The act of submitting a pull request or patch will be understood as an 
affirmation of the following:

::

  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.


.. _LICENSE: https://github.com/parmoo/parmoo/blob/main/LICENSE
