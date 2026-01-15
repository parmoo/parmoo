Release Process
===============

A release can be undertaken only by a project administrator.
A project administrator should have an administrator role on the ParMOO
GitHub, PyPI, and ReadTheDocs pages.

Before release
--------------

- A release branch should be taken off ``develop`` (or ``develop`` pulls
  controlled).

- Release notes for this version are added to the ``CHANGELOG.rst`` file.

- Version number is updated wherever it appears and ``+dev`` suffix is removed
  (in ``parmoo/version.py``, ``README.rst``, and ``docs/refs.rst``).

- Check ``README.rst``: 

  - *Citing ParMOO* entries correct (including version and year of documentation)?

    - Citations in ``docs/refs.rst`` and ``docs/tutorials/local_method.rst`` 
      consistent with associated entries in ``README.rst``?

  - Coverage badge branch set to ``main`` (for badge and link)?

  - Email the address listed under Resources and confirm that you have 
    received a response; if not, this address needs to be updated also in 
    ``docs/refs.rst``, ``docs/quickstart.rst``, ``CONTRIBUTING.rst``, 
    ``pyproject.toml``, ``SUPPORT.rst``, and possibly other places.

- Check for spelling mistakes and typos in the docs and Python docstrings:
  - ``pyspelling -c .github/config/.spellcheck.yml``

- ``pyproject.toml`` and ``parmoo/__init__.py`` are checked to ensure all
  information is up to date.

- ``MANIFEST.in`` and ``pyproject.toml`` ``packages`` list are checked for
  completeness.  In particular, if new modules or subdirectories have been
  added, the ``pyproject.toml`` file must be updated accordingly.  Locally, try
  out ``python -m build --sdist`` and check created tarball contains correct
  files and directories for PyPI package. Note: this command requires the
  ``build`` package be installed on your Python, although it is not part of any
  ``parmoo`` requirement lists since it is only used for creating releases.

- Check that ``parmoo`` requirements (in ``REQUIREMENTS.txt``)
  are compatible with ``readthedocs.io`` (in ``.readthedocs.yml``)

- Tests are run with source to be released (this may iterate):

  - On-line CI (GitHub Actions) tests must pass [#tests1]_. 

  - Documentation must build and display correctly wherever hosted (currently
    readthedocs.org).

- Pull request from either the develop or release branch to main requesting
  one or more reviewers (including at least one other administrator).

- Reviewer will check that all tests have passed [#tests1]_ and will then approve merge.

During release
--------------

An administrator will take the following steps.

- Merge the pull request into main.

- Once CI tests have passed [#tests1]_ on main:

  - A GitHub release will be taken from the main

  - A tarball (source distribution) will be uploaded to PyPI (should be done
    via ``twine`` by an admin using PyPI-API-token authentication)

- If the merge was made from a release branch (instead of develop), merge this
  branch into develop.

- Create a new commit on develop that:
  - Appends ``+dev`` to the version number (wherever it appears)
  - Changes the coverage badge branch to ``develop`` (for badge and link).

- Update the version number cited in the parMOO solver farm at
  https://github.com/parmoo/parmoo-solver-farm 
  and check that the citation, contact, etc. information there is up to date.

After release
-------------

- Ensure all relevant GitHub issues are closed.

- Check that the conda-forge package has tracked latest release
  and update dependency list if needed -- an admin will need to approve the
  automatically generated PR on https://github.com/conda-forge/parmoo-feedstock


.. rubric:: Footnote

.. [#tests1] If coverage tests do not pass (i.e., coverage decreases), this should be
  noted in PR/review as well as in the release notes.
