Release Process
===============

A release can be undertaken only by a project administrator.
A project administrator should have an administrator role on the ParMOO
GitHub, PyPI, and readthedocs pages.

Before release
--------------

- A release branch should be taken off ``develop`` (or ``develop`` pulls
  controlled).

- Release notes for this version are added to the ``CHANGELOG.rst`` file.

- Version number is updated wherever it appears and ``+dev`` suffix is removed
  (in ``parmoo/version.py``, ``README.rst``, and ``docs/refs.rst``).

- Check ``README.rst``: 
  - *Citing ParMOO* correct?
  - ``docs/refs.rst`` correct?
  - Coverage badge branch set to ``main`` (for badge and link)?

- Check for spelling mistakes and typos in the docs and Python docstrings:
  - ``pyspelling -c .github/workflows/.spellcheck.yml``

- ``setup.py`` and ``parmoo/__init__.py`` are checked to ensure all
  information is up to date.

- ``MANIFEST.in`` is checked. Locally, try out ``python setup.py sdist`` and
  check created tarball contains correct files and directories for PyPI
  package.

- Check that ``parmoo`` requirements (in ``REQUIREMENTS.txt``)
  are compatible with ``readthedocs.io`` (in ``.readthedocs.yml``)

- Tests are run with source to be released (this may iterate):

  - On-line CI (GitHub Actions) tests must pass.

  - Documentation must build and display correctly wherever hosted (currently
    readthedocs.org).

- Pull request from either the develop or release branch to main requesting
  one or more reviewers (including at least one other administrator).

- Reviewer will check that all tests have passed and will then approve merge.

During release
--------------

An administrator will take the following steps.

- Merge the pull request into main.

- Once CI tests have passed on main:

  - A GitHub release will be taken from the main

  - A tarball (source distribution) will be uploaded to PyPI (should be done
    via ``twine`` by an admin using PyPI-API-token authentication)

- If the merge was made from a release branch (instead of develop), merge this
  branch into develop.

- Create a new commit on develop that:
  - Appends ``+dev`` to the version number (wherever it appears)
  - Changes the coverage badge branch to ``develop`` (for badge and link).

After release
-------------

- Ensure all relevant GitHub issues are closed.

- Check that the conda-forge package has tracked latest release
  and update dependency list if needed -- an admin will need to approve the
  automatically generated PR on https://github.com/conda-forge/parmoo-feedstock
