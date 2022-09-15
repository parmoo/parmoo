Advanced Installation
=====================

ParMOO can be installed with ``pip`` or directly from its GitHub_ source.

ParMOO's base has the following dependencies, which may be automatically
installed depending on your choice of method:

 * Python_ 3.6+
 * numpy_ -- for data structures and performant numerical linear algebra
 * scipy_ -- for scientific calculations needed for specific modules
 * pyDOE_ -- for generating experimental designs
 * pandas_ -- for exporting the resulting databases

Additional dependencies are needed to use the additional features in
``parmoo.extras``:

 * libEnsemble_ -- for managing parallel simulation evaluations

And for using the Pareto front visualization library in ``parmoo.viz``:

 * plotly_ -- for generating interactive plots
 * dash_ -- for hosting interactive plots in your browser
 * kaleido_ -- for exporting static plots post-interaction

If you want to run the tests (in ``parmoo.tests``), then you will also need:

 * pytest_,
 * pytest-cov_, and
 * flake8_.

pip
---

The easiest way to install is via the PyPI package manager (``pip`` utility).
To install the latest release:

.. code-block:: bash

    pip install < --user > parmoo

where the braces around ``< --user >`` indicates that the ``--user`` flag is
optional.

Note that the default install will not install the extra dependencies,
such as libEnsemble_.

To install *all* dependencies, use:

.. code-block:: bash

    pip install < --user > parmoo[extras]

To check the installation by running the full test suite, use:

.. code-block:: bash

    python3 setup.py test

which will also install the test dependencies (pytest_, pytest-cov_, and
flake8_).

Install from GitHub source
--------------------------

You may want to install ParMOO from its GitHub_ source code, so that
you can easily pull the latest updates.

The easiest way to do this is to clone it from our GitHub_ and then
``pip`` install it in-place by using the ``-e .`` option.
In a bash shell, that looks like this.

.. code-block:: bash

   git clone https://github.com/parmoo/parmoo
   cd parmoo
   pip install -e .

This command will use the ``setup.py`` file to generate an ``egg`` inside
the ``parmoo`` base directory.

Alternatively, you could just add the ``parmoo`` base directory to your
``PYTHONPATH`` environment variable. In the bash shell, this looks like:

.. code-block:: bash

   git clone https://github.com/parmoo/parmoo
   cd parmoo
   export PYTHONPATH=$PYTHONPATH:`pwd`

However, this technique will not install any of ParMOO's dependencies.

Additionally, if you would like to use libEnsemble_ to handle parallel
function evaluations (from :mod:`extras.libe`),
you will need to also install libEnsemble_.

To install libEnsemble with PyPI, use

.. code-block:: bash

   pip3 install libensemble

or visit the libEnsemble_documentation_ for detailed installation instructions.

After installation, you can run the tests using either:

.. code-block:: bash

    python3 setup.py test

(if you used the ``pip install -e .`` method), or:

.. code-block:: bash

    parmoo/tests/run-tests.sh -cu<rl>


.. _Actions: https://github.com/parmoo/parmoo/actions
.. _dash: https://dash.plotly.com
.. _flake8: https://flake8.pycqa.org/en/latest
.. _GitHub: https://github.com/parmoo/parmoo
.. _kaleido: https://github.com/plotly/Kaleido
.. _libEnsemble: https://github.com/Libensemble/libensemble
.. _libEnsemble_documentation: https://libensemble.readthedocs.io/en/main/advanced_installation.html
.. _numpy: https://numpy.org
.. _pandas: https://pandas.pydata.org
.. _plotly: https://plotly.com/python
.. _pyDOE: https://pythonhosted.org/pyDOE
.. _pytest: https://docs.pytest.org/en/7.0.x
.. _pytest-cov: https://pytest-cov.readthedocs.io/en/latest
.. _Python: https://www.python.org/downloads
.. _ReadTheDocs: https://parmoo.readthedocs.org
.. _scipy: https://scipy.org
