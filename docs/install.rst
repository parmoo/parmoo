Advanced Installation
=====================

Before you install ParMOO, make sure that you have all of the dependencies.
 * `Python 3.6+ <https://www.python.org/downloads/>`_
 * `numpy <https://numpy.org/>`_ -- for data structures and performant
   numerical linear algebra
 * `scipy <https://scipy.org>`_ -- for important scientific calculations
   (such as optimization solvers) needed by certain modules
 * `pyDOE <https://pythonhosted.org/pyDOE/>`_ -- for generating experimental
   designs

Next, you will need to clone ParMOO and place the base directory into
your system's Python3 load path.

On Debian-based systems with a bash shell, this looks like
the following.

.. code-block:: bash

   git clone [parmoo url]
   cd [parmoo base]
   export PYTHONPATH=$PYTHONPATH:`pwd`

To make the installation permanent, you could
 * ensure that you have added the ParMOO base directory to your
   Python3 path (e.g., by appending the last command above with
   `pwd` replaced by the base directory to your shell startup file;
   ``~/.bashrc`` for the bash shell) or
 * place a copy of the ``[parmoo base]/parmoo`` directory in your system's
   default library load path
   (e.g., ``~/.local/lib/pythonX.X/site-packages`` on many Unix/Linux systems).

Additionally, if you would like to use libEnsemble to handle parallel
function evaluations (from :mod:`extras.libe`),
you will need to also install
`libEnsemble <https://github.com/Libensemble/libensemble>`_.

To install libEnsemble with PyPI, use

.. code-block:: bash

   pip3 install libensemble

or visit the libEnsemble documentation for detailed
`installation instructions <https://libensemble.readthedocs.io/en/main/advanced_installation.html>`_.
