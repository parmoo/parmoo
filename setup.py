#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This is for setting up parmoo, license and details can be
# found at https://github.com/parmoo/parmoo/

from setuptools import setup
from setuptools.command.test import test as TestCommand

exec(open("parmoo/version.py").read())


class Run_TestSuite(TestCommand):
    def run_tests(self):
        import os
        import sys
        py_version = sys.version_info[0]
        print("Python version from setup.py is", py_version)
        run_string = "parmoo/tests/run-tests.sh -curl" #+ str(py_version)
        os.system(run_string)


class ToxTest(TestCommand):
    user_options = []

    def initialize_options(self):
        TestCommand.initialize_options(self)

    def run_tests(self):
        import tox
        tox.cmdline()


setup(
    name="parmoo",
    version=__version__,
    description="A parallel multiobjective optimization solver that seeks to exploit simulation-based structure in objective and constraint functions",
    url="https://github.com/parmoo/parmoo",
    author="Tyler H. Chang and Stefan M. Wild",
    author_email="parmoo@mcs.anl.gov",
    license="BSD 3-clause",

    packages=["parmoo",
              "parmoo.acquisitions",
              "parmoo.extras",
              "parmoo.optimizers",
              "parmoo.searches",
              "parmoo.surrogates",
              "parmoo.simulations",
              "parmoo.objectives",
              "parmoo.constraints",
              "parmoo.tests",
              "parmoo.tests.unit_tests",
              "parmoo.tests.libe_tests",
              "parmoo.tests.regression_tests"],

    install_requires=["numpy", "scipy", "pyDOE"],

    # If run tests through setup.py - downloads these but does not install
    tests_require=["pytest", "pytest-cov", "flake8"],

    extras_require={
        'extras': ["libensemble"],
        'docs': ["sphinx", "sphinxcontrib.bibtex", "sphinx_rtd_theme"]},

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules"],

    cmdclass={'test': Run_TestSuite, 'tox': ToxTest}
)
