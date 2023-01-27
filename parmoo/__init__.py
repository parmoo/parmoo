"""
ParMOO.

A parallel multiobjective optimization solver that seeks to exploit
simulation-based structure in objective and constraint functions.

"""

from .version import __version__
__author__ = "Tyler H. Chang, Stefan M. Wild, and Hyrum Dickinson"
__credits__ = "Argonne National Laboratory"

from .moop import MOOP
