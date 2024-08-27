
""" Use ParMOO to "solve" an infeasible problem.

Ensure that ParMOO correctly identifies the most nearly feasible point.

"""

from parmoo import MOOP
from parmoo.optimizers import GlobalSurrogate_BFGS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights
from parmoo.searches import LatinHypercube
import os
import numpy as np

# For this test, use all user-defined functions

def sim(x):
    " User sim for sample problem. "
    return [sum([x[name] for name in x])]

def obj_func(x, sx):
    " User objective for sample problem. "
    return 3.0 * sx["Sample sim"]

def obj_grad(x, sx):
    " User gradient for sample problem. "
    dx, ds = {}, {}
    for key in x:
        dx[key] = 0.
    for key in sx:
        ds[key] = 0.
    ds["Sample sim"] = 3.0
    return dx, ds

def const_func(x, sx):
    return sum([(x[name] - 1.0) ** 2 for name in x])

def const_grad(x, sx):
    " User constraint gradient for sample problem. "
    dx, ds = {}, {}
    for key in x:
        dx[key] = -2.
    for key in sx:
        ds[key] = 0.
    return dx, ds

# Create a MOOP
moop = MOOP(GlobalSurrogate_BFGS)

# Add 2 continuous design variables
for i in range(2):
    moop.addDesign({'name': f"x{i+1}", 'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "Sample sim",
                    'm': 1,
                    'sim_func': sim,
                    'search': LatinHypercube,
                    'surrogate': GaussRBF,
                    'hyperparams': {}})

# Add user objective functions
moop.addObjective({'name': "My objective",
                   'obj_func': obj_func,
                   'obj_grad': obj_grad})

# Add user constraint functions, with only 1 feasible point
moop.addConstraint({'name': "My constraint",
                    'con_func': const_func,
                    'con_grad': const_grad})

# Add 1 acquisition function
moop.addAcquisition({'acquisition': UniformWeights, 'hyperparams': {}})

# Solve the problem with 5 iterations to search for feasible point
moop.solve(5)

# Check that all solutions are (nearly) feasible
assert(all([np.all(np.abs(moop.getPF()[name] - 1.0 < 1.0e-8))
            for name in moop.getDesignType().names]))
