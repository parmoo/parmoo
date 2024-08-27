
""" Use ParMOO to solve a convex, user-defined problem.

Uses unnamed variables and public function definitions to define the problem.

Also turns on checkpointing, to test ParMOO's checkpointing of unnamed
variables and globally defined functions.

"""

import numpy as np
import os
from parmoo import MOOP
from parmoo.optimizers import LocalSurrogate_BFGS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube

# Set the problem dimensions
NUM_DES = 3

# For this test, use all user-defined functions

def sim(x):
    " User sim for sample problem. "
    return [sum([x[i] for i in x])]

def obj1_func(x, sx):
    " User obj1 for sample problem. "
    return sum([sx[i] for i in sx])

def obj1_grad(x, sx):
    " User obj1 for sample problem. "
    dx, ds = {}, {}
    for key in x:
        dx[key] = 0.
    for key in sx:
        ds[key] = 1.
    return dx, ds

def obj2_func(x, sx):
    " User obj2 for sample problem. "
    return 1.0 - sum([sx[i] for i in sx])

def obj2_grad(x, sx):
    " User obj2 for sample problem. "
    dx, ds = {}, {}
    for key in x:
        dx[key] = 0.
    for key in sx:
        ds[key] = -1.
    return dx, ds

# Create a MOOP
moop = MOOP(LocalSurrogate_BFGS)

# Add NUM_DES continuous design variables
for i in range(NUM_DES):
    moop.addDesign({'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "Sample sim",
                    'm': 1,
                    'sim_func': sim,
                    'search': LatinHypercube,
                    'surrogate': GaussRBF,
                    'hyperparams': {}})

# Add user objective functions
moop.addObjective({'obj_func': obj1_func, 'obj_grad': obj1_grad},
                  {'obj_func': obj2_func, 'obj_grad': obj2_grad})

# Add 10 acquisition functions
for i in range(10):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 5 iterations with checkpointing on
moop.setCheckpoint(True)
moop.solve(5)

# Check that 150 simulations were evaluated and solutions are feasible
assert(moop.getObjectiveData().shape[0] == 150)
assert(moop.getSimulationData()['Sample sim'].shape[0] == 150)
assert(moop.getPF().shape[0] > 0)

# Clean up test directory (remove checkpoint files)
os.remove("parmoo.moop")
os.remove("parmoo.simdb.json")
os.remove("parmoo.surrogate.1")
os.remove("parmoo.optimizer")
