
""" Use ParMOO to solve a convex, user-defined problem.

Uses unnamed variables and public function definitions to define the problem.

Also turns on checkpointing, to test ParMOO's checkpointing of unnamed
variables and globally defined functions.

"""

from parmoo import MOOP
from parmoo.optimizers import TR_LBFGSB
from parmoo.surrogates import LocalGaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
import os
import numpy as np

# Set the problem dimensions
NUM_DES = 3

# For this test, use all user-defined functions

def sim(x):
    " User sim for sample problem. "
    return [sum(x)]

def obj1(x, sx, der=0):
    " User obj1 for sample problem. "
    if der == 1:
        return np.zeros(len(x))
    elif der == 2:
        return np.ones(len(sx))
    else:
        return sum(sx)

def obj2(x, sx, der=0):
    " User obj2 for sample problem. "
    if der == 1:
        return np.zeros(len(x))
    elif der == 2:
        return -np.ones(len(sx))
    else:
        return 1.0 - sum(sx)

# Create a MOOP
moop = MOOP(TR_LBFGSB)

# Add NUM_DES continuous design variables
for i in range(NUM_DES):
    moop.addDesign({'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "Sample sim",
                    'm': 1,
                    'sim_func': sim,
                    'search': LatinHypercube,
                    'surrogate': LocalGaussRBF,
                    'hyperparams': {}})

# Add user objective functions
moop.addObjective({'obj_func': obj1}, {'obj_func': obj2})

# Add NUM_OBJ acquisition funcitons
for i in range(10):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 5 iterations with checkpointing on
moop.setCheckpoint(True)
moop.solve(5)

# Check that 150 simulations were evaluated and solutions are feasible
assert(moop.getObjectiveData()['x_vals'].shape[0] == 150)
assert(moop.getSimulationData()[0]['s_vals'].shape[0] == 150)
assert(moop.getPF()['f_vals'].shape[0] > 0)

# Clean up test directory (remove checkpoint files)
os.remove("parmoo.moop")
os.remove("parmoo.simdb.json")
os.remove("parmoo.surrogate.1")
