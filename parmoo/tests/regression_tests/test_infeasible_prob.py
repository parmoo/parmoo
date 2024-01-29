
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
    return [sum([x[name] for name in x.dtype.names])]

def obj(x, sx, der=0):
    " User objective for sample problem. "
    if der == 1:
        return np.zeros(1, x.dtype)[0]
    elif der == 2:
        dsx = np.zeros(1, sx.dtype)[0]
        dsx["Sample sim"] = 3.0
        return dsx
    else:
        return 3.0 * sx["Sample sim"]

def const(x, sx, der=0):
    " User constraint for sample problem. "
    if der == 1:
        dx = np.ones(1, x.dtype)[0]
        for name in x.dtype.names:
            dx[name] *= -2.0
        return dx
    elif der == 2:
        return np.zeros(1, sx.dtype)[0]
    else:
        return sum([(x[name] - 1.0) ** 2 for name in x.dtype.names])

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
moop.addObjective({'name': "My objective", 'obj_func': obj})

# Add user constraint functions, with only 1 feasible point
moop.addConstraint({'name': "My constraint", 'constraint': const})

# Add 1 acquisition function
moop.addAcquisition({'acquisition': UniformWeights, 'hyperparams': {}})

# Solve the problem with 5 iterations to search for feasible point
moop.solve(5)

# Check that all solutions are (nearly) feasible
assert(all([np.all(np.abs(moop.getPF()[name] - 1.0 < 1.0e-8))
            for name in moop.getDesignType().names]))
