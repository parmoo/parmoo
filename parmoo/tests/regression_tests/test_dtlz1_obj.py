
""" Use ParMOO to solve the DTLZ1 problem, treating DTLZ1 as an objective.

Uses unnamed variables, the g1_sim simulation function, and the
dtlz1_obj objective functions to define the problem.

"""

from parmoo import MOOP
from parmoo.optimizers import GlobalSurrogate_BFGS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import g1_sim as sim_func
from parmoo.objectives.dtlz import dtlz1_obj as obj_func
from parmoo.objectives.dtlz import dtlz1_grad as obj_grad
import numpy as np

# Set the problem dimensions
NUM_DES = 3
NUM_OBJ = 3

# Create a MOOP
moop = MOOP(GlobalSurrogate_BFGS)

# Add NUM_DES continuous design variables
for i in range(NUM_DES):
    moop.addDesign({'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "g1",
                    'm': 1,
                    'sim_func': sim_func(moop.getDesignType(),
                                         offset=0.6),
                    'search': LatinHypercube,
                    'surrogate': GaussRBF,
                    'hyperparams': {}})

# Add NUM_OBJ objective functions
for i in range(NUM_OBJ):
    moop.addObjective({'name': f"DTLZ1 objective {i+1}",
                       'obj_func': obj_func(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            obj_ind=i, num_obj=NUM_OBJ),
                       'obj_grad': obj_grad(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            obj_ind=i, num_obj=NUM_OBJ)
                        })

# Define 2 constraints to nudge the solver in the right direction

def min_constraint_func(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] >= 0.55 """

    fx = 0.
    for i in range(NUM_OBJ - 1, NUM_DES):
        fx += (0.55 - x[f"x{i+1}"])
    return fx

def min_constraint_grad(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] >= 0.55 """

    dx, ds = {}, {}
    for i in x:
        dx[i] = 0.
    for i in sx:
        ds[i] = 0.
    for i in range(NUM_OBJ - 1, NUM_DES):
        dx[f"x{i+1}"] = -1.
    return dx, ds

def max_constraint_func(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] <= 0.65 """

    fx = 0.
    for i in range(NUM_OBJ - 1, NUM_DES):
        fx += (x[f"x{i+1}"] - 0.65)
    return fx

def max_constraint_grad(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] <= 0.65 """

    dx, ds = {}, {}
    for i in x:
        dx[i] = 0.
    for i in sx:
        ds[i] = 0.
    for i in range(NUM_OBJ - 1, NUM_DES):
        dx[f"x{i+1}"] = 1.
    return dx, ds

# Add 2 constraints to the problem
moop.addConstraint({'name': "Lower Bounds",
                    'con_func': min_constraint_func,
                    'con_grad': min_constraint_grad
                    })
moop.addConstraint({'name': "Upper Bounds",
                    'con_func': max_constraint_func,
                    'con_grad': max_constraint_grad
                    })

# Add 10 acquisition funcitons
for i in range(10):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 5 iterations
moop.solve(5)

# Check that 150 simulations were evaluated
assert(moop.getObjectiveData()['x1'].shape[0] == 150)
# Check that some solutions were found
assert(moop.getSimulationData()['g1']['x1'].shape[0] == 150)
assert(moop.getPF()['x1'].shape[0] > 0)
