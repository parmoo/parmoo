
""" Use ParMOO to solve the DTLZ1 problem, treating DTLZ1 as a simulation.

Uses named variables, the dtlz1_sim simulation function, and the
single_sim_out objective functions to define the problem.

"""

import numpy as np
from parmoo import MOOP
from parmoo.optimizers import GlobalSurrogate_BFGS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import dtlz1_sim as sim_func
from parmoo.objectives import SingleSimObjective as obj_func
from parmoo.objectives import SingleSimGradient as obj_grad

# Set the problem dimensions
NUM_DES = 3
NUM_OBJ = 3

# Create a MOOP
moop = MOOP(GlobalSurrogate_BFGS)

# Add NUM_DES continuous design variables
for i in range(NUM_DES):
    moop.addDesign({'name': f"x{i+1}", 'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "DTLZ1",
                    'm': NUM_OBJ,
                    'sim_func': sim_func(moop.getDesignType(),
                                         num_obj=NUM_OBJ,
                                         offset=0.6),
                    'search': LatinHypercube,
                    'surrogate': GaussRBF,
                    'hyperparams': {}})

# Add NUM_OBJ objective functions
for i in range(NUM_OBJ):
    moop.addObjective({'name': f"f{i+1}",
                       'obj_func': obj_func(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            ("DTLZ1", i), goal="min"),
                       'obj_grad': obj_grad(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            ("DTLZ1", i), goal="min"),
                      })

# Define 2 constraints to nudge the solver in the right direction

def min_constraint_func(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] >= 0.55 """

    fx = 0.0
    for i in range(NUM_OBJ - 1, NUM_DES):
        fx += (0.55 - x[f"x{i+1}"])
    return fx

def min_constraint_grad(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] >= 0.55 """

    dx, ds = {}, {}
    for key in x:
        dx[key] = 0.
    for key in sx:
        ds[key] = 0.
    for i in range(NUM_OBJ - 1, NUM_DES):
        dx[f"x{i+1}"] = -1.0
    return dx, ds

def max_constraint_func(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] <= 0.65 """

    fx = 0.0
    for i in range(NUM_OBJ - 1, NUM_DES):
        fx += (x[f"x{i+1}"] - 0.65)
    return fx

def max_constraint_grad(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] <= 0.65 """

    dx, ds = {}, {}
    for key in x:
        dx[key] = 0.
    for key in sx:
        ds[key] = 0.
    for i in range(NUM_OBJ - 1, NUM_DES):
        dx[f"x{i+1}"] = 1.0
    return dx, ds

# Add 2 constraints to the problem
moop.addConstraint({'name': "Lower Bounds",
                    'con_func': min_constraint_func,
                    'con_grad': min_constraint_grad})
moop.addConstraint({'name': "Upper Bounds",
                    'con_func': max_constraint_func,
                    'con_grad': max_constraint_grad})

# Add 5 acquisition functions
for i in range(5):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 5 iterations
moop.solve(5)

# Check that 125 simulations were evaluated and solutions are feasible
assert(moop.getObjectiveData().shape[0] == 125)
assert(moop.getSimulationData()['DTLZ1'].shape[0] == 125)
assert(moop.getPF().shape[0] > 0)
