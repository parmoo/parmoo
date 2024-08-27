
""" Use ParMOO to solve the DTLZ3 problem, treating DTLZ3 as an objective.

Uses unnamed variables, the g1_sim simulation function, and the
dtlz3_obj objective functions to define the problem.

"""

import numpy as np
from parmoo import MOOP
from parmoo.optimizers import GlobalSurrogate_PS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import g1_sim as sim_func
from parmoo.objectives.dtlz import dtlz3_obj as obj_func

# Set the problem dimensions
NUM_DES = 3
NUM_OBJ = 3

# Create a MOOP
moop = MOOP(GlobalSurrogate_PS)

# Add NUM_DES continuous design variables
for i in range(NUM_DES):
    moop.addDesign({'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "g1",
                    'm': 1,
                    'sim_func': sim_func(moop.getDesignType(),
                                         num_obj=NUM_OBJ,
                                         offset=0.6),
                    'search': LatinHypercube,
                    'surrogate': GaussRBF,
                    'hyperparams': {}})

# Add NUM_OBJ objective functions
for i in range(NUM_OBJ):
    moop.addObjective({'name': f"DTLZ3 objective {i+1}",
                       'obj_func': obj_func(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            obj_ind=i, num_obj=NUM_OBJ)})

# Define 2 constraints to nudge the solver in the right direction

def min_constraint(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] >= 0.5 """

    fx = 0.0
    for i in range(NUM_OBJ - 1, NUM_DES):
        fx += (0.5 - x[f"x{i+1}"])
    return fx

def max_constraint(x, sx):
    """ x[NUM_OBJ-1:NUM_DES] <= 0.7 """

    fx = 0.0
    for i in range(NUM_OBJ - 1, NUM_DES):
        fx += (x[f"x{i+1}"] - 0.7)
    return fx

# Add 2 constraints to the problem
moop.addConstraint({'name': "Lower Bounds", 'con_func': min_constraint})
moop.addConstraint({'name': "Upper Bounds", 'con_func': max_constraint})

# Add 5 acquisition functions
for i in range(5):
    moop.addAcquisition({'acquisition': UniformWeights, 'hyperparams': {}})

# Solve the problem with 5 iterations
moop.solve(5)

# Check that 125 simulations were evaluated and solutions were found
assert(moop.getObjectiveData()['x1'].shape[0] == 125)
assert(moop.getSimulationData()['g1'].shape[0] == 125)
assert(moop.getPF().shape[0] > 0)
