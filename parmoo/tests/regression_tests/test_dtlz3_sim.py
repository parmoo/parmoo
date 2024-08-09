
""" Use ParMOO to solve the DTLZ3 problem, treating DTLZ3 as a simulation.

Uses named variables, the dtlz3_sim simulation function, and the
SingleSimObjective objective functions to define the problem.

Also activates ParMOO's checkpointing feature, in order to test checkpointing.

"""

import numpy as np
import os
from parmoo import MOOP
from parmoo.optimizers import GlobalSurrogate_PS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import FixedWeights
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import dtlz3_sim as sim_func
from parmoo.objectives import SingleSimObjective as obj_func

# Set the problem dimensions
NUM_DES = 3
NUM_OBJ = 3

# Create a MOOP
moop = MOOP(GlobalSurrogate_PS)

# Add NUM_DES continuous design variables
for i in range(NUM_DES):
    moop.addDesign({'name': f"x{i+1}", 'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "DTLZ3",
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
                                            ("DTLZ3", i), goal="min")})

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
moop.addConstraint({'name': "Lower Bounds", 'constraint': min_constraint})
moop.addConstraint({'name': "Upper Bounds", 'constraint': max_constraint})

# Add NUM_OBJ acquisition functions
for i in range(NUM_OBJ):
    moop.addAcquisition({'acquisition': FixedWeights,
                         'hyperparams': {'weights': np.eye(NUM_OBJ)[i]}})


# Solve the problem with 5 iterations + checkpointing
moop.setCheckpoint(True)
moop.solve(5)

# Check that 115 simulations were evaluated and solutions are feasible
assert(moop.getObjectiveData().shape[0] == (100 + NUM_OBJ*5))
assert(moop.getSimulationData()['DTLZ3'].shape[0] == (100 + NUM_OBJ*5))
assert(moop.getPF().shape[0] > 0)

# Clean up test directory (remove checkpoint files)
os.remove("parmoo.moop")
os.remove("parmoo.simdb.json")
os.remove("parmoo.surrogate.1")
