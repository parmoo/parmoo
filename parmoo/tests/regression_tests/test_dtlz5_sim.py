
""" Use ParMOO to solve the DTLZ5 problem, treating DTLZ5 as a simulation.

Uses named variables, the dtlz3_sim simulation function, and the
SingleSimObjective objective functions to define the problem.

"""

import numpy as np
from parmoo import MOOP
from parmoo.optimizers import LocalSurrogate_BFGS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import dtlz5_sim as sim_func
from parmoo.objectives import SingleSimObjective as obj_func
from parmoo.objectives import SingleSimGradient as obj_grad

# Set the problem dimensions
NUM_DES = 3
NUM_OBJ = 3

# Create a MOOP
moop = MOOP(LocalSurrogate_BFGS)

# Add NUM_DES continuous design variables
for i in range(NUM_DES):
    moop.addDesign({'name': f"x{i+1}", 'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "DTLZ5",
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
                                            ("DTLZ5", i), goal="min"),
                       'obj_grad': obj_grad(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            ("DTLZ5", i), goal="min")})

# Add 5 acquisition functions
for i in range(5):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 5 iterations
moop.solve(5)

# Check that 125 simulations were evaluated and solutions are feasible
assert(moop.getObjectiveData().shape[0] == 125)
assert(moop.getSimulationData()['DTLZ5'].shape[0] == 125)
assert(moop.getPF().shape[0] > 0)
