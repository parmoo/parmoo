
""" Use ParMOO to solve the DTLZ2 problem, treating DTLZ2 as an objective.

Uses named variables, the g2_sim simulation function, and the
dtlz2_obj objective functions to define the problem.

"""

import numpy as np
from parmoo import MOOP
from parmoo.optimizers import LocalSurrogate_BFGS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import g2_sim as sim_func
from parmoo.objectives.dtlz import dtlz2_obj as obj_func
from parmoo.objectives.dtlz import dtlz2_grad as obj_grad
from parmoo.constraints import SingleSimBound as const_func
from parmoo.constraints import SingleSimBoundGradient as const_grad

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
moop.addSimulation({'name': "g2",
                    'm': 1,
                    'sim_func': sim_func(moop.getDesignType(),
                                         num_obj=NUM_OBJ,
                                         offset=0.6),
                    'search': LatinHypercube,
                    'surrogate': GaussRBF,
                    'hyperparams': {}})

# Add NUM_OBJ objective functions
for i in range(NUM_OBJ):
    moop.addObjective({'name': f"DTLZ2 objective {i+1}",
                       'obj_func': obj_func(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            obj_ind=i, num_obj=NUM_OBJ),
                       'obj_grad': obj_grad(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            obj_ind=i, num_obj=NUM_OBJ)})

# Add a constraint
moop.addConstraint({'name': "Max Sim Bounds",
                    'con_func': const_func(moop.getDesignType(),
                                           moop.getSimulationType(),
                                           sim_ind="g2",
                                           bound_type="upper",
                                           bound=2.0),
                    'con_grad': const_grad(moop.getDesignType(),
                                           moop.getSimulationType(),
                                           sim_ind="g2",
                                           bound_type="upper",
                                           bound=2.0)})

# Add 5 acquisition functions
for i in range(5):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 5 iterations
moop.solve(5)

# Check that 125 simulations were evaluated and solutions are feasible
assert(moop.getObjectiveData()['x1'].shape[0] == 125)
assert(moop.getSimulationData()['g2'].shape[0] == 125)
assert(all([sum([fi[f"DTLZ2 objective {i+1}"]**2 for i in range(NUM_OBJ)]) <= 4
            for fi in moop.getPF()]))
