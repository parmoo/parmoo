
""" Use ParMOO to solve the DTLZ2 problem, treating DTLZ2 as a simulation.

Uses unnamed variables, the dtlz2_sim simulation function, and the
single_sim_out objective functions to define the problem.

"""

import numpy as np
from parmoo import MOOP
from parmoo.optimizers import LocalSurrogate_BFGS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import dtlz2_sim as sim_func
from parmoo.objectives import SingleSimObjective as obj_func
from parmoo.objectives import SingleSimGradient as obj_grad
from parmoo.constraints import SumOfSimSquaresBound as const_func
from parmoo.constraints import SumOfSimSquaresBoundGradient as const_grad

# Set the problem dimensions
NUM_DES = 3
NUM_OBJ = 3

# Create a MOOP
moop = MOOP(LocalSurrogate_BFGS)

# Add NUM_DES continuous design variables
for i in range(NUM_DES):
    moop.addDesign({'ub': 1.0, 'lb': 0.0,
                    'des_type': "continuous", 'des_tol': 1.0e-8})

# Add the simulation
moop.addSimulation({'name': "DTLZ2",
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
                                            ("DTLZ2", i), goal="min"),
                       'obj_grad': obj_grad(moop.getDesignType(),
                                            moop.getSimulationType(),
                                            ("DTLZ2", i), goal="min")})

# Add a constraint
moop.addConstraint({'name': "SOS Sim Bounds",
                    'con_func': const_func(moop.getDesignType(),
                                           moop.getSimulationType(),
                                           sim_inds=[("DTLZ2", 0),
                                                     ("DTLZ2", 1),
                                                     ("DTLZ2", 2)],
                                           bound_type="upper",
                                           bound=4.0),
                    'con_grad': const_grad(moop.getDesignType(),
                                           moop.getSimulationType(),
                                           sim_inds=[("DTLZ2", 0),
                                                     ("DTLZ2", 1),
                                                     ("DTLZ2", 2)],
                                           bound_type="upper",
                                           bound=4.0)})

# Add 5 acquisition functions
for i in range(5):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with the equivalent of 5 iterations
moop.solve(sim_max=125)

# Check that 150 simulations were evaluated
assert(moop.getObjectiveData().shape[0] == 125)
assert(moop.getSimulationData()['DTLZ2'].shape[0] == 125)
assert(all([fi['f1']**2 + fi['f2']**2 + fi['f3']**2 <= 4
            for fi in moop.getPF()]))
