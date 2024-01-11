
""" Use ParMOO to solve the DTLZ2 problem, treating DTLZ2 as a simulation.

Uses unnamed variables, the dtlz2_sim simulation function, and the
single_sim_out objective functions to define the problem.

"""

from parmoo import MOOP
from parmoo.optimizers import LocalSurrogate_BFGS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import dtlz2_sim as sim_func
from parmoo.objectives import single_sim_out as obj_func
from parmoo.constraints import sos_sim_bound as const_func
import numpy as np

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
                                            i, goal="min")})

# Add a constraint
moop.addConstraint({'name': "SOS Sim Bounds",
                    'constraint': const_func(moop.getDesignType(),
                                             moop.getSimulationType(),
                                             sim_inds=[0, 1, 2],
                                             type="upper",
                                             bound=4.0)})

# Add 10 acquisition funcitons
for i in range(10):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with the equivalent of 5 iterations
moop.solve(sim_max=150)

# Check that 150 simulations were evaluated
assert(moop.getObjectiveData()['x_vals'].shape[0] == 150)
# Check that some solutions were found
assert(moop.getSimulationData()[0]['x_vals'].shape[0] == 150)
assert(all([sum(fi**2) <= 4.0 for fi in moop.getPF()['f_vals']]))
