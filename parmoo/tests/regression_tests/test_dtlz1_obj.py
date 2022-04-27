
""" Use ParMOO to solve the DTLZ1 problem, treating DTLZ1 as an objective.

Uses unnamed variables, the g1_sim simulation function, and the
dtlz1_obj objective functions to define the problem.

"""

from parmoo import MOOP
from parmoo.optimizers import LBFGSB
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.simulations.dtlz import g1_sim as sim_func
from parmoo.objectives.dtlz import dtlz1_obj as obj_func
import numpy as np

# Set the problem dimensions
NUM_DES = 3
NUM_OBJ = 3

# Create a MOOP
moop = MOOP(LBFGSB)

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
                                            obj_ind=i, num_obj=NUM_OBJ)})

# Define 2 constraints to nudge the solver in the right direction

def min_constraint(x, sx, der=0):
    """ x[NUM_OBJ-1:NUM_DES] >= 0.55 """

    if der == 1:
        dx = np.zeros(x.shape[0], dtype=x.dtype)
        for i in range(NUM_OBJ - 1, NUM_DES):
            dx[i] = -1.0
        return dx
    elif der == 2:
        return np.zeros(sx.shape[0], dtype=sx.dtype)
    else:
        fx = 0.0
        for i in range(NUM_OBJ - 1, NUM_DES):
            fx += (0.55 - x[i])
        return fx

def max_constraint(x, sx, der=0):
    """ x[NUM_OBJ-1:NUM_DES] <= 0.65 """

    if der == 1:
        dx = np.zeros(x.shape[0], dtype=x.dtype)
        for i in range(NUM_OBJ - 1, NUM_DES):
            dx[i] = 1.0
        return dx
    elif der == 2:
        return np.zeros(sx.shape[0], dtype=sx.dtype)
    else:
        fx = 0.0
        for i in range(NUM_OBJ - 1, NUM_DES):
            fx += (x[i] - 0.65)
        return fx

# Add 2 constraints to the problem
moop.addConstraint({'name': "Lower Bounds", 'constraint': min_constraint})
moop.addConstraint({'name': "Upper Bounds", 'constraint': max_constraint})

# Add 10 acquisition funcitons
for i in range(10):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 5 iterations
moop.solve(5)

# Check that 150 simulations were evaluated
assert(moop.getObjectiveData()['x_vals'].shape[0] == 150)
# Check that some solutions were found
assert(moop.getSimulationData()[0]['x_vals'].shape[0] == 150)
assert(moop.getPF()['f_vals'].shape[0] > 0)
