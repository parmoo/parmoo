
""" Use ParMOO to solve a convex, user-defined problem, with mixed variables
(continuous, integer, categorical, and custom).

Uses named variables and public function definitions to define the problem.

"""

from parmoo import MOOP
from parmoo.optimizers import TR_LBFGSB
from parmoo.surrogates import LocalGaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
import os
import numpy as np

# For this test, use all user-defined functions

def sim(x):
    " User sim for sample problem. "
    # Categorical variables
    if x["cat var 1"] == "good":
        result = 0.0
    else:
        result = 1.0
    if x["cat var 2"] != "low":
        result += 1.0
    return np.array([result + x["cont var"] ** 2 + x["int var"] ** 2 +
                     float(x["custom var"]) ** 2,
                     result + (x["cont var"] - 1.0) ** 2 + x["int var"] ** 2 +
                     float(x["custom var"]) ** 2])

def obj1(x, sx, der=0):
    " User obj1 for sample problem. "
    if der == 1:
        return np.zeros(1, dtype=moop.getDesignType())[0]
    elif der == 2:
        return np.ones(1, dtype=moop.getSimulationType())[0]
    else:
        return sx["my sim"][0]

def obj2(x, sx, der=0):
    if der == 1:
        return np.zeros(1, dtype=moop.getDesignType())[0]
    elif der == 2:
        return np.ones(1, dtype=moop.getSimulationType())[0]
    else:
        return sx["my sim"][1]

# Create a MOOP
moop = MOOP(TR_LBFGSB)

# Add design variables
moop.addDesign({'name': "cont var", 'ub': 1.0, 'lb': 0.0,
                'des_type': "continuous", 'des_tol': 1.0e-8})
moop.addDesign({'name': "int var", 'ub': 5, 'lb': -5,
                'des_type': "integer"})
moop.addDesign({'name': "cat var 1", 'des_type': "categorical",
                'levels': ["good", "bad"]})
moop.addDesign({'name': "cat var 2", 'des_type': "categorical",
                'levels': ["low", "med", "high"]})
moop.addDesign({'name': "custom var", 'des_type': "custom",
                'embedding_size': 1,
                'embedder': lambda x: float(x),
                'extracter': lambda x: str(x)})

# Add the simulation
moop.addSimulation({'name': "my sim",
                    'm': 2,
                    'sim_func': sim,
                    'search': LatinHypercube,
                    'surrogate': LocalGaussRBF,
                    'hyperparams': {}})

# Add user objective functions
moop.addObjective({'obj_func': obj1}, {'obj_func': obj2})

# Add NUM_OBJ acquisition funcitons
for i in range(10):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 5 iterations with checkpointing on
moop.solve(5)

# Check that 150 simulations were evaluated and solutions are feasible
#assert(moop.getObjectiveData()['f1'].shape[0] == 150)
#assert(moop.getSimulationData()['my sim'].shape[0] == 150)
#assert(moop.getPF()['f1'].shape[0] > 0)
print(moop.getPF())
