
""" Use ParMOO to solve a convex, user-defined problem, with mixed variables
(continuous, integer, categorical, and custom).

Uses named variables and public function definitions to define the problem.

"""

import numpy as np
import os
from parmoo import MOOP
from parmoo.optimizers import GlobalSurrogate_PS
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.embeddings import ContinuousEmbedder

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

def obj1(x, sx):
    " User obj1 for sample problem. "
    return sx["my sim"][0]

def obj2(x, sx):
    " User obj2 for sample problem. "
    return sx["my sim"][1]

# Create a MOOP
moop = MOOP(GlobalSurrogate_PS)

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
                'ub': 1.0, 'lb': 0.0,
                'embedding_size': 1,
                'embedder': ContinuousEmbedder})

# Add the simulation
moop.addSimulation({'name': "my sim",
                    'm': 2,
                    'sim_func': sim,
                    'search': LatinHypercube,
                    'surrogate': GaussRBF,
                    'hyperparams': {'search_budget': 20}})

# Add user objective functions
moop.addObjective({'obj_func': obj1}, {'obj_func': obj2})

# Add 1 single acquisition function
for i in range(1):
    moop.addAcquisition({'acquisition': RandomConstraint, 'hyperparams': {}})

# Solve the problem with 10 iterations with checkpointing on
moop.solve(10)

# Check that 30 simulations were evaluated and solutions are feasible
assert(moop.getSimulationData()["my sim"].size == 30)
assert(moop.getObjectiveData().size == 30)
assert(moop.getPF().size > 0)
