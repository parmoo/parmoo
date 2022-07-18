import os
os.system("rm parmoo.moop")

import numpy as np
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import LocalGPS
from parmoo.viz import *

moop = MOOP(LocalGPS)

# Define a simulation to use below
def sim_func(x):
    if x["MyCat"] == 0:
        return np.array([(x["MyDes"]) ** 2, (x["MyDes"] - 1.0) ** 2])
    else:
        return np.array([99.9, 99.9])

# Add a design variable, simulation, objective, and constraint.
# Note the 'name' keys for each
moop.addDesign({'name': "MyDes",
                   'des_type': "continuous",
                   'lb': 0.0, 'ub': 1.0})
moop.addDesign({'name': "MyCat",
                   'des_type': "categorical",
                   'levels': 2})

moop.addSimulation({'name': "MySim",
                       'm': 2,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': 20}})

moop.addObjective({'name': "MyObj",
                      'obj_func': lambda x, s: sum(s["MySim"])})

moop.addConstraint({'name': "MyCon",
                       'constraint': lambda x, s: 0.1 - x["MyDes"]})

# # Extract numpy dtypes for all of this MOOP's inputs/outputs
# des_dtype = moop.getDesignType()
# obj_dtype = moop.getObjectiveType()
# sim_dtype = moop.getSimulationType()
#
# # Display the dtypes as strings
# print("Design variable type:   " + str(des_dtype))
# print("Simulation output type: " + str(sim_dtype))
# print("Objective type:         " + str(obj_dtype))
#
# # display Objectives
# printObjectives(moop)
#
# # display PF
# printPF(moop)

# # Get and print full simulation database
# sim_db = moop.getSimulationData()
# print("Simulation data:")
# print(sim_db)

# Display solution
vizTest(moop)

