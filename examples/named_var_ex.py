
import numpy as np
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import LocalGPS

my_moop = MOOP(LocalGPS)

# Define a simulation to use below
def sim_func(x):
    if x["MyCat"] == 0:
        return np.array([(x["MyDes"]) ** 2, (x["MyDes"] - 1.0) ** 2])
    else:
        return np.array([99.9, 99.9])

# Add a design variable, simulation, objective, and constraint.
# Note the 'name' keys for each
my_moop.addDesign({'name': "MyDes",
                   'des_type': "continuous",
                   'lb': 0.0, 'ub': 1.0})
my_moop.addDesign({'name': "MyCat",
                   'des_type': "categorical",
                   'levels': 2})

my_moop.addSimulation({'name': "MySim",
                       'm': 2,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': 20}})

my_moop.addObjective({'name': "MyObj",
                      'obj_func': lambda x, s: sum(s["MySim"])})

my_moop.addConstraint({'name': "MyCon",
                       'constraint': lambda x, s: 0.1 - x["MyDes"]})

# Extract final objective and simulation databases
des_dtype = my_moop.getDesignType()
obj_dtype = my_moop.getObjectiveType()
sim_dtype = my_moop.getSimulationType()

# Print the data types
print("Design variable type:   " + str(des_dtype))
print("Simulation output type: " + str(sim_dtype))
print("Objective type:         " + str(obj_dtype))
