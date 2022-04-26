
import numpy as np
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import LocalGPS

my_moop = MOOP(LocalGPS)

# Define a simulation to use below
def sim_func(x):
    return np.array([(x["MyDes"]) ** 2, (x["MyDes"] - 1.0) ** 2])

# Add a design variable, simulation, objective, and constraint.
# Note the 'name' keys for each
my_moop.addDesign({'name': "MyDes",
                   'des_type': "continuous",
                   'lb': 0.0, 'ub': 1.0})

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

# Add one acquisition and solve with 0 iterations to initialize databases
my_moop.addAcquisition({'acquisition': UniformWeights})
my_moop.solve(0)

# Extract final objective and simulation databases
obj_db = my_moop.getObjectiveData()
sim_db = my_moop.getSimulationData()

# Print the data types
print("objective database type: " + str(obj_db.dtype))
print("Simulation database keys: " + str([key for key in sim_db.keys()]))
for key in sim_db.keys():
    print("'" + key + "'" + " database type: " + str(sim_db[key].dtype))
