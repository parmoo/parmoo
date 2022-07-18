
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
    return np.array([(x[0]) ** 2, (x[0] - 1.0) ** 2])

# Add a design variable, simulation, objective, and constraint, w/o name key
moop.addDesign({'des_type': "continuous",
                   'lb': 0.0, 'ub': 1.0})

moop.addSimulation({'m': 2,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': 20}})

moop.addObjective({'obj_func': lambda x, s: sum(s)})

moop.addConstraint({'constraint': lambda x, s: 0.1 - x[0]})

# Extract numpy dtypes for all of this MOOP's inputs/outputs
des_dtype = moop.getDesignType()
sim_dtype = moop.getSimulationType()
obj_dtype = moop.getObjectiveType()
const_dtype = moop.getConstraintType()

# Display the dtypes as strings
print("Design variable type:   " + str(des_dtype))
print("Simulation output type: " + str(sim_dtype))
print("Objective type:         " + str(obj_dtype))
print("Constraint type:        " + str(const_dtype))
print()

# Add one acquisition and solve with 0 iterations to initialize databases
moop.addAcquisition({'acquisition': UniformWeights})
moop.solve(0)

# Extract final objective and simulation databases
obj_db = moop.getObjectiveData()
sim_db = moop.getSimulationData()

# Print the objective database dtypes
print("Objective database keys: " + str([key for key in obj_db.keys()]))
for key in obj_db.keys():
    print("\t'" + key + "'" + " dtype: " + str(obj_db[key].dtype))
    print("\t'" + key + "'" + " shape: " + str(obj_db[key].shape))
print()

# Print the simulation database dtypes
print("Simulation database type: " + str(type(sim_db)))
print("Simulation database length: " + str(len(sim_db)))
for i, dbi in enumerate(sim_db):
    print("\tsim_db[" + str(i) + "] database keys: " +
          str([key for key in dbi.keys()]))
    for key in dbi.keys():
        print("\t\t'" + key + "'" + " dtype: " + str(dbi[key].dtype))
        print("\t\t'" + key + "'" + " shape: " + str(dbi[key].shape))

# Display solution
vizTest(moop)
