
import numpy as np
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import LocalGPS
import logging

# Create a new MOOP
my_moop = MOOP(LocalGPS)

# Add 1 continuous and 1 categorical design variable
my_moop.addDesign({'name': "x1",
                   'des_type': "continuous",
                   'lb': 0.0, 'ub': 1.0})
my_moop.addDesign({'name': "x2", 'des_type': "categorical",
                   'levels': 3})

# Create a simulation function
def sim_func(x):
   if x["x2"] == 0:
      return np.array([(x["x1"] - 0.2) ** 2, (x["x1"] - 0.8) ** 2])
   else:
      return np.array([99.9, 99.9])

# Add the simulation function to the MOOP
my_moop.addSimulation({'name': "MySim",
                       'm': 2,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': 20}})

# Define the 2 objectives as named Python functions
def obj1(x, s): return s["MySim"][0]
def obj2(x, s): return s["MySim"][1]

# Define the constraint as a function
def const(x, s): return 0.1 - x["x1"]

# Add 2 objectives
my_moop.addObjective({'name': "f1", 'obj_func': obj1})
my_moop.addObjective({'name': "f2", 'obj_func': obj2})

# Add 1 constraint
my_moop.addConstraint({'name': "c1", 'constraint': const})

# Add 3 acquisition functions (generates batches of size 3)
for i in range(3):
   my_moop.addAcquisition({'acquisition': UniformWeights,
                           'hyperparams': {}})

# Turn on logging with timestamps
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Use checkpointing without saving a separate data file (in "parmoo.moop" file)
my_moop.setCheckpoint(True, checkpoint_data=False, filename="parmoo")

# Solve the problem with 4 iterations
my_moop.solve(4)

# Create a new MOOP object and reload the MOOP from parmoo.moop file
new_moop = MOOP(LocalGPS)
new_moop.load("parmoo")

# Do another iteration
new_moop.solve(5)

# Display the solution
results = new_moop.getPF()
print(results, "\n dtype=" + str(results.dtype))
