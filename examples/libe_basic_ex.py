
import numpy as np
from parmoo.extras.libe import libE_MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import LocalGPS
from parmoo.viz import *

# Create a libE_MOOP
moop = libE_MOOP(LocalGPS)

# Add 2 design variables (one continuous and one categorical)
moop.addDesign({'name': "x1",
                   'des_type': "continuous",
                   'lb': 0.0, 'ub': 1.0})
moop.addDesign({'name': "x2", 'des_type': "categorical",
                   'levels': 3})

# All functions are defined below.

def sim_func(x):
   if x["x2"] == 0:
      return np.array([(x["x1"] - 0.2) ** 2, (x["x1"] - 0.8) ** 2])
   else:
      return np.array([99.9, 99.9])

def obj_f1(x, s):
    return s["MySim"][0]

def obj_f2(x, s):
    return s["MySim"][1]

def const_c1(x, s):
    return 0.1 - x["x1"]

# Add the simulation (note the budget of 20 sim evals during the search phase)
moop.addSimulation({'name': "MySim",
                       'm': 2,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': 20}})

# Add the objectives
moop.addObjective({'name': "f1", 'obj_func': obj_f1})
moop.addObjective({'name': "f2", 'obj_func': obj_f2})

# Add the constraint
moop.addConstraint({'name': "c1", 'constraint': const_c1})

# Add 3 acquisition functions
for i in range(3):
   moop.addAcquisition({'acquisition': UniformWeights,
                           'hyperparams': {}})

# Turn on checkpointing -- creates the files parmoo.moop and parmoo.surrogate.1
moop.setCheckpoint(True, checkpoint_data=False, filename="parmoo")

# Use sim_max = 30 to perform just 30 simulations
moop.solve(sim_max=30)
results = moop.getPF()

# Display the solution
print(results, "\n dtype=" + str(results.dtype))

# Display solution
vizTest(moop)
