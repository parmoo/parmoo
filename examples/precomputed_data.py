
import numpy as np
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import GlobalSurrogate_PS

my_moop = MOOP(GlobalSurrogate_PS)

my_moop.addDesign({'name': "x1",
                   'des_type': "continuous",
                   'lb': 0.0, 'ub': 1.0})
my_moop.addDesign({'name': "x2", 'des_type': "categorical",
                   'levels': 3})

def sim_func(x):
   if x["x2"] == 0:
      return np.array([(x["x1"] - 0.2) ** 2, (x["x1"] - 0.8) ** 2])
   else:
      return np.array([99.9, 99.9])

my_moop.addSimulation({'name': "MySim",
                       'm': 2,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': 20}})

my_moop.addObjective({'name': "f1", 'obj_func': lambda x, s: s["MySim"][0]})
my_moop.addObjective({'name': "f2", 'obj_func': lambda x, s: s["MySim"][1]})

my_moop.addAcquisition({'acquisition': UniformWeights})

# This step is needed to finalize the MOOP definition. If you are using the
# solve command it is done automatically, but it must be done manually before
# any pre-existing data can be added.
my_moop.compile()

# Precompute one simulation value for demo
des_val = np.zeros(1, dtype=[("x1", float), ("x2", int)])[0]
sim_val = sim_func(des_val)

# Add the precomputed simulation value from above
my_moop.updateSimDb(des_val, sim_val, "MySim")

# Get and display initial database
sim_db = my_moop.getSimulationData()
print(sim_db)
