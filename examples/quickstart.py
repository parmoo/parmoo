
import numpy as np
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import LocalGPS

my_moop = MOOP(LocalGPS)

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

my_moop.addConstraint({'name': "c1", 'constraint': lambda x, s: 0.1 - x["x1"]})

for i in range(3):
   my_moop.addAcquisition({'acquisition': UniformWeights,
                           'hyperparams': {}})

my_moop.solve(5)
results = my_moop.getPF()

# Display solution
print(results, "\n dtype=" + str(results.dtype))
