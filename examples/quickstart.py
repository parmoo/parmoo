
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

moop.addDesign({'name': "A",
                   'des_type': "continuous",
                   'lb': 0.0, 'ub': 1.0})
moop.addDesign({'name': "B", 'des_type': "categorical",
                   'levels': 3})

def sim_func(x):
   if x["B"] == 0:
      return np.array([(x["A"] - 0.2) ** 2, (x["A"] - 0.8) ** 2])
   else:
      return np.array([99.9, 99.9])

moop.addSimulation({'name': "MySim",
                       'm': 2,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': 20}})

moop.addObjective({'name': "C", 'obj_func': lambda x, s: s["MySim"][0]})
moop.addObjective({'name': "D", 'obj_func': lambda x, s: s["MySim"][1]})

moop.addConstraint({'name': "E", 'constraint': lambda x, s: 0.1 - x["A"]})

for i in range(3):
   moop.addAcquisition({'acquisition': UniformWeights,
                           'hyperparams': {}})

moop.solve(5)

# Display solution
vizTest(moop)
