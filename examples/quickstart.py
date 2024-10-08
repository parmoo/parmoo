
import numpy as np
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.acquisitions import RandomConstraint
from parmoo.optimizers import GlobalSurrogate_PS

# Fix the random seed for reproducibility using the np_random_gen hyperparams
my_moop = MOOP(GlobalSurrogate_PS, hyperparams={'np_random_gen': 0})

my_moop.addDesign({'name': "x1",
                   'des_type': "continuous",
                   'lb': 0.0, 'ub': 1.0})
# Note: the 'levels' key can contain a list of strings, but jax can only jit
# numeric types, so integer level IDs are strongly recommended
my_moop.addDesign({'name': "x2", 'des_type': "categorical",
                   'levels': [-1, 1]})

def sim_func(x):
   sx = np.array([(x["x1"] - 0.2) ** 2, (x["x1"] - 0.8) ** 2])
   ## The following 2 lines are equivalent, but jax cannot jit if statements.
   ## Uncomment below to see the difference in execution speed from jit
   # if x["x2"] != 1: sx += 99.
   sx += 99. - 99. * (x["x2"] == 1)
   return sx

my_moop.addSimulation({'name': "MySim",
                       'm': 2,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': GaussRBF,
                       'hyperparams': {'search_budget': 20}})

def f1(x, s): return s["MySim"][0]
def f2(x, s): return s["MySim"][1]
my_moop.addObjective({'name': "f1", 'obj_func': f1})
my_moop.addObjective({'name': "f2", 'obj_func': f2})

def c1(x, s): return 0.1 - x["x1"]
my_moop.addConstraint({'name': "c1", 'constraint': c1})

for i in range(3):
   my_moop.addAcquisition({'acquisition': RandomConstraint,
                           'hyperparams': {}})

my_moop.solve(5)
results = my_moop.getPF(format="pandas")

# Display solution
print(results)

# Plot results -- must have extra viz dependencies installed
from parmoo.viz import scatter
# The optional arg `output` exports directly to jpeg instead of interactive mode
scatter(my_moop, output="jpeg")
