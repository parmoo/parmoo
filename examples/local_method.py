
import numpy as np
import pandas as pd
from parmoo import MOOP
from parmoo.searches import LatinHypercube
from parmoo.surrogates import LocalGaussRBF
from parmoo.acquisitions import RandomConstraint, FixedWeights
from parmoo.optimizers import TR_LBFGSB
from parmoo.objectives import single_sim_out
import logging

# Fix the random seed for reproducibility
np.random.seed(0)

# Switch to using the TR_LBFGSB solver to solve surrogate problems in
# a trust region with multi-start LBFGSB
my_moop = MOOP(TR_LBFGSB)

# Massive 50-variable black-box optimization problem
# Completely hopeless for methods that rely on global models
for i in range(1, 51):
    my_moop.addDesign({'name': f"x{i}",
                       'des_type': "continuous",
                       'lb': 0.0, 'ub': 1.0})

# A simple convex simulation output with 2 outputs
def sim_func(x):
    xx = np.zeros(50)
    for i in range(50):
        xx[i] = x[f"x{i+1}"]
    # 25 variables that don't affect tradeoff, but need to be minimized
    tail = np.linalg.norm(xx[25:] - 0.5) ** 2 / 25
    # 25 variables that do affect tradeoff
    s1 = np.linalg.norm(xx[:25] - 0.2) ** 2 / 25 + tail
    s2 = np.linalg.norm(xx[:25] - 0.8) ** 2 / 25 + tail
    return np.array([s1, s2])

# Using a local surrogate to dodge the curse of dimensionality
# Notice that search_budget has to be greater than the number of variables
my_moop.addSimulation({'name': "MySim",
                       'm': 2,
                       'sim_func': sim_func,
                       'search': LatinHypercube,
                       'surrogate': LocalGaussRBF,
                       'hyperparams': {'search_budget': 200}})

# 2 objectives (using the single_sim_out library objective to minimize a
# single output of the simulation function)
my_moop.addObjective({'name': "f1",
                      'obj_func': single_sim_out(my_moop.getDesignType(),
                                                 my_moop.getSimulationType(),
                                                 ("MySim", 0))})
my_moop.addObjective({'name': "f2",
                      'obj_func': single_sim_out(my_moop.getDesignType(),
                                                 my_moop.getSimulationType(),
                                                 ("MySim", 1))})

# When solving big problems, it's often better to fix some acquisitions so
# we can focus on a few high-quality solutions
my_moop.addAcquisition({'acquisition': FixedWeights,
                        'hyperparams': {'weights': np.eye(2)[0]}})
my_moop.addAcquisition({'acquisition': FixedWeights,
                        'hyperparams': {'weights': np.eye(2)[1]}})
my_moop.addAcquisition({'acquisition': FixedWeights,
                        'hyperparams': {'weights': np.ones(2) / 2}})
# Keep one randomized acquisition to get some coverage of the Pareto front
my_moop.addAcquisition({'acquisition': RandomConstraint,
                        'hyperparams': {}})

# Turn on logging with timestamps
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# 200 iterations * 4 acquisition funcs + 200 point search = 1000 eval budget
# This could take a few mins to run...
my_moop.solve(200)

# Display the values of x26, ..., x50 for all solution points
results = my_moop.getPF(format="pandas")
results[[f"x{i}" for i in range(26, 51)]].to_csv("local_method.csv")

# Plot results -- must have extra viz dependencies installed
from parmoo.viz import scatter
# The optional arg `output` exports directly to jpg instead of interactive mode
scatter(my_moop, output="jpeg")
