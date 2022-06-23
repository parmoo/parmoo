import os
os.system("rm parmoo.moop")

import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from parmoo import MOOP
from parmoo.acquisitions import RandomConstraint
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.optimizers import LBFGSB
from parmoo.objectives.dtlz import dtlz2_obj
from parmoo.simulations.dtlz import g2_sim
from parmoo.viz.plots import *

n = 5 # number of design variables
o = 3 # number of objectives
q = 5 # batch size (number of acquisitions)

# Create MOOP
moop = MOOP(LBFGSB)
# Add n design variables
for i in range(n):
    moop.addDesign({'name': f"x{i+1}", 'des_type': 'continuous',
                    'lb': 0.0, 'ub': 1.0, 'des_tol': 1.0e-8})

# Create the g2 simulation
moop.addSimulation({'name': "g2",
                    'm': 1,
                    'sim_func': g2_sim(moop.getDesignType(),
                                       num_obj=o, offset=0.5),
                    'search': LatinHypercube,
                    'surrogate': GaussRBF,
                    'hyperparams': {'search_budget': 10*n}})
# Add o objectives
for i in range(o):
    moop.addObjective({'name': f"DTLZ2 Obj {i+1}",
                        'obj_func': dtlz2_obj(moop.getDesignType(),
                                              moop.getSimulationType(),
                                              i, num_obj=o)})
# Add q acquisition functions
for i in range(q):
    moop.addAcquisition({'acquisition': RandomConstraint})
# Solve the MOOP with 20 iterations
moop.solve(20) 

printMOOP(moop)