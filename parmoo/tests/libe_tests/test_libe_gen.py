# """
# Apply the libE generator to dtlz2.
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_6-hump_camel_persistent_uniform_sampling.py
#    python3 test_6-hump_camel_persistent_uniform_sampling.py \
#           --nworkers 3 --comms local
#    python3 test_6-hump_camel_persistent_uniform_sampling.py \
#           --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
#
# """

""" Test the parmoo_gen generator function for libE.

Create a libEnsemble run and test the parmoo generator.

"""

import numpy as np
from parmoo.extras.libe import libE_MOOP
from parmoo.optimizers import GlobalSurrogate_PS
from parmoo.surrogates import GaussRBF
from parmoo.searches import LatinHypercube
from parmoo.acquisitions import RandomConstraint
from parmoo.tests.libe_tests.libe_funcs import *

# On MacOS and Windows, libE runs using Python MP must be
# enclosed in an "if __name__ == '__main__':" block, as below
if __name__ == "__main__":
    # Create a libE_MOOP
    moop = libE_MOOP(GlobalSurrogate_PS, hyperparams={})
    # Add n design vars
    for i in range(n):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    # Add simulation
    moop.addSimulation({'m': o,
                        'sim_func': dtlz2_sim_unnamed,
                        'hyperparams': {'search_budget': 100},
                        'search': LatinHypercube,
                        'surrogate': GaussRBF,
                        'sim_db': {},
                        'des_tol': 0.00000001})
    # Add o objectives
    moop.addObjective({'obj_func': obj1_unnamed})
    moop.addObjective({'obj_func': obj2_unnamed})
    moop.addObjective({'obj_func': obj3_unnamed})
    # Add 4 acquisition functions
    for i in range(4):
        moop.addAcquisition({'acquisition': RandomConstraint})
    
    # Solve
    moop.solve(sim_max=200)
    assert(moop.getObjectiveData()['x_vals'].shape[0] == 200)
    
    #print(moop.getPF())
    
    # Create a libE_MOOP with named variables
    moop = libE_MOOP(GlobalSurrogate_PS, hyperparams={})
    # Add n design vars
    for i in range(n):
        moop.addDesign({'name': "x" + str(i + 1), 'lb': 0.0, 'ub': 1.0})
    
    # Add simulation
    moop.addSimulation({'name': "DTLZ2",
                        'm': o,
                        'sim_func': dtlz2_sim_named,
                        'hyperparams': {'search_budget': 100},
                        'search': LatinHypercube,
                        'surrogate': GaussRBF,
                        'sim_db': {},
                        'des_tol': 0.00000001})
    # Add o objectives
    moop.addObjective({'name': "obj1", 'obj_func': obj1_named})
    moop.addObjective({'name': "obj2", 'obj_func': obj2_named})
    moop.addObjective({'name': "obj3", 'obj_func': obj3_named})
    # Add 4 acquisition functions
    for i in range(4):
        moop.addAcquisition({'acquisition': RandomConstraint})
    
    # Solve
    moop.solve(sim_max=200)
    assert(moop.getObjectiveData()['x1'].shape[0] == 200)
