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
from parmoo.optimizers import LocalGPS
from parmoo.surrogates import GaussRBF
from parmoo.searches import LatinHypercube
from parmoo.acquisitions import RandomConstraint

# Solve a 5d problem with 3 objectives
n = 5
o = 3

# Define functions for unnamed runs

def dtlz2_sim_unnamed(x):
    """ Evaluates the sim function for a collection of points given in
    ``H['x']``.

    """

    import math

    # Create output array for sim outs
    f = np.zeros(o)
    # Compute the kernel function g(x)
    gx = np.dot(x[o-1:n]-0.5, x[o-1:n]-0.5)
    # Compute the simulation outputs
    f[0] = (1.0 + gx)
    for y in x[:o-1]:
        f[0] *= math.cos(math.pi * y / 2.0)
    for i in range(1, o):
        f[i] = (1.0 + gx) * math.sin(math.pi * x[o-1-i] / 2.0)
        for y in x[:o-1-i]:
            f[i] *= math.cos(math.pi * y / 2.0)
    return f
    
def obj1_unnamed(x, s): return s[0]
def obj2_unnamed(x, s): return s[1]
def obj3_unnamed(x, s): return s[2]

# Define functions for named runs

def dtlz2_sim_named(x):
    """ Evaluates the sim function for a collection of points given in
    ``H['x']``.

    """

    # Unpack names into array
    xx = np.zeros(n)
    for i, name in enumerate(moop.moop.des_names):
        xx[i] = x[name[0]]
    # Use dtlz2_sim to evaluate
    return dtlz2_sim_unnamed(xx)

def obj1_named(x, s): return s['DTLZ2'][0]
def obj2_named(x, s): return s['DTLZ2'][1]
def obj3_named(x, s): return s['DTLZ2'][2]

# On MacOS and Windows, libE runs using Python MP must be
# enclosed in an "if __name__ == '__main__':" block, as below
if __name__ == "__main__":
    # Create a libE_MOOP
    moop = libE_MOOP(LocalGPS, hyperparams={})
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
    moop.solve()
    assert(moop.getObjectiveData()['x_vals'].shape[0] == 200)
    
    #print(moop.getPF())
    
    # Create a libE_MOOP with named variables
    moop = libE_MOOP(LocalGPS, hyperparams={})
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
    moop.solve()
    assert(moop.getObjectiveData()['x1'].shape[0] == 200)
