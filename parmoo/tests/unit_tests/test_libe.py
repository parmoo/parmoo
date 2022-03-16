import numpy as np
from parmoo.optimizers import LocalGPS
from parmoo.surrogates import GaussRBF
from parmoo.searches import LatinHypercube
from parmoo.acquisitions import RandomConstraint
import pytest


@pytest.mark.extra
def test_libE_parmoo_persis_gen_unnamed():
    try:
        from parmoo.extras.libe import libE_MOOP, parmoo_persis_gen
    except BaseException:
        pytest.skip("libEnsemble or its dependencies not importable. " +
                    "Skipping.")

    # Solve a 5d problem with 3 objectives
    n = 5
    o = 3
    
    def dtlz2_sim(x):
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
    
    # Create a libE_MOOP
    moop = libE_MOOP(LocalGPS, hyperparams={})
    # Add n design vars
    for i in range(n):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    # Add simulation
    moop.addSimulation({'m': o,
                        'sim_func': dtlz2_sim,
                        'hyperparams': {'search_budget': 100},
                        'search': LatinHypercube,
                        'surrogate': GaussRBF,
                        'sim_db': {},
                        'des_tol': 0.00000001})
    # Add o objectives
    def obj1(x, s): return s[0]
    def obj2(x, s): return s[1]
    def obj3(x, s): return s[2]
    moop.addObjective({'obj_func': obj1})
    moop.addObjective({'obj_func': obj2})
    moop.addObjective({'obj_func': obj3})
    # Add 4 acquisition functions
    for i in range(4):
        moop.addAcquisition({'acquisition': RandomConstraint})

    # Create libE dictionaries
    libE_info = {'comm': {}}
    persis_info = {'moop': moop.moop,
                   'nworkers': 4}
    #sim_specs = {'sim_f': moop.moop_sim,
    #             'in': ['x', 'sim_name'],
    #             'out': [('f', float, max_m)]}
    gen_specs = {'gen_f': parmoo_persis_gen,
                 'persis_in': ['x', 'sim_name', 'f'],
                 'out': [('x', float, n),
                         ('sim_name', int)],
                 'user': {}}
    H = []
    
    # Solve
    H, persis_info, exit_code = parmoo_persis_gen(H, persis_info, gen_specs,
                                                  libE_info)

    # Check that output is correct
    assert(persis_info['moop'].getObjectiveData()['x_vals'].shape[0] == 200)
    
#@pytest.mark.extra
#def test_libE_parmoo_persis_gen_named()
#    try:
#        from parmoo.extras.libe import libE_MOOP
#    except:
#        pytest.skip("libEnsemble or its dependencies not importable. " +
#                    "Skipping.")
#
#    # Solve a 5d problem with 3 objectives
#    n = 5
#    o = 3
#    
#    def dtlz2_sim_named(x):
#        """ Evaluates the sim function for a collection of points given in
#        ``H['x']``.
#    
#        """
#    
#        # Unpack names into array
#        xx = np.zeros(n)
#        for i, name in enumerate(moop.moop.des_names):
#            xx[i] = x[name[0]]
#        # Use dtlz2_sim to evaluate
#        return dtlz2_sim(xx)
#
#    # Create a libE_MOOP with named variables
#    moop = libE_MOOP(LocalGPS, hyperparams={})
#    # Add n design vars
#    for i in range(n):
#        moop.addDesign({'name': "x" + str(i + 1), 'lb': 0.0, 'ub': 1.0})
#    
#    
#    # Add simulation
#    moop.addSimulation({'name': "DTLZ2",
#                        'm': o,
#                        'sim_func': dtlz2_sim_named,
#                        'hyperparams': {'search_budget': 100},
#                        'search': LatinHypercube,
#                        'surrogate': GaussRBF,
#                        'sim_db': {},
#                        'des_tol': 0.00000001})
#    # Add o objectives
#    def obj1(x, s): return s['DTLZ2'][0]
#    def obj2(x, s): return s['DTLZ2'][1]
#    def obj3(x, s): return s['DTLZ2'][2]
#    moop.addObjective({'name': "obj1", 'obj_func': obj1})
#    moop.addObjective({'name': "obj2", 'obj_func': obj2})
#    moop.addObjective({'name': "obj3", 'obj_func': obj3})
#    # Add 4 acquisition functions
#    for i in range(4):
#        moop.addAcquisition({'acquisition': RandomConstraint})
#    
#    # Solve
#    moop.solve()
#    assert(moop.getObjectiveData()['x1'].shape[0] == 200)

if __name__ == "__main__":
    test_libE_parmoo_persis_gen_unnamed()
