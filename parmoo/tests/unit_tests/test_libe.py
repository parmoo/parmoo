
# @pytest.mark.extra
def test_libE_parmoo_persis_gen():
    """ Test the parmoo_persis_gen function in extras/libe.py.

    Generate several bad inputs to check for bad input.

    """

    import pytest

    try:
        from parmoo.extras.libe import parmoo_persis_gen
    except BaseException:
        pytest.skip("libEnsemble or its dependencies not importable. " +
                    "Skipping.")

    # Create libE dictionaries
    libE_info = {'comm': {}}
    persis_info = {'nworkers': 4}
    gen_specs = {'gen_f': parmoo_persis_gen,
                 'persis_in': ['x', 'sim_name', 'f'],
                 'out': [('x', float, 3),
                         ('sim_name', int)],
                 'user': {}}
    H = []

    # Try persis_gen with no persis_info['moop'] key
    with pytest.raises(KeyError):
        H, persis_info, exit_code = parmoo_persis_gen(H, persis_info,
                                                      gen_specs,
                                                      libE_info)
    # Try persis_gen with bad persis_info['moop'] key
    persis_info['moop'] = "hello world"
    with pytest.raises(TypeError):
        H, persis_info, exit_code = parmoo_persis_gen(H, persis_info,
                                                      gen_specs,
                                                      libE_info)


# @pytest.mark.extra
def test_libE_MOOP():
    """ Test the libE_MOOP class from extras/libe.py.

    Create a problem and check that the databases are empty.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import RandomConstraint
    from parmoo.optimizers import LocalGPS
    import numpy as np
    import os
    import pytest

    try:
        from parmoo.extras.libe import libE_MOOP
    except BaseException:
        pytest.skip("libEnsemble or its dependencies not importable. " +
                    "Skipping.")

    # Make all functions global for save/load
    global id_named, obj1, obj2, obj3, const1

    # Create a 5d problem with 3 objectives
    n = 5
    o = 3

    def id_named(x):
        """ Evaluates the sim function for a collection of points given in
        ``H['x']``.

        """

        # Unpack names into array
        xx = np.zeros(n)
        for i, name in enumerate(moop.moop.des_names):
            xx[i] = x[name[0]]
        # Return ID
        return xx[:3]

    # Create a libE_MOOP with named variables
    moop = libE_MOOP(LocalGPS)
    assert(isinstance(moop.moop, MOOP))
    moop = libE_MOOP(LocalGPS, hyperparams={})
    assert(isinstance(moop.moop, MOOP))
    # Add n design vars
    for i in range(n):
        moop.addDesign({'name': "x" + str(i + 1), 'lb': 0.0, 'ub': 1.0})
    assert(len(moop.getDesignType()) == n)
    assert(all([dt[1] == "f8" for dt in moop.getDesignType()]))
    # Add simulation
    moop.addSimulation({'name': "Eye",
                        'm': o,
                        'sim_func': id_named,
                        'hyperparams': {'search_budget': 100},
                        'search': LatinHypercube,
                        'surrogate': GaussRBF,
                        'sim_db': {},
                        'des_tol': 0.00000001})
    assert(len(moop.getSimulationType()) == 1)
    assert(all([dt[1] == "f8" and dt[2] == o
                for dt in moop.getSimulationType()]))
    # Add o objectives
    def obj1(x, s): return s['Eye'][0]
    def obj2(x, s): return s['Eye'][1]
    def obj3(x, s): return s['Eye'][2]
    moop.addObjective({'name': "obj1", 'obj_func': obj1})
    moop.addObjective({'name': "obj2", 'obj_func': obj2})
    moop.addObjective({'name': "obj3", 'obj_func': obj3})
    assert(len(moop.getObjectiveType()) == 3)
    assert(all([dt[1] == "f8" for dt in moop.getObjectiveType()]))
    # Add 1 constraint
    def const1(x, s): return x["x5"] - 0.5
    moop.addConstraint({'name': "c1", 'constraint': const1})
    assert(len(moop.getConstraintType()) == 1)
    assert(all([dt[1] == "f8" for dt in moop.getConstraintType()]))
    # Add 4 acquisition functions
    for i in range(4):
        moop.addAcquisition({'acquisition': RandomConstraint})
    assert(len(moop.moop.acquisitions) == 4)
    # Perform 0 iteration manually
    batch = moop.iterate(0)
    for (xi, i) in batch:
        moop.evaluateSimulation(xi, i)
    moop.updateAll(0, batch)
    # Add a value in the simulation database
    x_val = np.zeros(1, dtype=moop.getDesignType())[0]
    sx_val = np.zeros(1, dtype=moop.getSimulationType())[0]
    moop.update_sim_db(x_val, sx_val["Eye"], "Eye")
    assert(np.all(moop.check_sim_db(x_val, "Eye") == 0))
    moop.addData(x_val, sx_val)
    # Check Pareto front, objective data, sim data
    assert(moop.getPF()['x1'].shape[0] == 1)
    assert(moop.getObjectiveData()['x1'].shape[0] == 101)
    assert(moop.getSimulationData()['Eye']['x1'].shape[0] == 101)
    # Test checkpointing features
    moop.setCheckpoint(True)
    moop.save()
    moop.load()
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.surrogate.1")


# @pytest.mark.extra
def test_libE_MOOP_bad_solve():
    """ Test the libE_MOOP.solve() method from extras/libe.py.

    Create a problem and check that the solve method handles bad input.

    """

    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import RandomConstraint
    from parmoo.optimizers import LocalGPS
    import pytest

    try:
        from parmoo.extras.libe import libE_MOOP
    except BaseException:
        pytest.skip("libEnsemble or its dependencies not importable. " +
                    "Skipping.")

    # Create a libE_MOOP
    moop = libE_MOOP(LocalGPS)

    # Add 1 design var
    moop.addDesign({'lb': 0.0, 'ub': 1.0})

    # Add 1 simulation
    def id_named(x): return x[0]
    moop.addSimulation({'name': "Eye",
                        'm': 1,
                        'sim_func': id_named,
                        'hyperparams': {'search_budget': 100},
                        'search': LatinHypercube,
                        'surrogate': GaussRBF,
                        'sim_db': {},
                        'des_tol': 0.00000001})

    # Add 1 objective
    def obj1(x, s): return s[0]
    moop.addObjective({'obj_func': obj1})

    # Add 1 acquisition function
    moop.addAcquisition({'acquisition': RandomConstraint})

    # Hard code bad CL args
    import sys
    sys.argv.append("--comms")
    sys.argv.append("local")
    sys.argv.append("--nworkers")
    sys.argv.append("1")

    # Solve with bad CL args
    with pytest.raises(ValueError):
        moop.solve()


if __name__ == "__main__":
    test_libE_parmoo_persis_gen()
    test_libE_MOOP()
    test_libE_MOOP_bad_solve()
