
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
    import pytest

    try:
        from parmoo.extras.libe import libE_MOOP
    except BaseException:
        pytest.skip("libEnsemble or its dependencies not importable. " +
                    "Skipping.")

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
    assert(moop.moop.n == n)

    # Add simulation
    moop.addSimulation({'name': "Eye",
                        'm': o,
                        'sim_func': id_named,
                        'hyperparams': {'search_budget': 100},
                        'search': LatinHypercube,
                        'surrogate': GaussRBF,
                        'sim_db': {},
                        'des_tol': 0.00000001})
    assert(moop.moop.m_total == o)

    # Add o objectives
    def obj1(x, s): return s['Eye'][0]
    def obj2(x, s): return s['Eye'][1]
    def obj3(x, s): return s['Eye'][2]
    moop.addObjective({'name': "obj1", 'obj_func': obj1})
    moop.addObjective({'name': "obj2", 'obj_func': obj2})
    moop.addObjective({'name': "obj3", 'obj_func': obj3})
    assert(moop.moop.o == 3)

    # Add 1 constraint
    def const1(x, s): return x["x1"] - 0.5
    moop.addConstraint({'name': "c1", 'constraint': const1})
    assert(moop.moop.p == 1)

    # Add 4 acquisition functions
    for i in range(4):
        moop.addAcquisition({'acquisition': RandomConstraint})
    assert(len(moop.moop.acquisitions) == 4)

    # Check Pareto front, objective data, sim data
    assert(moop.getPF()['x1'].shape[0] == 0)
    assert(moop.getObjectiveData()['x1'].shape[0] == 0)
    assert(moop.getSimulationData()['Eye']['x1'].shape[0] == 0)


if __name__ == "__main__":
    test_libE_parmoo_persis_gen()
    test_libE_MOOP()
