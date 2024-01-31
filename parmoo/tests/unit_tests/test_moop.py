
def test_MOOP_init():
    """ Check that the MOOP class handles initialization properly.

    Initialize several MOOP objects, and check that their internal fields
    appear correct.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    import pytest

    # Try several invalid inputs
    with pytest.raises(TypeError):
        MOOP(5.0)
    with pytest.raises(TypeError):
        MOOP(lambda w, x, y, z: 0.0)
    with pytest.raises(TypeError):
        MOOP(LocalSurrogate_PS, hyperparams=[])
    # Try a few valid inputs
    moop = MOOP(LocalSurrogate_PS)
    assert (moop.n_feature == 0 and moop.n_latent == 0 and moop.s == 0 and
            moop.m_total == 0 and moop.o == 0 and moop.p == 0)
    moop = MOOP(LocalSurrogate_PS, hyperparams={'test': 0})
    assert (moop.n_feature == 0 and moop.n_latent == 0 and moop.s == 0 and
            moop.m_total == 0 and moop.o == 0 and moop.p == 0)
    assert (moop.hyperparams['test'] == 0)


def test_MOOP_addSimulation():
    """ Check that the MOOP class handles adding new simulations properly.

    Initialize several MOOPs, and add several simulations. Check that
    the metadata is updated correctly.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    import numpy as np
    import pytest

    # Create 4 SimGroups for later
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'m': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    g3 = {'name': "Bobo1",
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g4 = {'name': "Bobo2",
          'm': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    # Initialize a MOOP and add 3 design variables
    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    # Now add one simulation and check
    moop.addSimulation(g1)
    assert (moop.n_latent == 3 and moop.s == 1 and moop.m_total == 1
            and moop.o == 0 and moop.p == 0)
    # Initialize another MOOP with 3 design variables
    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addSimulation(g1, g2)
    assert (moop.n_latent == 3 and moop.s == 2 and moop.m_total == 3
            and moop.o == 0 and moop.p == 0)
    moop.addSimulation(g3, g4)
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addSimulation(g4)
    # Check the names
    assert (moop.sim_schema[0][0] == "sim1")
    assert (moop.sim_schema[1][0] == "sim2")
    assert (moop.sim_schema[2][0] == "Bobo1")
    assert (moop.sim_schema[3][0] == "Bobo2")


def test_pack_unpack_sim():
    """ Check that the MOOP class handles simulation packing correctly.

    Initialize a MOOP objecti with and without design variable names.
    Add 2 simulations and pack/unpack each output.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    import numpy as np

    # Create 2 simulations for later
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'m': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    # Create a MOOP and test simulation unpacking
    moop = MOOP(LocalSurrogate_PS)
    # Add two continuous variables and two simulations
    moop.addDesign({'name': "x0", 'lb': 0.0, 'ub': 1000.0},
                   {'name': "x1", 'lb': -1.0, 'ub': 0.0})
    moop.addSimulation(g1, g2)
    # Create a solution vector
    sx = np.array([1.0, 2.0, 3.0])
    sxx = np.zeros(1, dtype=moop.sim_schema)
    sxx[0]['sim1'] = 1.0
    sxx[0]['sim2'][:] = np.array([2.0, 3.0])
    # Check packing
    assert (all(moop._pack_sim(sxx) == sx))
    # Check unpacking
    assert (moop._unpack_sim(sx)['sim1'] == sxx[0]['sim1'])
    assert (moop._unpack_sim(sx)['sim2'][0] == sxx[0]['sim2'][0])
    assert (moop._unpack_sim(sx)['sim2'][1] == sxx[0]['sim2'][1])


def test_MOOP_addObjective():
    """ Check that the MOOP class handles adding objectives properly.

    Initialize a MOOP object and check that the addObjective() function works
    correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np
    import pytest

    # Create 2 SimGroups for later
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups, one of which has 2 outputs
    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(g1, g2)
    # Try to add bad objectives and check that appropriate errors are raised
    with pytest.raises(TypeError):
        moop.addObjective(0)
    with pytest.raises(AttributeError):
        moop.addObjective({})
    with pytest.raises(TypeError):
        moop.addObjective({'obj_func': 0})
    with pytest.raises(ValueError):
        moop.addObjective({'obj_func': lambda x: 0.0})
    # Add an objective after an acquisition
    with pytest.raises(RuntimeError):
        moop1 = MOOP(LocalSurrogate_PS)
        moop1.acquisitions.append(0)
        moop1.addObjective({'obj_func': lambda x, s: 0.0})
    # Check that no objectives were added yet
    assert (moop.o == 0)
    # Now add 3 good objectives
    moop.addObjective({'obj_func': lambda x, s: x[0]})
    assert (moop.o == 1)
    moop.addObjective({'obj_func': lambda x, s: s[0]},
                      {'obj_func': lambda x, s, der=0: s[1]})
    assert (moop.o == 3)
    moop.addObjective({'name': "Bobo", 'obj_func': lambda x, s: s[0]})
    assert (moop.o == 4)
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addObjective({'name': "Bobo", 'obj_func': lambda x, s: s[0]})
    assert (moop.obj_schema[0] == ("f1", 'f8'))
    assert (moop.obj_schema[1] == ("f2", 'f8'))
    assert (moop.obj_schema[2] == ("f3", 'f8'))
    assert (moop.obj_schema[3] == ("Bobo", 'f8'))


def test_MOOP_addConstraint():
    """ Check that the MOOP class handles adding constraints properly.

    Initialize a MOOP object and check that the addConstraint() function works
    correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np
    import pytest

    # Create 2 SimGroups for later
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups, one of which has 2 outputs
    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(g1, g2)
    # Try to add bad constraints and check that appropriate errors are raised
    with pytest.raises(TypeError):
        moop.addConstraint(0)
    with pytest.raises(AttributeError):
        moop.addConstraint({})
    with pytest.raises(TypeError):
        moop.addConstraint({'constraint': 0})
    with pytest.raises(ValueError):
        moop.addConstraint({'constraint': lambda x: 0.0})
    # Check that no constraints were added yet
    assert (moop.p == 0)
    # Now add 3 good constraints
    moop.addConstraint({'constraint': lambda x, s: x[0]})
    assert (moop.p == 1)
    moop.addConstraint({'constraint': lambda x, s: s[0]},
                       {'constraint': lambda x, s, der=0: s[1] + s[2]})
    assert (moop.p == 3)
    moop.addConstraint({'name': "Bobo", 'constraint': lambda x, s: s[0]})
    assert (moop.p == 4)
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addConstraint({'name': "Bobo", 'constraint': lambda x, s: s[0]})
    assert (moop.con_schema[0] == ("c1", 'f8'))
    assert (moop.con_schema[1] == ("c2", 'f8'))
    assert (moop.con_schema[2] == ("c3", 'f8'))
    assert (moop.con_schema[3] == ("Bobo", 'f8'))


def test_MOOP_addAcquisition():
    """ Check that the MOOP class handles adding acquisitions properly.

    Initialize a MOOP object and check that the addAcquisition() function works
    correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.acquisitions import UniformWeights
    import numpy as np
    import pytest

    # Create 2 SimGroups for later
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups, one of which has 2 outputs
    moop = MOOP(LocalSurrogate_PS)
    # Try to add acquisition functions without design variables
    with pytest.raises(RuntimeError):
        moop.addAcquisition({'acquisition': UniformWeights})
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    # Try to add acquisition functions without objectives
    with pytest.raises(RuntimeError):
        moop.addAcquisition({'acquisition': UniformWeights})
    moop.addSimulation(g1, g2)
    moop.addObjective({'obj_func': lambda x, s: s[0]},
                      {'obj_func': lambda x, s: s[1]},
                      {'obj_func': lambda x, s: s[2]})
    # Try to add bad acquisition functions and check for an appropriate error
    with pytest.raises(TypeError):
        moop.addAcquisition(0)
    with pytest.raises(AttributeError):
        moop.addAcquisition({})
    with pytest.raises(TypeError):
        moop.addAcquisition({'acquisition': UniformWeights,
                             'hyperparams': 0})
    with pytest.raises(TypeError):
        moop.addAcquisition({'acquisition': 0,
                             'hyperparams': {}})
    with pytest.raises(TypeError):
        moop.addAcquisition({'acquisition': GaussRBF,
                             'hyperparams': {}})
    # Check that no acquisitions were added yet
    assert (len(moop.acquisitions) == 0)
    # Now add 3 good acquisitions
    moop.addAcquisition({'acquisition': UniformWeights})
    assert (len(moop.acquisitions) == 1)
    moop.addAcquisition({'acquisition': UniformWeights},
                        {'acquisition': UniformWeights, 'hyperparams': {}})
    assert (len(moop.acquisitions) == 3)


def test_MOOP_getTypes():
    """ Check that the MOOP class handles getting dtypes properly.

    Initialize a MOOP object, add design variables, simulations, objectives,
    and constraints, and get the corresponding types.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np

    # Create a simulation for later
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}

    # Create a new MOOP
    moop = MOOP(LocalSurrogate_PS)
    # Check that all types are None
    assert (moop.getDesignType() is None)
    assert (moop.getSimulationType() is None)
    assert (moop.getObjectiveType() is None)
    assert (moop.getConstraintType() is None)
    # Add some variables, simulations, objectives, and constraints
    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'name': "x2", 'des_type': "categorical", 'levels': 3})
    moop.addSimulation(g1)
    moop.addObjective({'obj_func': lambda x, s: [sum(s)]})
    moop.addConstraint({'constraint': lambda x, s: [sum(s) - 1]})
    # Check the dtypes
    assert (np.zeros(1, dtype=moop.getDesignType()).size == 1)
    assert (np.zeros(1, dtype=moop.getSimulationType()).size == 1)
    assert (np.zeros(1, dtype=moop.getObjectiveType()).size == 1)
    assert (np.zeros(1, dtype=moop.getConstraintType()).size == 1)


def test_MOOP_evaluateSimulation():
    """ Check that the MOOP class handles evaluating simulations properly.

    Initialize a MOOP object and check that the evaluateSimulation() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np
    import pytest

    # Create 2 SimGroups for later
    g1 = {'name': "g1",
          'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'name': "g2",
          'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [sum([(xi-1.0)**2 for xi in x])],
          'surrogate': GaussRBF}
    # Initialize 2 MOOPs with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1)
    moop2 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop2.addDesign({'name': "x" + str(i+1), 'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g2)
    x1 = np.zeros(3)
    x2 = np.zeros(1, dtype=moop2.des_schema)[0]
    y1 = np.ones(3)
    y2 = np.ones(1, dtype=moop2.des_schema)[0]
    # Check database with bad values
    with pytest.raises(ValueError):
        moop1.checkSimDb(x1, -1)
    with pytest.raises(ValueError):
        moop2.checkSimDb(x2, "hello world")
    # Try to update database and evaluate sims with bad values
    with pytest.raises(TypeError):
        moop1.updateSimDb(x1, np.zeros(1), 5.0)
    with pytest.raises(ValueError):
        moop1.updateSimDb(x1, np.zeros(1), -1)
    with pytest.raises(ValueError):
        moop2.updateSimDb(x2, np.zeros(1), "hello world")
    with pytest.raises(ValueError):
        moop1.checkSimDb(x1, -1)
    with pytest.raises(ValueError):
        moop2.checkSimDb(x2, "hello world")
    # Try 3 good evaluations
    moop1.evaluateSimulation(x1, "g1")
    moop1.evaluateSimulation(y1, "g1")
    moop2.evaluateSimulation(x2, "g2")
    assert (moop1.checkSimDb(x1, "g1") is not None)
    assert (moop1.checkSimDb(y1, "g1") is not None)
    assert (moop2.checkSimDb(x2, "g2") is not None)
    assert (moop2.checkSimDb(y2, "g2") is None)
    return


def test_MOOP_evaluate_surrogates():
    """ Check that the MOOP class handles evaluating surrogate models properly.

    Initialize a MOOP object and check that the _evaluate_surrogates() function
    works correctly.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS
    import pytest

    # Create 2 SimGroups for later
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': lambda x, s: x[0]},
                       {'obj_func': lambda x, s: s[0]},
                       {'obj_func': lambda x, s: s[1] + s[2]})
    # Try some bad evaluations
    with pytest.raises(TypeError):
        moop1.evaluateSimulation(np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        moop1.evaluateSimulation(np.zeros(3), -1)
    # Evaluate some data points and fit the surrogates
    moop1.evaluateSimulation(np.zeros(3), "sim1")
    moop1.evaluateSimulation(np.zeros(3), "sim2")
    moop1.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim1")
    moop1.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim2")
    moop1.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim1")
    moop1.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim2")
    moop1.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim1")
    moop1.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim2")
    moop1.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim1")
    moop1.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim2")
    moop1.evaluateSimulation(np.ones(3), "sim1")
    moop1.evaluateSimulation(np.ones(3), "sim2")
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.ones(3) * 0.5, np.ones(3) * 0.5)
    # Now do some good evaluations and check the results
    assert (np.linalg.norm(moop1._evaluate_surrogates(np.zeros(3)) -
                           np.array([0.0, np.sqrt(3.0), np.sqrt(0.75)]))
            < 0.00000001)
    assert (np.linalg.norm(moop1._evaluate_surrogates(np.array([0.5,
                                                                0.5, 0.5]))
                           - np.array([np.sqrt(0.75), np.sqrt(0.75), 0.0]))
            < 0.00000001)
    assert (np.linalg.norm(moop1._evaluate_surrogates(np.array([1.0, 0.0,
                                                                0.0]))
                           - np.array([1.0, np.sqrt(2.0), np.sqrt(0.75)]))
            < 0.00000001)
    assert (np.linalg.norm(moop1._evaluate_surrogates(np.array([0.0, 1.0,
                                                                0.0]))
                           - np.array([1.0, np.sqrt(2.0), np.sqrt(0.75)]))
            < 0.00000001)
    assert (np.linalg.norm(moop1._evaluate_surrogates(np.array([0.0, 0.0,
                                                                1.0]))
                           - np.array([1.0, np.sqrt(2.0), np.sqrt(0.75)]))
            < 0.00000001)
    assert (np.linalg.norm(moop1._evaluate_surrogates(np.ones(3)) -
                           np.array([np.sqrt(3.0), 0.0, np.sqrt(0.75)]))
            < 0.00000001)
    assert (np.linalg.norm(moop1._surrogate_uncertainty(np.zeros(3)))
            < 1.0e-4)
    assert (np.linalg.norm(moop1._surrogate_uncertainty(np.array([0.5,
                                                                  0.5, 0.5])))
            < 1.0e-4)
    assert (np.linalg.norm(moop1._surrogate_uncertainty(np.array([1.0,
                                                                  0.0, 0.0])))
            < 1.0e-4)
    assert (np.linalg.norm(moop1._surrogate_uncertainty(np.array([0.0,
                                                                  1.0, 0.0])))
            < 1.0e-4)
    assert (np.linalg.norm(moop1._surrogate_uncertainty(np.array([0.0,
                                                                  0.0, 1.0])))
            < 1.0e-4)
    assert (np.linalg.norm(moop1._surrogate_uncertainty(np.ones(3))) < 1.0e-4)
    xi = np.random.random_sample(3)
    assert (np.linalg.norm(moop1._surrogate_uncertainty(xi)) > 1.0e-4)
    # Adjust the scale and try again
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': lambda x, s: x[0]},
                       {'obj_func': lambda x, s: s[0]},
                       {'obj_func': lambda x, s: s[1] + s[2]})
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), "sim1")
    moop2.evaluateSimulation(np.zeros(3), "sim2")
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim1")
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim2")
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim1")
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim2")
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim1")
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim2")
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim1")
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim2")
    moop2.evaluateSimulation(np.ones(3), "sim1")
    moop2.evaluateSimulation(np.ones(3), "sim2")
    moop2._fit_surrogates()
    moop2._set_surrogate_tr(np.zeros(3), np.infty)
    # Now compare evaluations against the original surrogate
    x = moop1._embed(np.zeros(3))
    xx = moop2._embed(np.zeros(3))
    assert (np.linalg.norm(moop1._evaluate_surrogates(x) -
                           moop2._evaluate_surrogates(xx)) < 0.00000001)
    x = moop1._embed(np.ones(3))
    xx = moop2._embed(np.ones(3))
    assert (np.linalg.norm(moop1._evaluate_surrogates(x) -
                           moop2._evaluate_surrogates(xx)) < 0.00000001)


def test_MOOP_evaluate_objectives():
    """ Check that the MOOP class handles evaluating objectives properly.

    Initialize a MOOP object and check that the _evaluate_objectives() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np
    import pytest

    # Create 2 SimGroups for later
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': lambda x, s: x[0]},
                       {'obj_func': lambda x, s: s["sim1"][0]},
                       {'obj_func': lambda x, s: s["sim2"][0] + s["sim2"][1]})
    # Try some bad evaluations
    with pytest.raises(ValueError):
        moop1.evaluateSimulation(np.zeros(3), -1)
    # Evaluate some data points and fit the surrogates
    moop1.evaluateSimulation(np.zeros(3), "sim1")
    moop1.evaluateSimulation(np.zeros(3), "sim2")
    moop1.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim1")
    moop1.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim2")
    moop1.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim1")
    moop1.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim2")
    moop1.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim1")
    moop1.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim2")
    moop1.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim1")
    moop1.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim2")
    moop1.evaluateSimulation(np.ones(3), "sim1")
    moop1.evaluateSimulation(np.ones(3), "sim2")
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.ones(3) * 0.5, np.ones(3) * 0.5)
    # Now do some good evaluations and check the results
    x = np.zeros(3)
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([0.0, 0.0, np.sqrt(3) + np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_objectives(x, sx) - fx) < 1.0e-8)
    x = np.ones(3) * 0.5
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([0.5, np.sqrt(0.75), np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_objectives(x, sx) - fx) < 1.0e-8)
    x = np.eye(3)[0]
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([1.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_objectives(x, sx) - fx) < 1.0e-8)
    x = np.eye(3)[1]
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_objectives(x, sx) - fx) < 1.0e-8)
    x = np.eye(3)[2]
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_objectives(x, sx) - fx) < 1.0e-8)
    x = np.ones(3)
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([1.0, np.sqrt(3), np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_objectives(x, sx) - fx) < 1.0e-8)
    # Adjust the scale and try again
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': lambda x, s: x[0]},
                       {'obj_func': lambda x, s: s["sim1"][0]},
                       {'obj_func': lambda x, s: s["sim2"][0] + s["sim2"][1]})
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), "sim1")
    moop2.evaluateSimulation(np.zeros(3), "sim2")
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim1")
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim2")
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim1")
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim2")
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim1")
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim2")
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim1")
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim2")
    moop2.evaluateSimulation(np.ones(3), "sim1")
    moop2.evaluateSimulation(np.ones(3), "sim2")
    moop2._fit_surrogates()
    moop2._set_surrogate_tr(np.zeros(3), np.infty)
    # Now compare evaluations against the original surrogate
    x = moop1._embed(np.zeros(3))
    sx = moop1._evaluate_surrogates(x)
    xx = moop2._embed(np.zeros(3))
    ssx = moop2._evaluate_surrogates(xx)
    assert (np.linalg.norm(moop1._evaluate_objectives(x, sx) -
                           moop2._evaluate_objectives(xx, ssx)) < 1.0e-8)
    x = moop1._embed(np.ones(3))
    sx = moop1._evaluate_surrogates(x)
    xx = moop2._embed(np.ones(3))
    ssx = moop2._evaluate_surrogates(xx)
    assert (np.linalg.norm(moop1._evaluate_objectives(x, sx) -
                           moop2._evaluate_objectives(xx, ssx)) < 1.0e-8)


def test_MOOP_evaluate_constraints():
    """ Check that the MOOP class handles evaluating constraints properly.

    Initialize a MOOP object and check that the _evaluate_constraints() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np
    import pytest

    # Create 2 SimGroups for later
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    # Evaluate an empty constraint and check that a zero array is returned
    assert (all(moop1._evaluate_constraints(0, 0) == np.zeros(1)))
    # Now add 3 constraints
    moop1.addConstraint({'constraint': lambda x, s: x[0]})
    moop1.addConstraint({'constraint': lambda x, s: s["sim1"][0]})
    moop1.addConstraint({'constraint': lambda x, s: s["sim2"][0] + s["sim2"][1]})
    # Evaluate some data points and fit the surrogates
    moop1.evaluateSimulation(np.zeros(3), "sim1")
    moop1.evaluateSimulation(np.zeros(3), "sim2")
    moop1.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim1")
    moop1.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim2")
    moop1.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim1")
    moop1.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim2")
    moop1.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim1")
    moop1.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim2")
    moop1.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim1")
    moop1.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim2")
    moop1.evaluateSimulation(np.ones(3), "sim1")
    moop1.evaluateSimulation(np.ones(3), "sim2")
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.infty)
    # Now do some good evaluations and check the results
    x = np.zeros(3)
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([0.0, 0.0, np.sqrt(3) + np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_constraints(x, sx) - fx) < 1.0e-8)
    x = np.ones(3) * 0.5
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([0.5, np.sqrt(0.75), np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_constraints(x, sx) - fx) < 1.0e-8)
    x = np.eye(3)[0]
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([1.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_constraints(x, sx) - fx) < 1.0e-8)
    x = np.eye(3)[1]
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_constraints(x, sx) - fx) < 1.0e-8)
    x = np.eye(3)[2]
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_constraints(x, sx) - fx) < 1.0e-8)
    x = np.ones(3)
    sx = moop1._evaluate_surrogates(x)
    fx = np.array([1.0, np.sqrt(3), np.sqrt(0.75)])
    assert (np.linalg.norm(moop1._evaluate_constraints(x, sx) - fx) < 1.0e-8)
    # Adjust the scale and try again
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.addConstraint({'constraint': lambda x, s: x[0]})
    moop2.addConstraint({'constraint': lambda x, s: s["sim1"][0]})
    moop2.addConstraint({'constraint': lambda x, s: s["sim2"][0] + s["sim2"][1]})
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), "sim1")
    moop2.evaluateSimulation(np.zeros(3), "sim2")
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim1")
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), "sim2")
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim1")
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), "sim2")
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim1")
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), "sim2")
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim1")
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), "sim2")
    moop2.evaluateSimulation(np.ones(3), "sim1")
    moop2.evaluateSimulation(np.ones(3), "sim2")
    moop2._fit_surrogates()
    moop2._set_surrogate_tr(np.zeros(3), np.infty)
    # Now compare evaluations against the original surrogate
    x = moop1._embed(np.zeros(3))
    sx = moop1._evaluate_surrogates(x)
    xx = moop2._embed(np.zeros(3))
    ssx = moop2._evaluate_surrogates(xx)
    assert (np.linalg.norm(moop1._evaluate_constraints(x, sx) -
                           moop2._evaluate_constraints(xx, ssx)) < 1.0e-8)
    x = moop1._embed(np.ones(3))
    sx = moop1._evaluate_surrogates(x)
    xx = moop2._embed(np.ones(3))
    ssx = moop2._evaluate_surrogates(xx)
    assert (np.linalg.norm(moop1._evaluate_constraints(x, sx) -
                           moop2._evaluate_constraints(xx, ssx)) < 1.0e-8)


def test_MOOP_addData():
    """ Check that the MOOP class is able to add data to its internal database.

    Initialize a MOOP object and check that the addData(s, sx) function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np

    # Create 2 SimGroups for later
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups and 2 objectives
    moop1 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': lambda x, s: s[1]})
    moop1.addObjective({'obj_func': lambda x, s: s[0]})
    # Test adding some data
    moop1.iterate(0)
    moop1.addData(np.zeros(3), np.zeros(3))
    moop1.addData(np.zeros(3), np.zeros(3))
    moop1.addData(np.ones(3), np.ones(3))
    assert (moop1.data['f_vals'].shape == (2, 2))
    assert (moop1.data['x_vals'].shape == (2, 3))
    assert (moop1.data['c_vals'].shape == (2, 1))
    assert (moop1.n_dat == 2)
    # Initialize a new MOOP with 2 SimGroups and 2 objectives
    moop2 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': lambda x, s: s[1]})
    moop2.addObjective({'obj_func': lambda x, s: s[0]})
    # Now add 3 constraints
    moop2.addConstraint({'constraint': lambda x, s: x[0]})
    moop2.addConstraint({'constraint': lambda x, s: s[0]})
    moop2.addConstraint({'constraint': lambda x, s: s[1] + s[2]})
    # Test adding some data
    moop2.iterate(0)
    moop2.addData(np.zeros(3), np.zeros(3))
    moop2.addData(np.zeros(3), np.zeros(3))
    moop2.addData(np.array([0.0, 0.0, 1.0]), np.zeros(3))
    moop2.addData(np.ones(3), np.ones(3))
    assert (moop2.data['f_vals'].shape == (3, 2))
    assert (moop2.data['x_vals'].shape == (3, 3))
    assert (moop2.data['c_vals'].shape == (3, 3))
    assert (moop2.n_dat == 3)
    # Initialize a new MOOP with 2 SimGroups and 2 objectives
    moop3 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})
    moop3.addDesign({'des_type': "categorical", 'levels': 3})
    moop3.addSimulation(g1, g2)
    moop3.addObjective({'obj_func': lambda x, s: s[1]})
    moop3.addObjective({'obj_func': lambda x, s: s[0]})
    # Now add 3 constraints
    moop3.addConstraint({'constraint': lambda x, s: x[0]})
    moop3.addConstraint({'constraint': lambda x, s: s[0]})
    moop3.addConstraint({'constraint': lambda x, s: s[1] + s[2]})
    # Test adding some data
    moop3.iterate(0)
    moop3.addData(np.ones(4), np.ones(3))
    assert (moop3.data['f_vals'].shape == (1, 2))
    assert (moop3.data['x_vals'].shape == (1, 5))
    assert (moop3.data['c_vals'].shape == (1, 3))
    assert (moop3.n_dat == 1)


def test_MOOP_iterate():
    """ Test the MOOP class's iterator in objectives.py.

    Initialize several MOOP objects and perform iterations to produce
    a batch of candidate solutions.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS, LocalSurrogate_BFGS
    import numpy as np
    import pytest

    # Initialize two simulation groups with 1 output each
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[str(key)] for key in [0, 1, 2]])],
          'surrogate': GaussRBF,
          'search_budget': 20}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[str(key)] - 1 for key in [0, 1, 2]])],
          'surrogate': GaussRBF,
          'search_budget': 20}
    # Create a MOOP with 3 design variables and 2 simulations
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    with pytest.raises(AttributeError):
        moop1.iterate(1)
    for i in range(3):
        moop1.addDesign({'name': str(i), 'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    with pytest.raises(AttributeError):
        moop1.iterate(1)
    # Now add the two objectives
    def f1(x, sim): return sim["sim1"]
    def f2(x, sim): return sim["sim2"]
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    # Add 3 acquisition functions
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    # Try some invalid iterations
    with pytest.raises(ValueError):
        moop1.iterate(-1)
    with pytest.raises(TypeError):
        moop1.iterate(2.0)
    # Solve the MOOP with 1 iteration
    batch = moop1.iterate(0)
    batch = moop1.filterBatch(batch)
    for (x, i) in batch:
        moop1.evaluateSimulation(x, i)
    moop1.updateAll(0, batch)
    batch = moop1.iterate(1)
    batch = moop1.filterBatch(batch)
    for (x, i) in batch:
        moop1.evaluateSimulation(x, i)
    moop1.updateAll(1, batch)
    soln = moop1.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    for si in soln:
        assert (np.abs(g1['sim_func'](si) - si['f1']) < 1.0e-8)
        assert (np.linalg.norm(g2['sim_func'](si) - si['f2']) < 1.0e-8)

    g3 = {'m': 4,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [x[i] for i in range(4)],
          'surrogate': GaussRBF,
          'search_budget': 500}
    # Create a three objective toy problem, with one simulation
    moop2 = MOOP(LocalSurrogate_BFGS, hyperparams={'opt_budget': 100})
    for i in range(4):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0, 'des_tol': 0.1})
    moop2.addSimulation(g3)

    # Now add the three objectives
    def f3(x, sim, der=0):
        if der == 1:
            x_out = x.copy()
            for key in x:
                x_out[key] = 0.0
            return x_out
        elif der == 2:
            s_out = sim.copy()
            s_out["sim1"] *= 2.0
            s_out["sim1"][0] -= 0.2
            return s_out
        else:
            return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[0, :]) ** 2.0

    def f4(x, sim, der=0):
        if der == 1:
            x_out = x.copy()
            for key in x:
                x_out[key] = 0.0
            return x_out
        elif der == 2:
            s_out = sim.copy()
            s_out["sim1"] *= 2.0
            s_out["sim1"][1] -= 0.2
            return s_out
        else:
            return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[1, :]) ** 2.0

    def f5(x, sim, der=0):
        if der == 1:
            x_out = x.copy()
            for key in x:
                x_out[key] = 0.0
            return x_out
        elif der == 2:
            s_out = sim.copy()
            s_out["sim1"] *= 2.0
            s_out["sim1"][2] -= 0.2
            return s_out
        else:
            return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[2, :]) ** 2.0

    moop2.addObjective({'obj_func': f3},
                       {'obj_func': f4},
                       {'obj_func': f5})
    # Add 3 acquisition functions
    for i in range(3):
        moop2.addAcquisition({'acquisition': UniformWeights})
    # Do 2 iterates of the MOOP and extract the final database
    moop2.iterate(0)
    batch = [(0.1 * xi, "sim1") for xi in np.eye(4)]
    batch.append((0.1 * np.ones(4), "sim1"))
    for (x, i) in batch:
        moop2.evaluateSimulation(x, i)
    moop2.updateAll(0, batch)
    batch = moop2.iterate(1)
    batch = moop2.filterBatch(batch)
    for (x, i) in batch:
        moop2.evaluateSimulation(x, i)
    moop2.updateAll(1, batch)
    soln = moop2.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(4)
    for i in range(np.shape(soln['x_vals'])[0]):
        sim = soln['x_vals'][i]
        assert (np.linalg.norm(np.array([f3(soln['x_vals'][i], sim),
                                         f4(soln['x_vals'][i], sim),
                                         f5(soln['x_vals'][i], sim)]
                                        ).flatten()
                               - soln['f_vals'][i])
                < 0.00000001)
        assert (all(soln['x_vals'][i, :4] <= 0.2))

    g4 = {'m': 4,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: x[0:4] + abs(x[4] - 1.0),
          'surrogate': GaussRBF,
          'search_budget': 500}
    # Create a three objective toy problem, with one simulation
    moop3 = MOOP(LocalSurrogate_BFGS, hyperparams={})
    for i in range(4):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})
    moop3.addDesign({'des_type': "categorical", 'levels': 3})
    moop3.addSimulation(g4)

    # Now add the three objectives
    def f6(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.array([2.0 * sim[0] - 2.0,
                             2.0 * sim[1],
                             2.0 * sim[2],
                             2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - np.eye(4)[0, :]) ** 2.0

    def f7(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.array([2.0 * sim[0],
                             2.0 * sim[1] - 2.0,
                             2.0 * sim[2],
                             2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - np.eye(4)[1, :]) ** 2.0

    def f8(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.array([2.0 * sim[0],
                             2.0 * sim[1],
                             2.0 * sim[2] - 2.0,
                             2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - np.eye(4)[2, :]) ** 2.0

    moop3.addObjective({'obj_func': f6},
                       {'obj_func': f7},
                       {'obj_func': f8})
    # Add 3 acquisition functions
    for i in range(3):
        moop3.addAcquisition({'acquisition': UniformWeights})
    # Do 2 iterates of the MOOP and extract the final database
    moop3.iterate(0)
    batch = [(xi, 0) for xi in np.eye(5)]
    batch.append((np.ones(5), 0))
    for (x, i) in batch:
        moop3.evaluateSimulation(x, i)
    moop3.updateAll(0, batch)
    batch = moop3.iterate(1)
    batch = moop3.filterBatch(batch)
    for (x, i) in batch:
        moop3.evaluateSimulation(x, i)
    moop3.updateAll(1, batch)
    soln = moop3.getPF()
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(4)
    for i in range(np.shape(soln['x_vals'])[0]):
        sim = soln['x_vals'][i, :4] - abs(soln['x_vals'][i, 4] - 1.0)
        assert (np.linalg.norm(np.array([f6(soln['x_vals'][i], sim),
                                         f7(soln['x_vals'][i], sim),
                                         f8(soln['x_vals'][i], sim)]
                                        ).flatten()
                               - soln['f_vals'][i])
                < 0.00000001)
        assert (soln['x_vals'][i, 3] <= 0.1 and soln['x_vals'][i, 4] == 1.0)

    x_entry = np.zeros(1, dtype=np.dtype([("x0", float), ("x1", float),
                                          ("x2", object)]))
    x_entry[0]["x2"] = "0"
    g5 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [(x["x0"] - 1.0) * (x["x0"] - 1.0) +
                                 (x["x1"]) * (x["x1"]) + float(x["x2"])],
          'surrogate': GaussRBF,
          'search_budget': 100}
    # Solve a MOOP with categorical variables
    moop4 = MOOP(LocalSurrogate_BFGS, hyperparams={})
    moop4.addDesign({'name': "x0", 'lb': 0.0, 'ub': 1.0})
    moop4.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1.0})
    moop4.addDesign({'name': "x2", 'des_type': "categorical",
                     'levels': ["0", "1"]})
    moop4.addSimulation(g5)

    # Now add the two objectives
    def f9(x, sim, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)[0]
        elif der == 2:
            result = np.ones(1, dtype=sim.dtype)
            return result[0]
        else:
            return sim[0]

    def f10(x, sim, der=0):
        if der == 1:
            out = np.zeros(1, dtype=x.dtype)
            out['x0'] = 2.0 * x["x0"]
            out['x1'] = 2.0 * x["x1"] - 2.0
            out['x2'] = 0.0
            return out[0]
        elif der == 2:
            return np.zeros(1, dtype=sim.dtype)[0]
        else:
            return ((x["x0"]) * (x["x0"]) +
                    (x["x1"] - 1.0) * (x["x1"] - 1.0) + float(x["x2"]))

    moop4.addObjective({'obj_func': f9},
                       {'obj_func': f10})
    # Add 3 acquisition functions
    for i in range(3):
        moop4.addAcquisition({'acquisition': UniformWeights})
    # Do 2 iterates of the MOOP and extract the final database
    batch = moop4.iterate(0)
    batch = moop4.filterBatch(batch)
    for (x, i) in batch:
        moop4.evaluateSimulation(x, i)
    moop4.updateAll(0, batch)
    batch = moop4.iterate(1)
    batch = moop4.filterBatch(batch)
    for (x, i) in batch:
        moop4.evaluateSimulation(x, i)
    moop4.updateAll(1, batch)
    soln = moop4.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(1)
    for i, xi in enumerate(soln):
        sim[0] = ((xi["x0"] - 1.0) * (xi["x0"] - 1.0) +
                  (xi["x1"]) * (xi["x1"]) + float(xi["x2"]))
        assert (f9(soln[i], sim) - soln['f1'][i] < 1.0e-8 and
                f10(soln[i], sim) - soln['f2'][i] < 1.0e-8)
        assert (xi["x2"] == "0")


def test_MOOP_solve():
    """ Test the MOOP class's solver in objectives.py.

    Perform a test of the MOOP solver class by minimizing a 5 variable,
    biobjective convex function s.t. $x in [0, 1]^n$.

    The correctness of the solutions is difficult to assert , but we can
    assert  that the efficient points map onto the Pareto front, as
    expected.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights, RandomConstraint
    from parmoo.optimizers import LocalSurrogate_PS, GlobalSurrogate_BFGS
    import numpy as np
    import pytest

    # Initialize two simulation groups with 1 output each
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 25,
          'sim_func': lambda x: [np.linalg.norm(x-1.0)],
          'surrogate': GaussRBF}
    # Create a MOOP with 4 design variables and 2 simulations
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    for i in range(4):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    # Now add 2 objectives
    def f1(x, sim): return sim[0]
    def f2(x, sim): return sim[1]
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    # Add 3 acquisition functions
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    # Try to solve several invalid problems/budgets to test error handling
    with pytest.raises(ValueError):
        moop1.solve(-1)
    with pytest.raises(TypeError):
        moop1.solve(2.0)
    # Solve the MOOP with 6 iterations
    moop1.solve(6)
    soln = moop1.data
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    for i in range(np.shape(soln['x_vals'])[0]):
        assert (np.linalg.norm(np.array([g1['sim_func'](soln['x_vals'][i]),
                                         g2['sim_func'](soln['x_vals'][i])]
                                        ).flatten() - soln['f_vals'][i])
               < 0.00000001)
    # Create new single objective toy problem
    g3 = {'m': 1,
          'sim_func': lambda x: [x[0] + x[1]],
          'surrogate': GaussRBF,
          'search': LatinHypercube,
          'hyperparams': {'search_budget': 10}}
    g4 = {'m': 1,
          'sim_func': lambda x: [x[2] + x[3]],
          'surrogate': GaussRBF,
          'search': LatinHypercube,
          'hyperparams': {'search_budget': 20}}
    moop2 = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g3, g4)
    # Now add 1 objective
    def f3(x, sim): return sim[0] + sim[1]
    moop2.addObjective({'obj_func': f3})
    # Add 3 acquisition functions
    for i in range(3):
        moop2.addAcquisition({'acquisition': RandomConstraint})
    # Solve the MOOP and extract the final database with 6 iterations
    moop2.solve(6)
    soln = moop2.data
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    for i in range(np.shape(soln['x_vals'])[0]):
        assert (np.linalg.norm(np.array(g3['sim_func'](soln['x_vals'][i])) +
                               np.array(g4['sim_func'](soln['x_vals'][i])) -
                               soln['f_vals'][i]) < 0.00000001)

    # Create a 3 objective toy problem, with no simulations
    moop3 = MOOP(GlobalSurrogate_BFGS, hyperparams={})
    for i in range(4):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})

    # Now add the three objectives
    def f4(x, sim, der=0):
        if der == 1:
            return np.array([2.0 * x[0] - 2.0,
                             2.0 * x[1],
                             2.0 * x[2],
                             2.0 * x[3]])
        elif der == 2:
            return np.zeros(sim.size)
        else:
            return np.linalg.norm(x - np.eye(x.size)[0, :]) ** 2.0

    def f5(x, sim, der=0):
        if der == 1:
            return np.array([2.0 * x[0],
                             2.0 * x[1] - 2.0,
                             2.0 * x[2],
                             2.0 * x[3]])
        elif der == 2:
            return np.zeros(sim.size)
        else:
            return np.linalg.norm(x - np.eye(x.size)[1, :]) ** 2.0

    def f6(x, sim, der=0):
        if der == 1:
            return np.array([2.0 * x[0],
                             2.0 * x[1],
                             2.0 * x[2] - 2.0,
                             2.0 * x[3],
                             0.0])
        elif der == 2:
            return np.zeros(sim.size)
        else:
            return np.linalg.norm(x - np.eye(x.size)[2, :]) ** 2.0

    moop3.addObjective({'obj_func': f4},
                       {'obj_func': f5},
                       {'obj_func': f6})
    # Add 3 acquisition functions
    for i in range(3):
        moop3.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop3.solve(6)
    soln = moop3.data
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(0)
    for i in range(np.shape(soln['x_vals'])[0]):
        assert (np.linalg.norm(np.array([f4(soln['x_vals'][i], sim),
                                         f5(soln['x_vals'][i], sim),
                                         f6(soln['x_vals'][i], sim)]
                                        ).flatten()
                               - soln['f_vals'][i])
                < 0.00000001)

    # Create a 3 objective toy problem, with no simulations and 1 categorical
    moop4 = MOOP(GlobalSurrogate_BFGS, hyperparams={})
    for i in range(3):
        moop4.addDesign({'lb': 0.0, 'ub': 1.0})
    moop4.addDesign({'des_type': "categorical", 'levels': 3})
    moop4.addObjective({'obj_func': f4},
                       {'obj_func': f5},
                       {'obj_func': f6})
    # Add 3 acquisition functions
    for i in range(3):
        moop4.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop4.solve(6)
    soln = moop4.getPF()
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(0)
    for i in range(np.shape(soln['x_vals'])[0]):
        assert (np.linalg.norm(np.array([f4(soln['x_vals'][i], sim),
                                        f5(soln['x_vals'][i], sim),
                                         f6(soln['x_vals'][i], sim)]
                                        ).flatten()
                               - soln['f_vals'][i])
                < 0.00000001)


def test_MOOP_getPF():
    """ Test the getPF function.

    Create several MOOPs, evaluate simulations, and check the final Pareto
    front for correctness.

    """

    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np

    # Create a toy problem with 4 design variables
    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    # Now add three objectives
    def f1(x, sim): return np.linalg.norm(x - np.eye(4)[0, :]) ** 2.0
    moop.addObjective({'obj_func': f1})
    def f2(x, sim): return np.linalg.norm(x - np.eye(4)[1, :]) ** 2.0
    moop.addObjective({'obj_func': f2})
    def f3(x, sim): return np.linalg.norm(x - np.eye(4)[2, :]) ** 2.0
    moop.addObjective({'obj_func': f3})
    moop.addConstraint({'constraint': lambda x, s: -sum(x)})
    # Add 3 acquisition functions
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop.data = {'x_vals': np.zeros((5, 4)),
                 'f_vals': np.zeros((5, 3)),
                 'c_vals': np.zeros((5, 1))}
    moop.data['x_vals'][0, :] = np.array([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][1, :] = np.array([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop._evaluate_objectives(
                                   np.array([1.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][2, :] = np.array([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop._evaluate_objectives(
                                   np.array([0.0, 1.0, 0.0, 0.0]))
    moop.data['x_vals'][3, :] = np.array([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 1.0, 0.0]))
    moop.data['x_vals'][4, :] = np.array([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 1.0]))
    moop.n_dat = 5
    soln = moop.getPF()
    assert (soln['f_vals'].shape == (4, 3))
    # Create a toy problem with 4 design variables
    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'name': ('x' + str(i+1)), 'lb': 0.0, 'ub': 1.0})

    # Now add three objectives
    def f1(x, sim):
        return (x['x1'] - 1.0)**2 + (x['x2'])**2 + (x['x3'])**2 + (x['x4'])**2
    moop.addObjective({'obj_func': f1})

    def f2(x, sim):
        return (x['x1'])**2 + (x['x2'] - 1.0)**2 + (x['x3'])**2 + (x['x4'])**2
    moop.addObjective({'obj_func': f2})

    def f3(x, sim):
        return (x['x1'])**2 + (x['x2'])**2 + (x['x3'] - 1.0)**2 + (x['x4'])**2
    moop.addObjective({'obj_func': f3})
    moop.addConstraint({'constraint': lambda x, s: -sum(x)})
    # Add 3 acquisition functions
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop.data = {'x_vals': np.zeros((5, 4)),
                 'f_vals': np.zeros((5, 3)),
                 'c_vals': np.zeros((5, 1))}
    moop.data['x_vals'][0, :] = np.array([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][0, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][1, :] = np.array([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop._evaluate_objectives(
                                   np.array([1.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][1, :] = moop._evaluate_constraints(
                                   np.array([1.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][2, :] = np.array([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop._evaluate_objectives(
                                   np.array([0.0, 1.0, 0.0, 0.0]))
    moop.data['c_vals'][2, :] = moop._evaluate_constraints(
                                   np.array([0.0, 1.0, 0.0, 0.0]))
    moop.data['x_vals'][3, :] = np.array([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 1.0, 0.0]))
    moop.data['c_vals'][3, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 1.0, 0.0]))
    moop.data['x_vals'][4, :] = np.array([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 1.0]))
    moop.data['c_vals'][4, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 0.0, 1.0]))
    moop.n_dat = 5
    soln = moop.getPF()
    assert (soln.shape[0] == 4)


def test_MOOP_getSimulationData():
    """ Test the getSimulationData function.

    Create several MOOPs, evaluate simulations, and check the simulation
    database.

    """
    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np

    # Create 4 SimGroups for later
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'m': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    g3 = {'name': "Bobo1",
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [sum([x[name] ** 2.0
                                      for name in x.dtype.names])],
          'surrogate': GaussRBF}
    g4 = {'name': "Bobo2",
          'm': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [sum([(x[name] - 1.0) ** 2.0
                                      for name in x.dtype.names]),
                                 sum([(x[name] - 0.5) ** 2.0
                                      for name in x.dtype.names])],
          'surrogate': GaussRBF}
    # Create a toy problem with 4 design variables
    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(g1, g2)
    soln = moop.getSimulationData()
    assert (soln[0]['s_vals'].size == 0)
    assert (soln[1]['s_vals'].size == 0)
    # Evaluate 5 simulations
    moop.evaluateSimulation(np.array([0.0, 0.0, 0.0, 0.0]), "Bobo1")
    moop.evaluateSimulation(np.array([0.0, 0.0, 0.0, 0.0]), "Bobo2")
    moop.evaluateSimulation(np.array([1.0, 0.0, 0.0, 0.0]), "Bobo1")
    moop.evaluateSimulation(np.array([1.0, 0.0, 0.0, 0.0]), "Bobo2")
    moop.evaluateSimulation(np.array([0.0, 1.0, 0.0, 0.0]), "Bobo1")
    moop.evaluateSimulation(np.array([0.0, 1.0, 0.0, 0.0]), "Bobo2")
    moop.evaluateSimulation(np.array([0.0, 0.0, 1.0, 0.0]), "Bobo1")
    moop.evaluateSimulation(np.array([0.0, 0.0, 1.0, 0.0]), "Bobo2")
    moop.evaluateSimulation(np.array([0.0, 0.0, 0.0, 1.0]), "Bobo1")
    moop.evaluateSimulation(np.array([0.0, 0.0, 0.0, 1.0]), "Bobo2")
    soln = moop.getSimulationData()
    assert (soln[0]['s_vals'].shape == (5, 1))
    assert (soln[1]['s_vals'].shape == (5, 2))
    # Create a toy problem with 4 design variables
    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'name': ("x" + str(i + 1)), 'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(g3, g4)
    soln = moop.getSimulationData()
    assert (soln['Bobo1']['out'].size == 0)
    assert (soln['Bobo2']['out'].size == 0)
    # Evaluate 5 simulations
    sample_x = np.zeros(1, dtype=moop.des_schema)
    moop.evaluateSimulation(sample_x[0], "Bobo1")
    moop.evaluateSimulation(sample_x[0], "Bobo2")
    sample_x["x1"] = 1.0
    moop.evaluateSimulation(sample_x[0], "Bobo1")
    moop.evaluateSimulation(sample_x[0], "Bobo2")
    sample_x["x1"] = 0.0
    sample_x["x2"] = 1.0
    moop.evaluateSimulation(sample_x[0], "Bobo1")
    moop.evaluateSimulation(sample_x[0], "Bobo2")
    sample_x["x2"] = 0.0
    sample_x["x3"] = 1.0
    moop.evaluateSimulation(sample_x[0], "Bobo1")
    moop.evaluateSimulation(sample_x[0], "Bobo2")
    sample_x["x3"] = 0.0
    sample_x["x4"] = 1.0
    moop.evaluateSimulation(sample_x[0], "Bobo1")
    moop.evaluateSimulation(sample_x[0], "Bobo2")
    soln = moop.getSimulationData()
    assert (soln['Bobo1']['out'].shape == (5,))
    assert (soln['Bobo2']['out'].shape == (5, 2))


def test_MOOP_getObjectiveData():
    """ Test the getObjectiveData function.

    Create several MOOPs, evaluate simulations, and check the objective
    database.

    """

    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np

    # Create a toy problem with 4 design variables
    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    # Now add three objectives
    def f1(x, sim): return np.linalg.norm(x - np.eye(4)[0, :]) ** 2.0
    moop.addObjective({'obj_func': f1})
    def f2(x, sim): return np.linalg.norm(x - np.eye(4)[1, :]) ** 2.0
    moop.addObjective({'obj_func': f2})
    def f3(x, sim): return np.linalg.norm(x - np.eye(4)[2, :]) ** 2.0
    moop.addObjective({'obj_func': f3})
    moop.addConstraint({'constraint': lambda x, s: -sum(x)})
    # Add 3 acquisition functions
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop.data = {'x_vals': np.zeros((5, 4)),
                 'f_vals': np.zeros((5, 3)),
                 'c_vals': np.zeros((5, 1))}
    moop.data['x_vals'][0, :] = np.array([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][0, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][1, :] = np.array([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop._evaluate_objectives(
                                   np.array([1.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][1, :] = moop._evaluate_constraints(
                                   np.array([1.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][2, :] = np.array([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop._evaluate_objectives(
                                   np.array([0.0, 1.0, 0.0, 0.0]))
    moop.data['c_vals'][2, :] = moop._evaluate_constraints(
                                   np.array([0.0, 1.0, 0.0, 0.0]))
    moop.data['x_vals'][3, :] = np.array([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 1.0, 0.0]))
    moop.data['c_vals'][3, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 1.0, 0.0]))
    moop.data['x_vals'][4, :] = np.array([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 1.0]))
    moop.data['c_vals'][4, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 0.0, 1.0]))
    moop.n_dat = 5
    soln = moop.getObjectiveData()
    assert (soln['f_vals'].shape == (5, 3))
    # Create a toy problem with 4 design variables
    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'name': ('x' + str(i+1)), 'lb': 0.0, 'ub': 1.0})

    # Now add three objectives
    def f1(x, sim):
        return (x['x1'] - 1.0)**2 + (x['x2'])**2 + (x['x3'])**2 + (x['x4'])**2
    moop.addObjective({'obj_func': f1})

    def f2(x, sim):
        return (x['x1'])**2 + (x['x2'] - 1.0)**2 + (x['x3'])**2 + (x['x4'])**2
    moop.addObjective({'obj_func': f2})

    def f3(x, sim):
        return (x['x1'])**2 + (x['x2'])**2 + (x['x3'] - 1.0)**2 + (x['x4'])**2
    moop.addObjective({'obj_func': f3})
    moop.addConstraint({'constraint': lambda x, s: -sum(x)})
    # Add 3 acquisition functions
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop.data = {'x_vals': np.zeros((5, 4)),
                 'f_vals': np.zeros((5, 3)),
                 'c_vals': np.zeros((5, 1))}
    moop.data['x_vals'][0, :] = np.array([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][0, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][1, :] = np.array([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop._evaluate_objectives(
                                   np.array([1.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][1, :] = moop._evaluate_constraints(
                                   np.array([1.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][2, :] = np.array([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop._evaluate_objectives(
                                   np.array([0.0, 1.0, 0.0, 0.0]))
    moop.data['c_vals'][2, :] = moop._evaluate_constraints(
                                   np.array([0.0, 1.0, 0.0, 0.0]))
    moop.data['x_vals'][3, :] = np.array([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 1.0, 0.0]))
    moop.data['c_vals'][3, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 1.0, 0.0]))
    moop.data['x_vals'][4, :] = np.array([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 1.0]))
    moop.data['c_vals'][4, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 0.0, 1.0]))
    moop.n_dat = 5
    soln = moop.getObjectiveData()
    assert (soln.shape[0] == 5)


def test_MOOP_save_load1():
    """ Check that a MOOP object can be correctly saved/reloaded.

    Create and save a MOOP object, then reload and check that it is the same.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np
    import pytest
    import os

    # Functions sim1, sim2, f1, f2, c1 need to be global for save/load to work
    global sim1, sim2, f1, f2, c1

    # Initialize two simulation groups with 1 output each
    def sim1(x): return [np.linalg.norm(x)]
    def sim2(x): return [np.linalg.norm(x - 1.0)]
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 25,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    # Create two objectives for later
    def f1(x, sim): return sim[0]
    def f2(x, sim): return sim[1]
    # Create a simulation for later
    def c1(x, sim): return x[0] - 0.5
    # Create a MOOP with 3 design variables and 2 simulations
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    # Empty save
    moop1.save()
    # Add design variables
    for i in range(2):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addDesign({'des_type': "categorical", 'levels': 3})
    moop1.addSimulation(g1, g2)
    # Add 2 objectives
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    # Add 1 constraint
    moop1.addConstraint({'constraint': c1})
    # Add 3 acquisition functions
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    batch = moop1.iterate(0)
    batch = moop1.filterBatch(batch)
    for (xi, i) in batch:
        moop1.evaluateSimulation(xi, i)
    moop1.updateAll(0, batch)
    # Test save
    moop1.save()
    # Test load
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.load()
    # Check that save/load are correct
    check_moops(moop1, moop2)
    # Create a new MOOP with same specs
    moop3 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    for i in range(2):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})
    moop3.addDesign({'des_type': "categorical", 'levels': 3})
    moop3.addSimulation(g1, g2)
    moop3.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    moop3.addConstraint({'constraint': c1})
    for i in range(3):
        moop3.addAcquisition({'acquisition': UniformWeights})
    # Try to save and overwrite old data
    with pytest.raises(OSError):
        moop3.save()
    # Save a data point with moop1
    moop1.savedata(np.zeros(1, dtype=moop3.getDesignType())[0],
                   np.zeros(1), "sim1")
    # Try to overwrite with moop3
    with pytest.raises(OSError):
        moop3.savedata(np.zeros(1, dtype=moop3.getDesignType())[0],
                       np.zeros(1), "sim1")
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.simdb.json")
    os.remove("parmoo.surrogate.1")
    os.remove("parmoo.surrogate.2")
    os.remove("parmoo.search.1")
    os.remove("parmoo.search.2")
    os.remove("parmoo.optimizer")


def test_MOOP_save_load2():
    """ Check that a MOOP object can be correctly saved/reloaded.

    Create and save a MOOP object, then reload and check that it is the same.

    Use simulation/objective callable objects from the library.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.simulations.dtlz import dtlz2_sim
    from parmoo.objectives import single_sim_out
    from parmoo.constraints import single_sim_bound
    import os

    # Initialize the simulation group with 3 outputs
    sim1 = dtlz2_sim(3, num_obj=2)
    g1 = {'m': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    f1 = single_sim_out(3, 2, 0)
    f2 = single_sim_out(3, 2, 1)
    c1 = single_sim_bound(3, 2, 1)
    # Create a MOOP with 3 design variables and 2 simulations
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    # Test empty save
    moop1.save()
    # Add design variables
    for i in range(2):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addDesign({'des_type': "categorical", 'levels': 3})
    moop1.addSimulation(g1)
    # Add 2 objectives
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    # Add 1 constraint
    moop1.addConstraint({'constraint': c1})
    # Add 3 acquisition functions
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    # Test save
    moop1.save()
    # Test load
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.load()
    # Check that save/load are correct
    check_moops(moop1, moop2)
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.surrogate.1")
    os.remove("parmoo.search.1")
    os.remove("parmoo.optimizer")


def test_MOOP_checkpoint():
    """ Check that the MOOP object performs checkpointing correctly.

    Run 1 iteration of ParMOO, with checkpointing on.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np
    import os

    # Functions sim1, sim2, f1, f2, c1 need to be global for save/load to work
    global sim1, sim2, f1, f2, c1

    # Initialize two simulation groups with 1 output each
    def sim1(x): return [np.linalg.norm(x)]
    def sim2(x): return [np.linalg.norm(x - 1.0)]
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 25,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    # Create two objectives for later
    def f1(x, sim): return sim[0]
    def f2(x, sim): return sim[1]
    # Create a simulation for later
    def c1(x, sim): return x[0] - 0.5
    # Create a MOOP with 3 design variables and 2 simulations
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    # Add design variables
    for i in range(2):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addDesign({'des_type': "categorical", 'levels': 3})
    moop1.addSimulation(g1, g2)
    # Add 2 objectives
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    # Add 1 constraint
    moop1.addConstraint({'constraint': c1})
    # Add 3 acquisition functions
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    # Turn on checkpointing
    moop1.setCheckpoint(True)
    # One iteration
    batch = moop1.iterate(0)
    batch = moop1.filterBatch(batch)
    for (xi, i) in batch:
        moop1.evaluateSimulation(xi, i)
    moop1.updateAll(0, batch)
    # Test load
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.load()
    # Check that save/load are correct
    check_moops(moop1, moop2)
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.simdb.json")
    os.remove("parmoo.surrogate.1")
    os.remove("parmoo.surrogate.2")
    os.remove("parmoo.search.1")
    os.remove("parmoo.search.2")
    os.remove("parmoo.optimizer")


def check_moops(moop1, moop2):
    """ Auxiliary function for checking that 2 moops are equal.

    Check that all entries in moop1 = moop2

    Args:
        moop1 (MOOP): First moop to compare

        moop2 (MOOP): Second moop to compare

    """

    import numpy as np

    # Check scalars
    assert (moop2.n == moop1.n and moop2.m_total == moop1.m_total and
            moop2.o == moop1.o and moop2.p == moop1.p and
            moop2.s == moop1.s and moop2.n_dat == moop1.n_dat and
            moop2.n_cat_d == moop1.n_cat_d and moop2.n_cat == moop1.n_cat and
            moop2.n_cont == moop1.n_cont and moop2.lam == moop1.lam and
            moop2.iteration == moop1.iteration)
    # Check lists
    assert (all([dt2i == dt1i for dt2i, dt1i in zip(moop2.latent_des_tols,
                                                    moop1.latent_des_tols)]))
    assert (all([m2i == m1i for m2i, m1i in zip(moop2.m, moop1.m)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.sim_schema,
                                                      moop1.sim_schema)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.des_schema,
                                                      moop1.des_schema)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.obj_schema,
                                                      moop1.obj_schema)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.con_schema,
                                                      moop1.con_schema)]))
    # Check dictionaries
    assert (all([ki in moop2.hyperparams.keys()
                 for ki in moop1.hyperparams.keys()]))
    assert (all([ki in moop2.history.keys() for ki in moop1.history.keys()]))
    # Check np.ndarrays
    assert (np.all(np.array(moop2.latent_lb) == np.array(moop1.scaled_lb)))
    assert (np.all(np.array(moop2.latent_ub) == np.array(moop1.scaled_ub)))
    assert (all([moop2.data[ki].shape == moop1.data[ki].shape
                 for ki in moop2.data.keys()]))
    assert (all([all([moop2.sim_db[j][ki].shape == moop1.sim_db[j][ki].shape
                      for ki in ["x_vals", "s_vals"]])
                 for j in range(len(moop1.sim_db))]))
    for obj1, obj2 in zip(moop1.obj_funcs, moop2.obj_funcs):
        if hasattr(obj1, "__name__"):
            assert (obj1.__name__ == obj2.__name__)
        else:
            assert (obj1.__class__.__name__ == obj2.__class__.__name__)
    for sim1, sim2 in zip(moop1.sim_funcs, moop2.sim_funcs):
        if hasattr(sim1, "__name__"):
            assert (sim1.__name__ == sim2.__name__)
        else:
            assert (sim1.__class__.__name__ == sim2.__class__.__name__)
    for const1, const2 in zip(moop1.constraints, moop2.constraints):
        if hasattr(const1, "__name__"):
            assert (const1.__name__ == const2.__name__)
        else:
            assert (const1.__class__.__name__ == const2.__class__.__name__)
    # Check functions
    assert (moop2.optimizer.__name__ == moop1.optimizer.__name__)
    assert (all([s1.__class__.__name__ == s2.__class__.__name__
                 for s1, s2 in zip(moop1.searches, moop2.searches)]))
    assert (all([s1.__class__.__name__ == s2.__class__.__name__
                 for s1, s2 in zip(moop1.surrogates, moop2.surrogates)]))
    assert (all([s1.__class__.__name__ == s2.__class__.__name__
                 for s1, s2 in zip(moop1.acquisitions, moop2.acquisitions)]))


if __name__ == "__main__":
    test_MOOP_init()
    test_MOOP_addSimulation()
    test_pack_unpack_sim()
    test_MOOP_addObjective()
    test_MOOP_addConstraint()
    test_MOOP_addAcquisition()
    test_MOOP_getTypes()
    test_MOOP_evaluateSimulation()
    test_MOOP_evaluate_surrogates()
    test_MOOP_evaluate_objectives()
    test_MOOP_evaluate_constraints()
    test_MOOP_addData()
    test_MOOP_iterate()
    test_MOOP_solve()
    test_MOOP_getPF()
    test_MOOP_getSimulationData()
    test_MOOP_getObjectiveData()
    test_MOOP_save_load1()
    test_MOOP_save_load2()
    test_MOOP_checkpoint()
