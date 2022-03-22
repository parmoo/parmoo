
def test_MOOP_init():
    """ Check that the MOOP class handles initialization properly.

    Initialize several MOOP objects, and check that their internal fields
    appear correct.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalGPS
    import pytest

    # Try providing invalid SurrogateOptimizer objects
    with pytest.raises(TypeError):
        MOOP(5.0)
    with pytest.raises(TypeError):
        MOOP(lambda w, x, y, z: 0.0)
    # Test bad hyperparams dictionary
    with pytest.raises(TypeError):
        MOOP(LocalGPS, hyperparams=[])
    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalGPS)
    assert(moop.n == 0 and moop.n_cont == 0 and moop.n_cat == 0 and
           moop.n_cat_d == 0 and moop.s == 0 and moop.m_total == 0 and
           moop.o == 0 and moop.p == 0)
    # Initialize a MOOP with a hyperparameter list
    moop = MOOP(LocalGPS, hyperparams={'test': 0})
    assert(moop.n == 0 and moop.n_cont == 0 and moop.n_cat == 0 and
           moop.n_cat_d == 0 and moop.s == 0 and moop.m_total == 0 and
           moop.o == 0 and moop.p == 0)
    assert(moop.hyperparams['test'] == 0)


def test_MOOP_addDesign():
    """ Check that the MOOP class handles adding new design variables properly.

    Initialize a MOOP objects, and add several design variables. Check that
    the __embed__/__extract__ function work properly.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalGPS
    import pytest

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalGPS)
    # Try to add some bad design variable types
    with pytest.raises(TypeError):
        moop.addDesign([])
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': 1.0})
    with pytest.raises(ValueError):
        moop.addDesign({'des_type': "hello world"})
    # Add some bad continuous variables
    with pytest.raises(AttributeError):
        moop.addDesign({'des_type': "continuous"})
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': "continuous",
                        'des_tol': "hello world",
                        'lb': 0.0,
                        'ub': 1.0})
    with pytest.raises(ValueError):
        moop.addDesign({'des_type': "continuous",
                        'des_tol': -1.0,
                        'lb': 0.0,
                        'ub': 1.0})
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': "continuous",
                        'lb': "hello",
                        'ub': "world"})
    with pytest.raises(ValueError):
        moop.addDesign({'des_type': "continuous",
                        'lb': 0.0,
                        'ub': 0.0})
    with pytest.raises(TypeError):
        moop.addDesign({'name': 5,
                        'des_type': "continuous",
                        'lb': 0.0,
                        'ub': 1.0})
    # Add some bad categorical variables
    with pytest.raises(AttributeError):
        moop.addDesign({'des_type': "categorical"})
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': "categorical",
                        'levels': 1.0})
    with pytest.raises(ValueError):
        moop.addDesign({'des_type': "categorical",
                        'levels': 1})
    with pytest.raises(TypeError):
        moop.addDesign({'name': 5,
                        'des_type': "categorical",
                        'levels': 2})
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': "categorical",
                        'levels': [3, "hi"]})
    with pytest.raises(ValueError):
        moop.addDesign({'des_type': "categorical",
                        'levels': ["hi"]})
    # Add some bad continuous variables, using default option
    with pytest.raises(AttributeError):
        moop.addDesign({})
    with pytest.raises(TypeError):
        moop.addDesign({'des_tol': "hello world",
                        'lb': 0.0,
                        'ub': 1.0})
    with pytest.raises(ValueError):
        moop.addDesign({'des_tol': -1.0,
                        'lb': 0.0,
                        'ub': 1.0})
    with pytest.raises(TypeError):
        moop.addDesign({'lb': "hello",
                        'ub': "world"})
    with pytest.raises(ValueError):
        moop.addDesign({'lb': 0.0,
                        'ub': 0.0})
    with pytest.raises(TypeError):
        moop.addDesign({'name': 5,
                        'lb': 0.0,
                        'ub': 1.0})
    # Add variables out of order
    with pytest.raises(RuntimeError):
        moop1 = MOOP(LocalGPS)
        moop1.acquisitions.append(0)
        moop1.addDesign({'des_type': "continuous",
                         'lb': 0.0,
                         'ub': 1.0})
    with pytest.raises(RuntimeError):
        moop2 = MOOP(LocalGPS)
        moop2.sim_funcs.append(0)
        moop2.addDesign({'des_type': "continuous",
                         'lb': 0.0,
                         'ub': 1.0})
    # Now add some good continuous design variables
    assert(moop.n == 0)
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    assert(moop.n == 1)
    moop.addDesign({'des_type': "continuous",
                    'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert(moop.n == 2)
    moop.addDesign({'name': "x_3",
                    'des_type': "continuous",
                    'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert(moop.n == 3)
    moop.addDesign({'lb': 0.0,
                    'ub': 1.0})
    assert(moop.n == 4)
    moop.addDesign({'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert(moop.n == 5)
    moop.addDesign({'name': "x_6",
                    'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert(moop.n == 6)
    # Now add some good categorical design variables
    assert(moop.n_cat == 0)
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    assert(moop.n_cat == 1)
    moop.addDesign({'des_type': "categorical",
                    'levels': 3})
    assert(moop.n_cat == 2)
    moop.addDesign({'name': "x_9",
                    'des_type': "categorical",
                    'levels': 3})
    assert(moop.n_cat == 3)
    moop.addDesign({'name': "x_10",
                    'des_type': "categorical",
                    'levels': ["boy", "girl", "doggo"]})
    assert(moop.n_cat == 4)
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addDesign({'name': "x_9",
                        'des_type': "continuous",
                        'lb': 0.0,
                        'ub': 1.0})
    # Now add more continuous design variables
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    assert(moop.n_cont == 7)
    moop.addDesign({'lb': 0.0,
                    'ub': 1.0})
    assert(moop.n_cont == 8)


def test_MOOP_embed_extract():
    """ Test that the MOOP class is able to embed/extract design variables.

    Add several design variables and generate an embedding. Then embed and
    extract several inputs, and check that the results match up to the
    design tolerance. This test applies to the three hidden methods:
     * MOOP.__embed__(x)
     * MOOP.__extract__(x)
     * MOOP.__generate_encoding__()

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalGPS)
    # Add two continuous variables and check that they are embedded correctly
    moop.addDesign({'lb': 0.0,
                    'ub': 1000.0})
    moop.addDesign({'lb': -1.0,
                    'ub': 0.0})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(2)
        xi[0] *= 1000.0
        xi[1] -= 1.0
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert(all(xxi >= 0.0) and all(xxi <= 1.0))
        assert(xxi.size == moop.n)
        # Check extraction
        assert(all(moop.__extract__(xxi) - xi < 1.0e-8))
    # Test upper and lower bounds
    x0 = np.zeros(2)
    x0[0] *= 1000.0
    x0[1] -= 1.0
    xx0 = moop.__embed__(x0)
    # Check that embedding is legal
    assert(all(xx0 >= 0.0) and all(xx0 <= 1.0))
    assert(xx0.size == moop.n)
    # Check extraction
    assert(all(moop.__extract__(xx0) - x0 < 1.0e-8))
    x1 = np.ones(2)
    x1[0] *= 1000.0
    x1[1] -= 1.0
    xx1 = moop.__embed__(x1)
    # Check that embedding is legal
    assert(all(xx1 >= 0.0) and all(xx1 <= 1.0))
    assert(xx1.size == moop.n)
    # Check extraction
    assert(all(moop.__extract__(xx1) - x1 < 1.0e-8))
    # Add two categorical variables and check that they are embedded correctly
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    moop.addDesign({'des_type': "categorical",
                    'levels': 3})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(4)
        xi[0] *= 1000.0
        xi[1] -= 1.0
        xi[2:] = np.round(xi[2:])
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert(all(xxi >= 0.0) and all(xxi <= 1.0))
        assert(xxi.size == moop.n)
        # Check extraction
        assert(all(moop.__extract__(xxi) - xi < 1.0e-8))
    # Test upper and lower bounds
    x0 = np.zeros(4)
    x0[0] *= 1000.0
    x0[1] -= 1.0
    x0[2:] = np.round(x0[2:])
    xx0 = moop.__embed__(x0)
    # Check that embedding is legal
    assert(all(xx0 >= 0.0) and all(xx0 <= 1.0))
    assert(xx0.size == moop.n)
    # Check extraction
    assert(all(moop.__extract__(xx0) - x0 < 1.0e-8))
    x1 = np.ones(4)
    x1[0] *= 1000.0
    x1[1] -= 1.0
    x1[2:] = np.round(x1[2:])
    xx1 = moop.__embed__(x1)
    # Check that embedding is legal
    assert(all(xx1 >= 0.0) and all(xx1 <= 1.0))
    assert(xx1.size == moop.n)
    # Check extraction
    assert(all(moop.__extract__(xx1) - x1 < 1.0e-8))
    # Same as above, but reverse the order
    moop = MOOP(LocalGPS)
    # Add two categorical variables and check that they are embedded correctly
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    moop.addDesign({'des_type': "categorical",
                    'levels': 3})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(2)
        xi = np.round(xi)
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert(all(xxi >= 0.0) and all(xxi <= 1.0))
        assert(xxi.size == moop.n)
        # Check extraction
        assert(all(moop.__extract__(xxi) - xi < 1.0e-8))
    # Test upper and lower bounds
    x0 = np.zeros(2)
    x0 = np.round(x0)
    xx0 = moop.__embed__(x0)
    # Check that embedding is legal
    assert(all(xx0 >= 0.0) and all(xx0 <= 1.0))
    assert(xx0.size == moop.n)
    # Check extraction
    assert(all(moop.__extract__(xx0) - x0 < 1.0e-8))
    x1 = np.ones(2)
    x1 = np.round(x1)
    xx1 = moop.__embed__(x1)
    # Check that embedding is legal
    assert(all(xx1 >= 0.0) and all(xx1 <= 1.0))
    assert(xx1.size == moop.n)
    # Check extraction
    assert(all(moop.__extract__(xx1) - x1 < 1.0e-8))
    # Add two continuous variables and check that they are embedded correctly
    moop.addDesign({'lb': 0.0,
                    'ub': 1000.0})
    moop.addDesign({'lb': -1.0,
                    'ub': 0.0})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(4)
        xi[:2] = np.round(xi[:2])
        xi[2] *= 1000.0
        xi[3] -= 1.0
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert(all(xxi >= 0.0) and all(xxi <= 1.0))
        assert(xxi.size == moop.n)
        # Check extraction
        assert(all(moop.__extract__(xxi) - xi < 1.0e-8))
    # Test upper and lower bounds
    x0 = np.zeros(4)
    x0[:2] = np.round(x0[:2])
    x0[2] *= 1000.0
    x0[3] -= 1.0
    xx0 = moop.__embed__(x0)
    # Check that embedding is legal
    assert(all(xx0 >= 0.0) and all(xx0 <= 1.0))
    assert(xx0.size == moop.n)
    # Check extraction
    assert(all(moop.__extract__(xx0) - x0 < 1.0e-8))
    x1 = np.ones(4)
    x1[:2] = np.round(x1[:2])
    x1[2] *= 1000.0
    x1[3] -= 1.0
    xx1 = moop.__embed__(x1)
    # Check that embedding is legal
    assert(all(xx1 >= 0.0) and all(xx1 <= 1.0))
    assert(xx1.size == moop.n)
    # Check extraction
    assert(all(moop.__extract__(xx1) - x1 < 1.0e-8))
    # Now, create another MOOP where all variables are labeled
    moop = MOOP(LocalGPS)
    # Add two continuous variables and check that they are embedded correctly
    moop.addDesign({'name': "x0",
                    'lb': 0.0,
                    'ub': 1000.0})
    moop.addDesign({'name': "x1",
                    'lb': -1.0,
                    'ub': 0.0})
    # Test 5 random variables
    for i in range(5):
        nums = np.random.random_sample(2)
        xi = np.zeros(1, dtype=[("x0", float), ("x1", float)])
        xi["x0"] = 1000.0 * nums[0]
        xi["x1"] = nums[1] - 1.0
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert(all(xxi >= 0.0) and all(xxi <= 1.0))
        assert(xxi.size == moop.n)
        # Check extraction
        assert(all([moop.__extract__(xxi)[key] - xi[key] < 1.0e-8
                    for key in ["x0", "x1"]]))
    # Add two categorical variables and check that they are embedded correctly
    moop.addDesign({'name': "x2",
                    'des_type': "categorical",
                    'levels': 2})
    moop.addDesign({'name': "x3",
                    'des_type': "categorical",
                    'levels': ["biggie", "shortie", "shmedium"]})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(4)
        xi = np.zeros(1, dtype=[("x0", float), ("x1", float), ("x2", float),
                                ("x3", object)])
        xi["x0"] = 1000.0 * num[0]
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert(all(xxi >= 0.0) and all(xxi <= 1.0))
        assert(xxi.size == moop.n)
        # Check extraction
        assert(all([moop.__extract__(xxi)[key] - xi[key] < 1.0e-8
                    for key in ["x0", "x1", "x2"]]))
        assert(moop.__extract__(xxi)["x3"] == xi["x3"])


def test_MOOP_addSimulation():
    """ Check that the MOOP class handles adding new simulations properly.

    Initialize several MOOPs, and add several simulations. Check that
    the metadata is updated correctly.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalGPS
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
    moop = MOOP(LocalGPS)
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
    assert(moop.n == 3 and moop.s == 1 and moop.m_total == 1
           and moop.o == 0 and moop.p == 0)
    # Initialize another MOOP with 3 design variables
    moop = MOOP(LocalGPS)
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
    assert(moop.n == 3 and moop.s == 2 and moop.m_total == 3
           and moop.o == 0 and moop.p == 0)
    # Now test adding simulations with empty precomputed databases
    g1['sim_db'] = {}
    moop = MOOP(LocalGPS)
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addSimulation(g1)
    assert(moop.n == 3 and moop.s == 1 and moop.m_total == 1
           and moop.o == 0 and moop.p == 0)
    assert(moop.sim_db[0]['n'] == 0)
    # Now test adding a simulation with nonempty database, but empty lists
    g1['sim_db'] = {'x_vals': [], 's_vals': []}
    moop = MOOP(LocalGPS)
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addSimulation(g1)
    assert(moop.n == 3 and moop.s == 1 and moop.m_total == 1
           and moop.o == 0 and moop.p == 0)
    assert(moop.sim_db[0]['n'] == 0)
    # Now try a simulation with some data
    g1['sim_db'] = {'x_vals': [[0.0, 0.0, 0.0]], 's_vals': [[0.0]]}
    moop = MOOP(LocalGPS)
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addSimulation(g1)
    assert(moop.n == 3 and moop.s == 1 and moop.m_total == 1
           and moop.o == 0 and moop.p == 0)
    assert(moop.sim_db[0]['n'] == 1)
    moop.addSimulation(g3, g4)
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addSimulation(g4)
    # Check the names
    assert(moop.sim_names[0][0] == "sim1")
    assert(moop.sim_names[1][0] == "Bobo1")
    assert(moop.sim_names[2][0] == "Bobo2")


def test_pack_unpack_sim():
    """ Check that the MOOP class handles simulation packing/packing correctly

    Initialize a MOOP objecti with and without design variable names.
    Add 2 simulations and pack/unpack each output.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalGPS
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
    # Create an input
    sx = np.zeros(3)
    sx[0] = 1.0
    sx[1] = 2.0
    sx[2] = 3.0
    # Create a MOOP without named variables and test simulation unpacking
    moop = MOOP(LocalGPS)
    # Add two continuous variables and two simulations
    moop.addDesign({'lb': 0.0, 'ub': 1000.0},
                   {'lb': -1.0, 'ub': 0.0})
    moop.addSimulation(g1, g2)
    assert(all(moop.__pack_sim__(sx)[:] == sx[:]))
    assert(all(moop.__unpack_sim__(sx)[:] == sx[:]))
    # Create a MOOP with named variables and test simulation unpacking
    moop = MOOP(LocalGPS)
    # Add two continuous variables and two simulations
    moop.addDesign({'name': "x0", 'lb': 0.0, 'ub': 1000.0},
                   {'name': "x1", 'lb': -1.0, 'ub': 0.0})
    moop.addSimulation(g1, g2)
    # Create a solution vector
    sxx = np.zeros(1, dtype=moop.sim_names)
    sxx[0]['sim1'] = 1.0
    sxx[0]['sim2'][:] = np.array([2.0, 3.0])
    # Check packing
    assert(all(moop.__pack_sim__(sxx) == sx))
    # Check unpacking
    assert(moop.__unpack_sim__(sx)['sim1'] == sxx[0]['sim1'])
    assert(moop.__unpack_sim__(sx)['sim2'][0] == sxx[0]['sim2'][0])
    assert(moop.__unpack_sim__(sx)['sim2'][1] == sxx[0]['sim2'][1])


def test_MOOP_addObjective():
    """ Check that the MOOP class handles adding objectives properly.

    Initialize a MOOP object and check that the addObjective() function works
    correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
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
    moop = MOOP(LocalGPS)
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
    with pytest.raises(TypeError):
        moop.addObjective({'name': 5, 'obj_func': lambda x, s: 0.0})
    # Add an objective after an acquisition
    with pytest.raises(RuntimeError):
        moop1 = MOOP(LocalGPS)
        moop1.acquisitions.append(0)
        moop1.addObjective({'obj_func': lambda x, s: 0.0})
    # Check that no objectives were added yet
    assert(moop.o == 0)
    # Now add 3 good objectives
    moop.addObjective({'obj_func': lambda x, s: x[0]})
    assert(moop.o == 1)
    moop.addObjective({'obj_func': lambda x, s: s[0]},
                      {'obj_func': lambda x, s, der=0: s[1]})
    assert(moop.o == 3)
    moop.addObjective({'name': "Bobo", 'obj_func': lambda x, s: s[0]})
    assert(moop.o == 4)
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addObjective({'name': "Bobo", 'obj_func': lambda x, s: s[0]})
    assert(moop.obj_names[0] == ("f1", float))
    assert(moop.obj_names[1] == ("f2", float))
    assert(moop.obj_names[2] == ("f3", float))
    assert(moop.obj_names[3] == ("Bobo", float))


def test_MOOP_addConstraint():
    """ Check that the MOOP class handles adding constraints properly.

    Initialize a MOOP object and check that the addConstraint() function works
    correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
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
    moop = MOOP(LocalGPS)
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
    with pytest.raises(TypeError):
        moop.addConstraint({'name': 5, 'constraint': lambda x, s: 0.0})
    # Check that no constraints were added yet
    assert(moop.p == 0)
    # Now add 3 good constraints
    moop.addConstraint({'constraint': lambda x, s: x[0]})
    assert(moop.p == 1)
    moop.addConstraint({'constraint': lambda x, s: s[0]},
                       {'constraint': lambda x, s, der=0: s[1] + s[2]})
    assert(moop.p == 3)
    moop.addConstraint({'name': "Bobo", 'constraint': lambda x, s: s[0]})
    assert(moop.p == 4)
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addConstraint({'name': "Bobo", 'constraint': lambda x, s: s[0]})
    assert(moop.const_names[0] == ("c1", float))
    assert(moop.const_names[1] == ("c2", float))
    assert(moop.const_names[2] == ("c3", float))
    assert(moop.const_names[3] == ("Bobo", float))


def test_MOOP_addAcquisition():
    """ Check that the MOOP class handles adding acquisition functions properly.

    Initialize a MOOP object and check that the addAcquisition() function works
    correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
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
    moop = MOOP(LocalGPS)
    # Try to add acquisition functions without design variables
    with pytest.raises(ValueError):
        moop.addAcquisition({'acquisition': UniformWeights})
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    # Try to add acquisition functions without objectives
    with pytest.raises(ValueError):
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
    assert(len(moop.acquisitions) == 0)
    # Now add 3 good acquisitions
    moop.addAcquisition({'acquisition': UniformWeights})
    assert(len(moop.acquisitions) == 1)
    moop.addAcquisition({'acquisition': UniformWeights},
                        {'acquisition': UniformWeights, 'hyperparams': {}})
    assert(len(moop.acquisitions) == 3)


def test_MOOP_getTypes():
    """ Check that the MOOP class handles getting dtypes properly.

    Initialize a MOOP object, add design variables, simulations, objectives,
    and constraints, and get the corresponding types.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create a simulation for later
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}

    # Create a new MOOP
    moop = MOOP(LocalGPS)
    # Check that all types are None
    assert(moop.getDesignType() is None)
    assert(moop.getSimulationType() is None)
    assert(moop.getObjectiveType() is None)
    assert(moop.getConstraintType() is None)
    # Add some unnamed variables, simulations, objectives, and constraints
    moop.addDesign({'des_type': "continuous", 'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'des_type': "categorical", 'levels': 3})
    moop.addSimulation(g1)
    moop.addObjective({'obj_func': lambda x, s: [sum(s)]})
    moop.addConstraint({'constraint': lambda x, s: [sum(s) - 1]})
    assert(np.zeros(1, dtype=moop.getDesignType()).size == 2)
    assert(np.zeros(1, dtype=moop.getSimulationType()).size == 1)
    assert(np.zeros(1, dtype=moop.getObjectiveType()).size == 1)
    assert(np.zeros(1, dtype=moop.getConstraintType()).size == 1)
    # Add some named variables, simulations, objectives, and constraints
    moop = MOOP(LocalGPS)
    moop.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'name': "x2", 'des_type': "categorical", 'levels': 3})
    moop.addSimulation(g1)
    moop.addObjective({'obj_func': lambda x, s: [sum(s)]})
    moop.addConstraint({'constraint': lambda x, s: [sum(s) - 1]})
    assert(np.zeros(1, dtype=moop.getDesignType()).size == 1)
    assert(np.zeros(1, dtype=moop.getSimulationType()).size == 1)
    assert(np.zeros(1, dtype=moop.getObjectiveType()).size == 1)
    assert(np.zeros(1, dtype=moop.getConstraintType()).size == 1)


def test_MOOP_evaluateSimulation():
    """ Check that the MOOP class handles evaluating simulations properly.

    Initialize a MOOP object and check that the evaluateSimulation() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
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
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1)
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'name': "x" + str(i+1), 'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g2)
    x1 = np.zeros(3)
    x2 = np.zeros(1, dtype=moop2.des_names)[0]
    y1 = np.ones(3)
    y2 = np.ones(1, dtype=moop2.des_names)[0]
    # Check database with bad values
    with pytest.raises(TypeError):
        moop1.check_sim_db(x1, 5.0)
    with pytest.raises(ValueError):
        moop1.check_sim_db(x1, -1)
    with pytest.raises(ValueError):
        moop2.check_sim_db(x2, "hello world")
    # Update database with bad values
    with pytest.raises(TypeError):
        moop1.update_sim_db(x1, np.zeros(1), 5.0)
    with pytest.raises(ValueError):
        moop1.update_sim_db(x1, np.zeros(1), -1)
    with pytest.raises(ValueError):
        moop2.update_sim_db(x2, np.zeros(1), "hello world")
    # Evaluate simulation with bad values
    with pytest.raises(TypeError):
        moop1.check_sim_db(x1, 5.0)
    with pytest.raises(ValueError):
        moop1.check_sim_db(x1, -1)
    with pytest.raises(ValueError):
        moop2.check_sim_db(x2, "hello world")
    # Try 2 good evaluations
    moop1.evaluateSimulation(x1, 0)
    assert(moop1.check_sim_db(x1, 0) is not None)
    moop1.evaluateSimulation(y1, 0)
    assert(moop1.check_sim_db(y1, 0) is not None)
    moop2.evaluateSimulation(x2, "g2")
    assert(moop2.check_sim_db(x2, "g2") is not None)
    moop2.evaluateSimulation(y2, "g2")
    assert(moop2.check_sim_db(y2, "g2") is not None)
    return


def test_MOOP_evaluateSurrogates():
    """ Check that the MOOP class handles evaluating objectives properly.

    Initialize a MOOP object and check that the evaluateSurrogates() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
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
    moop1 = MOOP(LocalGPS)
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
    moop1.evaluateSimulation(np.zeros(3), 0)
    moop1.evaluateSimulation(np.zeros(3), 1)
    moop1.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 0)
    moop1.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 1)
    moop1.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 0)
    moop1.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 1)
    moop1.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 0)
    moop1.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 1)
    moop1.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 0)
    moop1.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 1)
    moop1.evaluateSimulation(np.ones(3), 0)
    moop1.evaluateSimulation(np.ones(3), 1)
    moop1.fitSurrogates()
    moop1.resetSurrogates(np.ones(3) * 0.5)
    # Now try some bad evaluations
    with pytest.raises(ValueError):
        moop1.evaluateSurrogates(10.0)
    with pytest.raises(ValueError):
        moop1.evaluateSurrogates(np.zeros(1))
    # Now do some good evaluations and check the results
    assert(np.linalg.norm(moop1.evaluateSurrogates(np.zeros(3)) -
                          np.asarray([0.0, 0.0, np.sqrt(3.0) + np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateSurrogates(np.asarray([0.5, 0.5, 0.5]))
                          - np.asarray([0.5, np.sqrt(0.75), np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateSurrogates(np.asarray([1.0, 0.0, 0.0]))
                          - np.asarray([1.0, 1.0, np.sqrt(2.0) +
                                        np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateSurrogates(np.asarray([0.0, 1.0, 0.0]))
                          - np.asarray([0.0, 1.0, np.sqrt(2.0) +
                                        np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateSurrogates(np.asarray([0.0, 0.0, 1.0]))
                          - np.asarray([0.0, 1.0, np.sqrt(2.0) +
                                        np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateSurrogates(np.ones(3)) -
                          np.asarray([1.0, np.sqrt(3.0), np.sqrt(0.75)]))
           < 0.00000001)
    # Adjust the scale and try again
    moop2 = MOOP(LocalGPS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addObjective({'obj_func': lambda x, s: x[0]},
                       {'obj_func': lambda x, s: s[0]},
                       {'obj_func': lambda x, s: s[1] + s[2]})
    moop2.addSimulation(g1, g2)
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), 0)
    moop2.evaluateSimulation(np.zeros(3), 1)
    moop2.evaluateSimulation(np.asarray([0.5, 0.5, 0.5]), 0)
    moop2.evaluateSimulation(np.asarray([0.5, 0.5, 0.5]), 1)
    moop2.evaluateSimulation(np.asarray([1.0, 0.0, 0.0]), 0)
    moop2.evaluateSimulation(np.asarray([1.0, 0.0, 0.0]), 1)
    moop2.evaluateSimulation(np.asarray([0.0, 1.0, 0.0]), 0)
    moop2.evaluateSimulation(np.asarray([0.0, 1.0, 0.0]), 1)
    moop2.evaluateSimulation(np.asarray([0.0, 0.0, 1.0]), 0)
    moop2.evaluateSimulation(np.asarray([0.0, 0.0, 1.0]), 1)
    moop2.evaluateSimulation(np.ones(3), 0)
    moop2.evaluateSimulation(np.ones(3), 1)
    moop2.fitSurrogates()
    # Now compare evaluations against the original surrogate
    x = moop1.__embed__(np.zeros(3))
    xx = moop2.__embed__(np.zeros(3))
    assert(np.linalg.norm(moop1.evaluateSurrogates(x) -
                          moop2.evaluateSurrogates(xx)) < 0.00000001)
    x = moop1.__embed__(np.ones(3))
    xx = moop2.__embed__(np.ones(3))
    assert(np.linalg.norm(moop1.evaluateSurrogates(x) -
                          moop2.evaluateSurrogates(xx)) < 0.00000001)


def test_MOOP_evaluateConstraints():
    """ Check that the MOOP class handles evaluating constraints properly.

    Initialize a MOOP object and check that the evaluateConstraints() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
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
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    # Evaluate an empty constraint and check that a zero array is returned
    assert(all(moop1.evaluateConstraints(np.zeros(3)) == np.zeros(1)))
    # Now add 3 constraints
    moop1.addConstraint({'constraint': lambda x, s: x[0]})
    moop1.addConstraint({'constraint': lambda x, s: s[0]})
    moop1.addConstraint({'constraint': lambda x, s: s[1] + s[2]})
    # Evaluate some data points and fit the surrogates
    moop1.evaluateSimulation(np.zeros(3), 0)
    moop1.evaluateSimulation(np.zeros(3), 1)
    moop1.evaluateSimulation(np.asarray([0.5, 0.5, 0.5]), 0)
    moop1.evaluateSimulation(np.asarray([0.5, 0.5, 0.5]), 1)
    moop1.evaluateSimulation(np.asarray([1.0, 0.0, 0.0]), 0)
    moop1.evaluateSimulation(np.asarray([1.0, 0.0, 0.0]), 1)
    moop1.evaluateSimulation(np.asarray([0.0, 1.0, 0.0]), 0)
    moop1.evaluateSimulation(np.asarray([0.0, 1.0, 0.0]), 1)
    moop1.evaluateSimulation(np.asarray([0.0, 0.0, 1.0]), 0)
    moop1.evaluateSimulation(np.asarray([0.0, 0.0, 1.0]), 1)
    moop1.evaluateSimulation(np.ones(3), 0)
    moop1.evaluateSimulation(np.ones(3), 1)
    moop1.fitSurrogates()
    # Now try some bad evaluations
    with pytest.raises(ValueError):
        moop1.evaluateConstraints(10.0)
    with pytest.raises(ValueError):
        moop1.evaluateConstraints(np.zeros(1))
    # Now do some good evaluations and check the results
    assert(np.linalg.norm(moop1.evaluateConstraints(np.zeros(3)) -
                          np.asarray([0.0, 0.0, np.sqrt(3.0) + np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateConstraints(np.asarray([0.5, 0.5,
                                                                0.5]))
                          - np.asarray([0.5, np.sqrt(0.75), np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateConstraints(np.asarray([1.0, 0.0,
                                                                0.0]))
                          - np.asarray([1.0, 1.0, np.sqrt(2.0)
                                        + np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateConstraints(np.asarray([0.0, 1.0,
                                                                0.0]))
                          - np.asarray([0.0, 1.0, np.sqrt(2.0)
                                        + np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateConstraints(np.asarray([0.0, 0.0,
                                                                1.0]))
                          - np.asarray([0.0, 1.0, np.sqrt(2.0)
                                        + np.sqrt(0.75)]))
           < 0.00000001)
    assert(np.linalg.norm(moop1.evaluateConstraints(np.ones(3)) -
                          np.asarray([1.0, np.sqrt(3.0), np.sqrt(0.75)]))
           < 0.00000001)
    # Adjust the scale and try again
    moop2 = MOOP(LocalGPS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.addConstraint({'constraint': lambda x, s: x[0]})
    moop2.addConstraint({'constraint': lambda x, s: s[0]})
    moop2.addConstraint({'constraint': lambda x, s: s[1] + s[2]})
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), 0)
    moop2.evaluateSimulation(np.zeros(3), 1)
    moop2.evaluateSimulation(np.asarray([0.5, 0.5, 0.5]), 0)
    moop2.evaluateSimulation(np.asarray([0.5, 0.5, 0.5]), 1)
    moop2.evaluateSimulation(np.asarray([1.0, 0.0, 0.0]), 0)
    moop2.evaluateSimulation(np.asarray([1.0, 0.0, 0.0]), 1)
    moop2.evaluateSimulation(np.asarray([0.0, 1.0, 0.0]), 0)
    moop2.evaluateSimulation(np.asarray([0.0, 1.0, 0.0]), 1)
    moop2.evaluateSimulation(np.asarray([0.0, 0.0, 1.0]), 0)
    moop2.evaluateSimulation(np.asarray([0.0, 0.0, 1.0]), 1)
    moop2.evaluateSimulation(np.ones(3), 0)
    moop2.evaluateSimulation(np.ones(3), 1)
    moop2.fitSurrogates()
    # Now compare evaluations against the original surrogate
    x = moop1.__embed__(np.zeros(3))
    xx = moop2.__embed__(np.zeros(3))
    assert(np.linalg.norm(moop1.evaluateConstraints(x) -
                          moop2.evaluateConstraints(xx)) < 0.00000001)
    x = moop1.__embed__(np.ones(3))
    xx = moop2.__embed__(np.ones(3))
    assert(np.linalg.norm(moop1.evaluateConstraints(x) -
                          moop2.evaluateConstraints(xx)) < 0.00000001)


def test_MOOP_evaluateLagrangian():
    """ Check that the MOOP class handles evaluating Lagrangian properly.

    Initialize a MOOP object and check that the evaluateLagrangian() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
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

    # Create several differentiable functions and constraints.
    def f1(x, s, der=0):
        if der == 0:
            return np.dot(x, x)
        if der == 1:
            return 2.0 * x
        if der == 2:
            return np.zeros(s.size)

    def f2(x, s, der=0):
        if der == 0:
            return np.dot(s - 0.5, s - 0.5)
        if der == 1:
            return np.zeros(x.size)
        if der == 2:
            return 2.0 * s - 1.0

    def c1(x, s, der=0):
        if der == 0:
            return x[0] - 0.25
        if der == 1:
            return np.ones(x.size)
        if der == 2:
            return np.zeros(x.size)

    def c2(x, s, der=0):
        if der == 0:
            return s[0] - 0.25
        if der == 1:
            return np.zeros(s.size)
        if der == 2:
            return np.ones(s.size)

    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addObjective({'obj_func': f1})
    assert(np.all(moop1.evaluateLagrangian(np.zeros(3)) == np.zeros(1)))
    assert(np.all(moop1.evaluateLagrangian(np.ones(3)) == 3.0 * np.ones(1)))
    moop1.addConstraint({'constraint': c1})
    assert(np.all(moop1.evaluateLagrangian(np.zeros(3)) == np.zeros(1)))
    assert(np.all(moop1.evaluateLagrangian(np.ones(3)) == 3.75 * np.ones(1)))
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.evaluateSimulation(np.ones(3), 0)
    moop1.evaluateSimulation(np.ones(3), 1)
    moop1.fitSurrogates()
    moop1.addObjective({'obj_func': f1})
    moop1.addObjective({'obj_func': f1})
    assert(np.all(moop1.evaluateLagrangian(np.zeros(3)) == np.zeros(1)))
    assert(np.all(moop1.evaluateLagrangian(np.ones(3)) == 3.0 * np.ones(1)))
    moop1.addConstraint({'constraint': c1})
    assert(np.all(moop1.evaluateLagrangian(np.zeros(3)) == np.zeros(1)))
    assert(np.all(moop1.evaluateLagrangian(np.ones(3)) == 3.75 * np.ones(1)))
    # Now try some bad evaluations
    with pytest.raises(ValueError):
        moop1.evaluateLagrangian(10.0)
    with pytest.raises(ValueError):
        moop1.evaluateLagrangian(np.zeros(1))
    # Adjust the scaling and compare
    moop2 = MOOP(LocalGPS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.evaluateSimulation(np.ones(3), 0)
    moop2.evaluateSimulation(np.ones(3), 1)
    moop2.fitSurrogates()
    moop2.addObjective({'obj_func': f1})
    moop2.addObjective({'obj_func': f1})
    moop2.addConstraint({'constraint': c1})
    x = moop1.__embed__(np.ones(3))
    xx = moop2.__embed__(np.ones(3))
    assert(np.linalg.norm(moop1.evaluateLagrangian(x) -
                          moop2.evaluateLagrangian(xx)) < 0.00000001)


def test_MOOP_evaluateGradients():
    """ Check that the MOOP class handles evaluating gradients properly.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
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

    # Create several differentiable functions and constraints.
    def f1(x, s, der=0):
        if der == 0:
            return np.dot(x, x)
        if der == 1:
            return 2.0 * x
        if der == 2:
            return np.zeros(s.size)

    def f2(x, s, der=0):
        if der == 0:
            return np.dot(s - 0.5, s - 0.5)
        if der == 1:
            return np.zeros(x.size)
        if der == 2:
            return 2.0 * s - 1.0

    def c1(x, s, der=0):
        if der == 0:
            return x[0] - 0.25
        if der == 1:
            return np.eye(x.size)[0]
        if der == 2:
            return np.zeros(s.size)

    def c2(x, s, der=0):
        if der == 0:
            return s[0] - 0.25
        if der == 1:
            return np.zeros(x.size)
        if der == 2:
            return np.eye(s.size)[0]

    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addObjective({'obj_func': f1})
    assert(np.all(moop1.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert(np.all(moop1.evaluateGradients(np.ones(3)) ==
                  2.0 * np.ones((1, 3))))
    moop1.addConstraint({'constraint': c1})
    assert(np.all(moop1.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    result = 2.0 * np.ones((1, 3))
    result[0, 0] = 3.0
    assert(np.all(moop1.evaluateGradients(np.ones(3)) == result))
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.evaluateSimulation(np.ones(3), 0)
    moop1.evaluateSimulation(np.ones(3), 1)
    moop1.fitSurrogates()
    moop1.addObjective({'obj_func': f1})
    assert(np.all(moop1.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert(np.all(moop1.evaluateGradients(np.ones(3)) ==
                  2.0 * np.ones((1, 3))))
    moop1.addConstraint({'constraint': c1})
    assert(np.all(moop1.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert(np.all(moop1.evaluateGradients(np.ones(3)) == result))
    result = np.zeros((2, 3))
    result[1, 0] = 1.0
    result[0, :] = 2.0
    result[0, 0] = 3.0
    moop1.addObjective({'obj_func': f2})
    assert(np.all(moop1.evaluateGradients(np.ones(3)) == result))
    moop1.addConstraint({'constraint': c2})
    assert(np.all(moop1.evaluateGradients(np.ones(3)) == result))
    # Now try some bad evaluations
    with pytest.raises(ValueError):
        moop1.evaluateGradients(10.0)
    with pytest.raises(ValueError):
        moop1.evaluateGradients(np.zeros(1))
    # Adjust the scaling and try again
    moop2 = MOOP(LocalGPS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.evaluateSimulation(np.ones(3), 0)
    moop2.evaluateSimulation(np.ones(3), 1)
    moop2.fitSurrogates()
    moop2.addObjective({'obj_func': f1})
    moop2.addConstraint({'constraint': c1})
    moop2.addObjective({'obj_func': f2})
    moop2.addConstraint({'constraint': c2})
    x = moop1.__embed__(np.ones(3))
    xx = moop2.__embed__(np.ones(3))
    assert(np.linalg.norm(moop1.evaluateLagrangian(x) -
                          moop2.evaluateLagrangian(xx)) < 0.00000001)

    # Initialize a MOOP with 2 SimGroups and 3 objectives with named designs
    g3 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: sum([x[name] ** 2.0
                                     for name in x.dtype.names]),
          'surrogate': GaussRBF}
    g4 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: sum([(x[name] - 1.0) * (x[name] - 0.5)
                                     for name in x.dtype.names]),
          'surrogate': GaussRBF}

    def f3(x, s, der=0):
        if der == 0:
            result = 0.0
            for name in x.dtype.names:
                result = result + x[name] ** 2
            return result
        if der == 1:
            result = np.zeros(1, dtype=x.dtype)
            for name in x.dtype.names:
                result[name] = 2.0 * x[name]
            return result[0]
        if der == 2:
            return np.zeros(1, dtype=s.dtype)[0]

    def f4(x, s, der=0):
        if der == 0:
            result = 0.0
            for name in s.dtype.names:
                if isinstance(s[name], np.ndarray):
                    result = result + np.dot(s[name] - 0.5, s[name] - 0.5)
                else:
                    result = result + (s[name] - 0.5) ** 2
            return result
        if der == 1:
            return np.zeros(1, dtype=x.dtype)[0]
        if der == 2:
            result = np.zeros(1, dtype=s.dtype)
            for name in s.dtype.names:
                result[name] = 2.0 * s[name] - 1.0
            return result[0]

    def c3(x, s, der=0):
        if der == 0:
            return x["x1"] - 0.25
        if der == 1:
            result = np.zeros(1, dtype=x.dtype)
            result["x1"] = 1.0
            return result[0]
        if der == 2:
            return np.zeros(1, dtype=s.dtype)[0]

    def c4(x, s, der=0):
        if der == 0:
            return s['sim1'] - 0.25
        if der == 1:
            return np.zeros(1, dtype=x.dtype)[0]
        if der == 2:
            result = np.zeros(1, dtype=s.dtype)
            result['sim1'] = 1.0
            return result[0]

    moop3 = MOOP(LocalGPS)
    for i in range(3):
        moop3.addDesign({'name': ('x' + str(i + 1)), 'lb': 0.0, 'ub': 1.0})
    moop3.addObjective({'obj_func': f3})
    assert(np.all(moop3.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert(np.all(moop3.evaluateGradients(np.ones(3)) ==
                  2.0 * np.ones((1, 3))))
    moop3.addConstraint({'constraint': c3})
    assert(np.all(moop3.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    result = 2.0 * np.ones((1, 3))
    result[0, 0] = 3.0
    assert(np.all(moop3.evaluateGradients(np.ones(3)) == result))
    moop3 = MOOP(LocalGPS)
    for i in range(3):
        moop3.addDesign({'name': ('x' + str(i + 1)), 'lb': 0.0, 'ub': 1.0})
    moop3.addSimulation(g3, g4)
    moop3.evaluateSimulation(np.ones(1, dtype=[("x1", float), ("x2", float),
                                               ("x3", float)]), 0)
    moop3.evaluateSimulation(np.ones(1, dtype=[("x1", float), ("x2", float),
                                               ("x3", float)]), 1)
    moop3.fitSurrogates()
    moop3.addObjective({'obj_func': f3})
    assert(np.all(moop3.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert(np.all(moop3.evaluateGradients(np.ones(3)) ==
                  2.0 * np.ones((1, 3))))
    moop3.addConstraint({'constraint': c3})
    assert(np.all(moop3.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert(np.all(moop3.evaluateGradients(np.ones(3)) == result))
    result = np.zeros((2, 3))
    result[1, 0] = 1.0
    result[0, :] = 2.0
    result[0, 0] = 3.0
    moop3.addObjective({'obj_func': f4})
    assert(np.all(moop1.evaluateGradients(np.ones(3)) == result))
    moop3.addConstraint({'constraint': c4})
    assert(np.all(moop1.evaluateGradients(np.ones(3)) == result))
    # Adjust the scaling and try again
    moop4 = MOOP(LocalGPS)
    moop4.addDesign({'name': "x1", 'lb': -1.0, 'ub': 1.0},
                    {'name': "x2", 'lb': 0.0, 'ub': 2.0},
                    {'name': "x3", 'lb': -0.5, 'ub': 1.5})
    moop4.addSimulation(g3, g4)
    moop4.evaluateSimulation(np.ones(1, dtype=[("x1", float), ("x2", float),
                                               ("x3", float)]), 0)
    moop4.evaluateSimulation(np.ones(1, dtype=[("x1", float), ("x2", float),
                                               ("x3", float)]), 1)
    moop4.fitSurrogates()
    moop4.addObjective({'obj_func': f3})
    moop4.addConstraint({'constraint': c3})
    moop4.addObjective({'obj_func': f4})
    moop4.addConstraint({'constraint': c4})
    x = moop3.__embed__(np.ones(1, dtype=[("x1", float), ("x2", float),
                                          ("x3", float)]))
    xx = moop4.__embed__(np.ones(1, dtype=[("x1", float), ("x2", float),
                                           ("x3", float)]))
    assert(np.linalg.norm(moop3.evaluateLagrangian(x) -
                          moop4.evaluateLagrangian(xx)) < 0.00000001)


def test_MOOP_addData():
    """ Check that the MOOP class is able to add data to its internal database.

    Initialize a MOOP object and check that the addData(s, sx) function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
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
    moop1 = MOOP(LocalGPS)
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
    assert(moop1.data['f_vals'].shape == (2, 2))
    assert(moop1.data['x_vals'].shape == (2, 3))
    assert(moop1.data['c_vals'].shape == (2, 1))
    assert(moop1.n_dat == 2)
    # Initialize a new MOOP with 2 SimGroups and 2 objectives
    moop2 = MOOP(LocalGPS)
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
    moop2.addData(np.asarray([0.0, 0.0, 1.0]), np.zeros(3))
    moop2.addData(np.ones(3), np.ones(3))
    assert(moop2.data['f_vals'].shape == (3, 2))
    assert(moop2.data['x_vals'].shape == (3, 3))
    assert(moop2.data['c_vals'].shape == (3, 3))
    assert(moop2.n_dat == 3)
    # Initialize a new MOOP with 2 SimGroups and 2 objectives
    moop3 = MOOP(LocalGPS)
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
    assert(moop3.data['f_vals'].shape == (1, 2))
    assert(moop3.data['x_vals'].shape == (1, 5))
    assert(moop3.data['c_vals'].shape == (1, 3))
    assert(moop3.n_dat == 1)


def test_MOOP_iterate():
    """ Test the MOOP class's iterator in objectives.py.

    Initialize several MOOP objects and perform iterations to produce
    a batch of candidate solutions.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF, LocalGaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalGPS, TR_LBFGSB
    import numpy as np
    import pytest

    # Initialize two simulation groups with 1 output each
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF,
          'search_budget': 20,
          'sim_db': {'x_vals': [[0.0, 0.0, 0.0]],
                     's_vals': [[0.0]]}}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0)],
          'surrogate': GaussRBF,
          'search_budget': 20,
          'sim_db': {'x_vals': [[0.0, 0.0, 0.0]],
                     's_vals': [[np.sqrt(3.0)]]}}
    # Create a MOOP with 3 design variables and 2 simulations
    moop1 = MOOP(LocalGPS, hyperparams={'opt_budget': 100})
    with pytest.raises(AttributeError):
        moop1.iterate(1)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    with pytest.raises(AttributeError):
        moop1.iterate(1)
    # Now add the two objectives
    def f1(x, sim): return sim[0]
    def f2(x, sim): return sim[1]
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
    for (x, i) in batch:
        moop1.evaluateSimulation(x, i)
    moop1.updateAll(0, batch)
    batch = moop1.iterate(1)
    for (x, i) in batch:
        moop1.evaluateSimulation(x, i)
    moop1.updateAll(1, batch)
    soln = moop1.getPF()
    # Assert that solutions were found
    assert(np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    for i in range(np.shape(soln['x_vals'])[0]):
        assert(np.linalg.norm(np.asarray([g1['sim_func'](soln['x_vals'][i]),
                                          g2['sim_func'](soln['x_vals'][i])]
                                         ).flatten() - soln['f_vals'][i])
               < 0.00000001)

    g3 = {'m': 4,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: x[0:4],
          'surrogate': LocalGaussRBF,
          'search_budget': 500,
          'sim_db': {'x_vals': [[0.0, 0.0, 0.0, 0.0]],
                     's_vals': [[0.0, 0.0, 0.0, 0.0]]}}
    # Create a three objective toy problem, with one simulation
    moop2 = MOOP(TR_LBFGSB, hyperparams={'opt_budget': 100})
    for i in range(4):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0, 'des_tol': 0.1})
    moop2.addSimulation(g3)

    # Now add the three objectives
    def f3(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.asarray([2.0 * sim[0] - 0.2,
                               2.0 * sim[1],
                               2.0 * sim[2],
                               2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - 0.1 * np.eye(4)[0, :]) ** 2.0

    def f4(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.asarray([2.0 * sim[0],
                               2.0 * sim[1] - 0.2,
                               2.0 * sim[2],
                               2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - 0.1 * np.eye(4)[1, :]) ** 2.0

    def f5(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.asarray([2.0 * sim[0],
                               2.0 * sim[1],
                               2.0 * sim[2] - 0.2,
                               2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - 0.1 * np.eye(4)[2, :]) ** 2.0

    moop2.addObjective({'obj_func': f3},
                       {'obj_func': f4},
                       {'obj_func': f5})
    # Add 3 acquisition functions
    for i in range(3):
        moop2.addAcquisition({'acquisition': UniformWeights})
    # Do 2 iterates of the MOOP and extract the final database
    moop2.iterate(0)
    batch = [(0.1 * xi, 0) for xi in np.eye(4)]
    batch.append((0.1 * np.ones(4), 0))
    for (x, i) in batch:
        moop2.evaluateSimulation(x, i)
    moop2.updateAll(0, batch)
    batch = moop2.iterate(1)
    for (x, i) in batch:
        moop2.evaluateSimulation(x, i)
    moop2.updateAll(1, batch)
    soln = moop2.getPF()
    # Assert that solutions were found
    assert(np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(4)
    for i in range(np.shape(soln['x_vals'])[0]):
        sim = soln['x_vals'][i]
        assert(np.linalg.norm(np.asarray([f3(soln['x_vals'][i], sim),
                                          f4(soln['x_vals'][i], sim),
                                          f5(soln['x_vals'][i], sim)]
                                         ).flatten()
                              - soln['f_vals'][i])
               < 0.00000001)
        assert(all(soln['x_vals'][i, :4] <= 0.2))

    g4 = {'m': 4,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: x[0:4] + abs(x[4] - 1.0),
          'surrogate': LocalGaussRBF,
          'search_budget': 500,
          'sim_db': {'x_vals': [[0.0, 0.0, 0.0, 0.0, 0.0]],
                     's_vals': [[1.0, 1.0, 1.0, 1.0]]}}
    # Create a three objective toy problem, with one simulation
    moop3 = MOOP(TR_LBFGSB, hyperparams={})
    for i in range(4):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})
    moop3.addDesign({'des_type': "categorical", 'levels': 3})
    moop3.addSimulation(g4)

    # Now add the three objectives
    def f6(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.asarray([2.0 * sim[0] - 2.0,
                               2.0 * sim[1],
                               2.0 * sim[2],
                               2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - np.eye(4)[0, :]) ** 2.0

    def f7(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.asarray([2.0 * sim[0],
                               2.0 * sim[1] - 2.0,
                               2.0 * sim[2],
                               2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - np.eye(4)[1, :]) ** 2.0

    def f8(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.asarray([2.0 * sim[0],
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
    for (x, i) in batch:
        moop3.evaluateSimulation(x, i)
    moop3.updateAll(1, batch)
    soln = moop3.getPF()
    # Assert that solutions were found
    assert(np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(4)
    for i in range(np.shape(soln['x_vals'])[0]):
        sim = soln['x_vals'][i, :4] - abs(soln['x_vals'][i, 4] - 1.0)
        assert(np.linalg.norm(np.asarray([f6(soln['x_vals'][i], sim),
                                          f7(soln['x_vals'][i], sim),
                                          f8(soln['x_vals'][i], sim)]
                                         ).flatten()
                              - soln['f_vals'][i])
               < 0.00000001)
        assert(soln['x_vals'][i, 3] <= 0.1 and soln['x_vals'][i, 4] == 1.0)

    x_entry = np.zeros(1, dtype=np.dtype([("x0", float), ("x1", float),
                                          ("x2", object)]))
    x_entry[0]["x2"] = "0"
    g5 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [(x["x0"] - 1.0) * (x["x0"] - 1.0) +
                                 (x["x1"]) * (x["x1"]) + float(x["x2"])],
          'surrogate': LocalGaussRBF,
          'search_budget': 100,
          'sim_db': {'x_vals': x_entry,
                     's_vals': np.asarray([[1.0]])}}
    # Solve a MOOP with categorical variables
    moop4 = MOOP(TR_LBFGSB, hyperparams={})
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
    for (x, i) in batch:
        moop4.evaluateSimulation(x, i)
    moop4.updateAll(0, batch)
    batch = moop4.iterate(1)
    for (x, i) in batch:
        moop4.evaluateSimulation(x, i)
    moop4.updateAll(1, batch)
    soln = moop4.getPF()
    # Assert that solutions were found
    assert(soln.size > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(1)
    for i, xi in enumerate(soln):
        sim[0] = ((xi["x0"] - 1.0) * (xi["x0"] - 1.0) +
                  (xi["x1"]) * (xi["x1"]) + float(xi["x2"]))
        assert(f9(soln[i], sim) - soln['f1'][i] < 1.0e-8 and
               f10(soln[i], sim) - soln['f2'][i] < 1.0e-8)
        assert(xi["x2"] == "0")


def test_MOOP_solve():
    """ Test the MOOP class's solver in objectives.py.

    Perform a test of the MOOP solver class by minimizing a 5 variable,
    biobjective convex function s.t. $x in [0, 1]^n$.

    The correctness of the solutions is difficult to assert, but we can
    assert that the efficient points map onto the Pareto front, as
    expected.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights, RandomConstraint
    from parmoo.optimizers import LocalGPS, LBFGSB
    import numpy as np
    import pytest

    # Initialize two simulation groups with 1 output each
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF,
          'sim_db': {'x_vals': [[0.0, 0.0, 0.0, 0.0]],
                     's_vals': [[0.0]]}}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 25,
          'sim_func': lambda x: [np.linalg.norm(x-1.0)],
          'surrogate': GaussRBF,
          'sim_db': {'x_vals': [[0.0, 0.0, 0.0, 0.0]],
                     's_vals': [[2.0]]}}
    # Create a MOOP with 4 design variables and 2 simulations
    moop1 = MOOP(LocalGPS, hyperparams={'opt_budget': 100})
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
    with pytest.raises(ValueError):
        moop1.solve(2.0)
    # Solve the MOOP with 6 iterations
    moop1.solve(6)
    soln = moop1.data
    # Assert that solutions were found
    assert(np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    for i in range(np.shape(soln['x_vals'])[0]):
        assert(np.linalg.norm(np.asarray([g1['sim_func'](soln['x_vals'][i]),
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
    moop2 = MOOP(LocalGPS, hyperparams={})
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
    assert(np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    for i in range(np.shape(soln['x_vals'])[0]):
        assert(np.linalg.norm(np.asarray(g3['sim_func'](soln['x_vals'][i])) +
                              np.asarray(g4['sim_func'](soln['x_vals'][i])) -
                              soln['f_vals'][i]) < 0.00000001)

    # Create a 3 objective toy problem, with no simulations
    moop3 = MOOP(LBFGSB, hyperparams={})
    for i in range(4):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})

    # Now add the three objectives
    def f4(x, sim, der=0):
        if der == 1:
            return np.asarray([2.0 * x[0] - 2.0,
                               2.0 * x[1],
                               2.0 * x[2],
                               2.0 * x[3]])
        elif der == 2:
            return np.zeros(sim.size)
        else:
            return np.linalg.norm(x - np.eye(x.size)[0, :]) ** 2.0

    def f5(x, sim, der=0):
        if der == 1:
            return np.asarray([2.0 * x[0],
                               2.0 * x[1] - 2.0,
                               2.0 * x[2],
                               2.0 * x[3]])
        elif der == 2:
            return np.zeros(sim.size)
        else:
            return np.linalg.norm(x - np.eye(x.size)[1, :]) ** 2.0

    def f6(x, sim, der=0):
        if der == 1:
            return np.asarray([2.0 * x[0],
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
    assert(np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(0)
    for i in range(np.shape(soln['x_vals'])[0]):
        assert(np.linalg.norm(np.asarray([f4(soln['x_vals'][i], sim),
                                          f5(soln['x_vals'][i], sim),
                                          f6(soln['x_vals'][i], sim)]
                                         ).flatten()
                              - soln['f_vals'][i])
               < 0.00000001)

    # Create a 3 objective toy problem, with no simulations and 1 categorical
    moop4 = MOOP(LBFGSB, hyperparams={})
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
    assert(np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(0)
    for i in range(np.shape(soln['x_vals'])[0]):
        assert(np.linalg.norm(np.asarray([f4(soln['x_vals'][i], sim),
                                          f5(soln['x_vals'][i], sim),
                                          f6(soln['x_vals'][i], sim)]
                                         ).flatten()
                              - soln['f_vals'][i])
               < 0.00000001)


def test_getPF():
    """ Test the getPF function.

    Create several MOOPs, evaluate simulations, and check the final Pareto
    front for correctness.

    """

    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create a toy problem with 4 design variables
    moop = MOOP(LocalGPS, hyperparams={})
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
    moop.data['x_vals'][0, :] = np.asarray([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][1, :] = np.asarray([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop.evaluateSurrogates(
                                   np.asarray([1.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][2, :] = np.asarray([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 1.0, 0.0, 0.0]))
    moop.data['x_vals'][3, :] = np.asarray([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 1.0, 0.0]))
    moop.data['x_vals'][4, :] = np.asarray([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 0.0, 1.0]))
    moop.n_dat = 5
    soln = moop.getPF()
    assert(soln['f_vals'].shape == (4, 3))
    # Create a toy problem with 4 design variables
    moop = MOOP(LocalGPS, hyperparams={})
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
    moop.data['x_vals'][0, :] = np.asarray([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][0, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][1, :] = np.asarray([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop.evaluateSurrogates(
                                   np.asarray([1.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][1, :] = moop.evaluateConstraints(
                                   np.asarray([1.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][2, :] = np.asarray([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 1.0, 0.0, 0.0]))
    moop.data['c_vals'][2, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 1.0, 0.0, 0.0]))
    moop.data['x_vals'][3, :] = np.asarray([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 1.0, 0.0]))
    moop.data['c_vals'][3, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 0.0, 1.0, 0.0]))
    moop.data['x_vals'][4, :] = np.asarray([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 0.0, 1.0]))
    moop.data['c_vals'][4, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 0.0, 0.0, 1.0]))
    moop.n_dat = 5
    soln = moop.getPF()
    assert(soln.shape[0] == 4)


def test_getSimulationData():
    """ Test the getSimulationData function.

    Create several MOOPs, evaluate simulations, and check the simulation
    database.

    """
    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.optimizers import LocalGPS
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
    moop = MOOP(LocalGPS, hyperparams={})
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(g1, g2)
    soln = moop.getSimulationData()
    assert(soln[0]['s_vals'].size == 0)
    assert(soln[1]['s_vals'].size == 0)
    # Evaluate 5 simulations
    moop.evaluateSimulation(np.asarray([0.0, 0.0, 0.0, 0.0]), 0)
    moop.evaluateSimulation(np.asarray([0.0, 0.0, 0.0, 0.0]), 1)
    moop.evaluateSimulation(np.asarray([1.0, 0.0, 0.0, 0.0]), 0)
    moop.evaluateSimulation(np.asarray([1.0, 0.0, 0.0, 0.0]), 1)
    moop.evaluateSimulation(np.asarray([0.0, 1.0, 0.0, 0.0]), 0)
    moop.evaluateSimulation(np.asarray([0.0, 1.0, 0.0, 0.0]), 1)
    moop.evaluateSimulation(np.asarray([0.0, 0.0, 1.0, 0.0]), 0)
    moop.evaluateSimulation(np.asarray([0.0, 0.0, 1.0, 0.0]), 1)
    moop.evaluateSimulation(np.asarray([0.0, 0.0, 0.0, 1.0]), 0)
    moop.evaluateSimulation(np.asarray([0.0, 0.0, 0.0, 1.0]), 1)
    soln = moop.getSimulationData()
    assert(soln[0]['s_vals'].shape == (5, 1))
    assert(soln[1]['s_vals'].shape == (5, 2))
    # Create a toy problem with 4 design variables
    moop = MOOP(LocalGPS, hyperparams={})
    for i in range(4):
        moop.addDesign({'name': ("x" + str(i + 1)), 'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(g3, g4)
    soln = moop.getSimulationData()
    assert(soln['Bobo1']['out'].size == 0)
    assert(soln['Bobo2']['out'].size == 0)
    # Evaluate 5 simulations
    sample_x = np.zeros(1, dtype=moop.des_names)
    moop.evaluateSimulation(sample_x[0], 0)
    moop.evaluateSimulation(sample_x[0], 1)
    sample_x["x1"] = 1.0
    moop.evaluateSimulation(sample_x[0], 0)
    moop.evaluateSimulation(sample_x[0], 1)
    sample_x["x1"] = 0.0
    sample_x["x2"] = 1.0
    moop.evaluateSimulation(sample_x[0], 0)
    moop.evaluateSimulation(sample_x[0], 1)
    sample_x["x2"] = 0.0
    sample_x["x3"] = 1.0
    moop.evaluateSimulation(sample_x[0], 0)
    moop.evaluateSimulation(sample_x[0], 1)
    sample_x["x3"] = 0.0
    sample_x["x4"] = 1.0
    moop.evaluateSimulation(sample_x[0], 0)
    moop.evaluateSimulation(sample_x[0], 1)
    soln = moop.getSimulationData()
    assert(soln['Bobo1']['out'].shape == (5,))
    assert(soln['Bobo2']['out'].shape == (5, 2))


def test_getObjectiveData():
    """ Test the getObjectiveData function.

    Create several MOOPs, evaluate simulations, and check the objective
    database.

    """

    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create a toy problem with 4 design variables
    moop = MOOP(LocalGPS, hyperparams={})
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
    moop.data['x_vals'][0, :] = np.asarray([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][0, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][1, :] = np.asarray([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop.evaluateSurrogates(
                                   np.asarray([1.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][1, :] = moop.evaluateConstraints(
                                   np.asarray([1.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][2, :] = np.asarray([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 1.0, 0.0, 0.0]))
    moop.data['c_vals'][2, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 1.0, 0.0, 0.0]))
    moop.data['x_vals'][3, :] = np.asarray([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 1.0, 0.0]))
    moop.data['c_vals'][3, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 0.0, 1.0, 0.0]))
    moop.data['x_vals'][4, :] = np.asarray([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 0.0, 1.0]))
    moop.data['c_vals'][4, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 0.0, 0.0, 1.0]))
    moop.n_dat = 5
    soln = moop.getObjectiveData()
    assert(soln['f_vals'].shape == (5, 3))
    # Create a toy problem with 4 design variables
    moop = MOOP(LocalGPS, hyperparams={})
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
    moop.data['x_vals'][0, :] = np.asarray([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][0, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][1, :] = np.asarray([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop.evaluateSurrogates(
                                   np.asarray([1.0, 0.0, 0.0, 0.0]))
    moop.data['c_vals'][1, :] = moop.evaluateConstraints(
                                   np.asarray([1.0, 0.0, 0.0, 0.0]))
    moop.data['x_vals'][2, :] = np.asarray([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 1.0, 0.0, 0.0]))
    moop.data['c_vals'][2, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 1.0, 0.0, 0.0]))
    moop.data['x_vals'][3, :] = np.asarray([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 1.0, 0.0]))
    moop.data['c_vals'][3, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 0.0, 1.0, 0.0]))
    moop.data['x_vals'][4, :] = np.asarray([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop.evaluateSurrogates(
                                   np.asarray([0.0, 0.0, 0.0, 1.0]))
    moop.data['c_vals'][4, :] = moop.evaluateConstraints(
                                   np.asarray([0.0, 0.0, 0.0, 1.0]))
    moop.n_dat = 5
    soln = moop.getObjectiveData()
    assert(soln.shape[0] == 5)


if __name__ == "__main__":
    test_MOOP_init()
    test_MOOP_addDesign()
    test_MOOP_embed_extract()
    test_MOOP_addSimulation()
    test_pack_unpack_sim()
    test_MOOP_addObjective()
    test_MOOP_addConstraint()
    test_MOOP_addAcquisition()
    test_MOOP_evaluateSimulation()
    test_MOOP_evaluateSurrogates()
    test_MOOP_evaluateConstraints()
    test_MOOP_evaluateLagrangian()
    test_MOOP_evaluateGradients()
    test_MOOP_addData()
    test_MOOP_iterate()
    test_MOOP_solve()
    test_getPF()
    test_getSimulationData()
    test_getObjectiveData()
