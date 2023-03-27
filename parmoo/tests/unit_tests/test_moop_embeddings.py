
def test_MOOP_addDesign_bad_cont():
    """ Check that the MOOP class handles adding bad continuous variables.

    Initialize a MOOP objects, and add several bad continuous design
    variables.

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
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addDesign({'name': "x_1", 'lb': 0.0, 'ub': 1.0})
        moop.addDesign({'name': "x_1",
                        'des_type': "continuous",
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


def test_MOOP_addDesign_bad_cat():
    """ Check that the MOOP class handles adding bad categorical variables.

    Initialize a MOOP objects, and add several bad categorical design
    variables.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalGPS
    import pytest

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalGPS)
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


def test_MOOP_addDesign_bad_int():
    """ Check that the MOOP class handles adding bad integer variables.

    Initialize a MOOP objects, and add several bad integer design
    variables.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalGPS
    import pytest

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalGPS)
    # Add some bad integer variables
    with pytest.raises(AttributeError):
        moop.addDesign({'des_type': "integer"})
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': "integer",
                        'des_tol': "hello world",
                        'lb': 0.0,
                        'ub': 1.0})
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': "integer",
                        'lb': "hello",
                        'ub': "world"})
    with pytest.raises(ValueError):
        moop.addDesign({'des_type': "integer",
                        'lb': 0,
                        'ub': 0})
    with pytest.raises(TypeError):
        moop.addDesign({'name': 5,
                        'des_type': "integer",
                        'lb': 0,
                        'ub': 1})


def test_MOOP_addDesign():
    """ Check that the MOOP class handles adding design variables properly.

    Initialize a MOOP objects, and add several design variables.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalGPS

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalGPS)
    # Now add some continuous and integer design variables
    assert (moop.n == 0)
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n == 1)
    moop.addDesign({'des_type': "integer",
                    'lb': 0,
                    'ub': 4})
    assert (moop.n == 2)
    assert (moop.n_int == 1)
    moop.addDesign({'name': "x_3",
                    'des_type': "continuous",
                    'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n == 3)
    moop.addDesign({'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n == 4)
    moop.addDesign({'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n == 5)
    moop.addDesign({'name': "x_6",
                    'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n == 6)
    # Now add some categorical design variables
    assert (moop.n_cat == 0)
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    assert (moop.n_cat == 1)
    moop.addDesign({'des_type': "categorical",
                    'levels': 3})
    assert (moop.n_cat == 2)
    moop.addDesign({'name': "x_9",
                    'des_type': "categorical",
                    'levels': 3})
    assert (moop.n_cat == 3)
    moop.addDesign({'name': "x_10",
                    'des_type': "categorical",
                    'levels': ["boy", "girl", "doggo"]})
    assert (moop.n_cat == 4)
    # Now add a custom design variables
    moop.addDesign({'des_type': "custom",
                    'embedding_size': 1,
                    'embedder': lambda x: x,
                    'extracter': lambda x: x})
    assert (moop.n_custom == 1)
    moop.addDesign({'des_type': "raw"})
    assert (moop.n_raw == 1)
    # Now add more continuous design variables
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n_cont == 6)
    moop.addDesign({'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n_cont == 7)
    # Check the design order
    right_order = [0, 7, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 5, 6]
    assert (all([moop.des_order[i] == right_order[i] for i in range(14)]))


def test_MOOP_embed_extract_unnamed1():
    """ Test that the MOOP class can embed/extract unnamed design variables.

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
    # Add integer and continuous vars and check that they embed correctly
    moop.addDesign({'des_type': "integer",
                    'lb': 0,
                    'ub': 1000})
    moop.addDesign({'lb': -1.0,
                    'ub': 0.0})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(2)
        xi[0] *= 1000.0
        xi[1] -= 1.0
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (moop.__extract__(xxi)[0] - xi[0] <= 0.5)
        assert (moop.__extract__(xxi)[1] - xi[1] < 1.0e-8)
    # Test upper and lower bounds
    x0 = np.zeros(2)
    x0[0] *= 1000.0
    x0[1] -= 1.0
    xx0 = moop.__embed__(x0)
    # Check that embedding is legal
    assert (all(xx0 >= 0.0) and all(xx0 <= 1.0))
    assert (xx0.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx0) - x0 < 1.0e-8))
    x1 = np.ones(2)
    x1[0] *= 1000.0
    x1[1] -= 1.0
    xx1 = moop.__embed__(x1)
    # Check that embedding is legal
    assert (all(xx1 >= 0.0) and all(xx1 <= 1.0))
    assert (xx1.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx1) - x1 < 1.0e-8))
    # Add two categorical variables and check that they embed correctly
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    moop.addDesign({'des_type': "categorical",
                    'levels': 3})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(4)
        xi[0] *= 1000
        xi[0] = int(xi[0])
        xi[1] -= 1.0
        xi[2:] = np.round(xi[2:])
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all(moop.__extract__(xxi) - xi < 1.0e-8))
    # Test upper and lower bounds
    x0 = np.zeros(4)
    x0[0] *= 1000.0
    x0[1] -= 1.0
    x0[2:] = np.round(x0[2:])
    xx0 = moop.__embed__(x0)
    # Check that embedding is legal
    assert (all(xx0 >= 0.0) and all(xx0 <= 1.0))
    assert (xx0.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx0) - x0 < 1.0e-8))
    x1 = np.ones(4)
    x1[0] *= 1000.0
    x1[1] -= 1.0
    x1[2:] = np.round(x1[2:])
    xx1 = moop.__embed__(x1)
    # Check that embedding is legal
    assert (all(xx1 >= 0.0) and all(xx1 <= 1.0))
    assert (xx1.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx1) - x1 < 1.0e-8))
    # Add a custom variable and raw variable and check that they embed
    moop.addDesign({'des_type': "custom",
                    'embedding_size': 1,
                    'embedder': lambda x: x,
                    'extracter': lambda x: x})
    moop.addDesign({'des_type': "raw"})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(6)
        xi[0] *= 1000
        xi[0] = int(xi[0])
        xi[1] -= 1.0
        xi[2:4] = np.round(xi[2:4])
        xi[5] *= 5.0
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi[:5] >= 0.0) and all(xxi[:5] <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all(moop.__extract__(xxi) - xi < 1.0e-8))
    # Test upper and lower bounds
    x0 = np.zeros(6)
    x0[0] *= 1000.0
    x0[1] -= 1.0
    x0[2:] = np.round(x0[2:])
    x0[2:4] = np.round(xi[2:4])
    x0[5] *= 5.0
    xx0 = moop.__embed__(x0)
    # Check that embedding is legal
    assert (all(xx0 >= 0.0) and all(xx0 <= 1.0))
    assert (xx0.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx0) - x0 < 1.0e-8))
    x1 = np.ones(6)
    x1[0] *= 1000.0
    x1[1] -= 1.0
    x1[2:] = np.round(x1[2:])
    x1[2:4] = np.round(xi[2:4])
    x1[5] *= 5.0
    xx1 = moop.__embed__(x1)
    # Check that embedding is legal
    assert (all(xx1[:5] >= 0.0) and all(xx1[:5] <= 1.0))
    assert (xx1.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx1) - x1 < 1.0e-8))


def test_MOOP_embed_extract_unnamed2():
    """ Test that the MOOP class can embed/extract unnamed design variables.

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

    # Same problem as in test_MOOP_embed_extract_unnamed1(), but reverse order
    moop = MOOP(LocalGPS)
    # Add a custom variable and raw variable and check that they embed
    moop.addDesign({'des_type': "raw"})
    moop.addDesign({'des_type': "custom",
                    'embedding_size': 1,
                    'embedder': lambda x: x,
                    'extracter': lambda x: x})
    # Add two categorical variables and check that they are embedded correctly
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    moop.addDesign({'des_type': "categorical",
                    'levels': 3})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(4)
        xi[0] *= 0.5
        xi[2:] = np.round(xi[2:])
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all(moop.__extract__(xxi) - xi < 1.0e-8))
    # Test upper and lower bounds
    x0 = np.zeros(4)
    x0[0] *= 0.5
    x0[2:] = np.round(x0[2:])
    xx0 = moop.__embed__(x0)
    # Check that embedding is legal
    assert (all(xx0 >= 0.0) and all(xx0 <= 1.0))
    assert (xx0.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx0) - x0 < 1.0e-8))
    x1 = np.ones(4)
    x1[0] *= 0.5
    x1[2:] = np.round(x1[2:])
    xx1 = moop.__embed__(x1)
    # Check that embedding is legal
    assert (all(xx1 >= 0.0) and all(xx1 <= 1.0))
    assert (xx1.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx1) - x1 < 1.0e-8))
    # Add two continuous variables and check that they are embedded correctly
    moop.addDesign({'lb': -1.0,
                    'ub': 0.0})
    moop.addDesign({'des_type': "integer",
                    'lb': 0,
                    'ub': 1000})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(6)
        xi[0] *= 0.5
        xi[2:4] = np.round(xi[2:4])
        xi[4] -= 1.0
        xi[5] *= 1000.0
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all(moop.__extract__(xxi) - xi < 1.0e-8))
    # Test upper and lower bounds
    x0 = np.zeros(6)
    x0[0] *= 0.5
    x0[2:4] = np.round(x0[:2])
    x0[4] -= 1.0
    x0[5] *= 1000.0
    xx0 = moop.__embed__(x0)
    # Check that embedding is legal
    assert (all(xx0 >= 0.0) and all(xx0 <= 1.0))
    assert (xx0.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx0) - x0 < 1.0e-8))
    x1 = np.ones(6)
    x1[0] *= 0.5
    x1[2:4] = np.round(x1[2:4])
    x1[4] -= 1.0
    x1[5] *= 1000.0
    xx1 = moop.__embed__(x1)
    # Check that embedding is legal
    assert (all(xx1 >= 0.0) and all(xx1 <= 1.0))
    assert (xx1.size == moop.n)
    # Check extraction
    assert (all(moop.__extract__(xx1) - x1 < 1.0e-8))


def test_MOOP_embed_extract_named1():
    """ Test that the MOOP class can embed/extract named design variables.

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

    # Now, create another MOOP where all variables are labeled
    moop = MOOP(LocalGPS)
    # Add two continuous variables and check that they are embedded correctly
    moop.addDesign({'name': "x0",
                    'des_type': "integer",
                    'lb': 0,
                    'ub': 1000})
    moop.addDesign({'name': "x1",
                    'lb': -1.0,
                    'ub': 0.0})
    # Test 5 random variables
    for i in range(5):
        nums = np.random.random_sample(2)
        xi = np.zeros(1, dtype=[("x0", float), ("x1", float)])
        xi["x0"] = int(1000.0 * nums[0])
        xi["x1"] = nums[1] - 1.0
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all([abs(moop.__extract__(xxi)[key] - xi[key]) < 1.0e-8
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
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all([abs(moop.__extract__(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2"]]))
        assert (moop.__extract__(xxi)["x3"] == xi["x3"])
    # Add an integer variables and check that it is embedded correctly
    moop.addDesign({'name': "x4",
                    'des_type': "int",
                    'lb': -5,
                    'ub': 5})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(5)
        xi = np.zeros(1, dtype=[("x0", float), ("x1", float), ("x2", float),
                                ("x3", object), ("x4", int)])[0]
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all([abs(moop.__extract__(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4"]]))
        assert (moop.__extract__(xxi)["x3"] == xi["x3"])
    # Add a custom variable and check that it is embedded correctly
    moop.addDesign({'name': "x5",
                    'des_type': "custom",
                    'embedding_size': 1,
                    'embedder': lambda x: float(x),
                    'extracter': lambda x: str(x)})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(6)
        xi = np.zeros(1, dtype=[("x0", float), ("x1", float), ("x2", float),
                                ("x3", object), ("x4", int), ("x5", "U5")])[0]
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xi["x5"] = str(num[5])
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all([abs(moop.__extract__(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4"]]))
        assert (moop.__extract__(xxi)["x3"] == xi["x3"])
        assert (abs(float(moop.__extract__(xxi)["x5"]) - float(xi["x5"]))
                < 1.0e-8)
    # Add a raw variable
    moop.addDesign({'name': "x6",
                    'des_type': "raw"})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(7)
        xi = np.zeros(1, dtype=[("x0", float), ("x1", float), ("x2", float),
                                ("x3", object), ("x4", int), ("x5", "U5"),
                                ("x6", float)])[0]
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xi["x5"] = str(num[5])
        xi["x6"] = num[6]
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all([abs(moop.__extract__(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4", "x6"]]))
        assert (moop.__extract__(xxi)["x3"] == xi["x3"])
        assert (abs(float(moop.__extract__(xxi)["x5"]) - float(xi["x5"]))
                < 1.0e-8)
    # Add another custom variable and check that it is embedded correctly
    moop.addDesign({'name': "x7",
                    'des_type': "custom",
                    'embedding_size': 3,
                    'embedder': lambda x: [float(x[0]), float(x[1]),
                                           float(x[2])],
                    'extracter': lambda x: (str(int(x[0])) + str(int(x[1])) +
                                            str(int(x[2])))})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(8)
        xi = np.zeros(1, dtype=[("x0", float), ("x1", float), ("x2", float),
                                ("x3", object), ("x4", int), ("x5", "U5"),
                                ("x6", float), ("x7", "U5")])[0]
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xi["x5"] = str(num[5])
        xi["x6"] = num[6]
        xi["x7"] = "010"
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all([abs(moop.__extract__(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4", "x6"]]))
        assert (moop.__extract__(xxi)["x3"] == xi["x3"])
        assert (abs(float(moop.__extract__(xxi)["x5"]) - float(xi["x5"]))
                < 1.0e-8)
        assert (moop.__extract__(xxi)["x7"] == xi["x7"])


def test_MOOP_embed_extract_named2():
    """ Test that the MOOP class can embed/extract named design variables.

    Add several design variables and generate an embedding. Then embed and
    extract several inputs, and check that the results match up to the
    design tolerance. This test applies to the three hidden methods:
     * MOOP.__embed__(x)
     * MOOP.__extract__(x)
     * MOOP.__generate_encoding__()

    Define the same problem as above, but add variables in reverse order.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Now, create another MOOP where all variables are labeled
    moop = MOOP(LocalGPS)
    # Add a custom variable
    moop.addDesign({'name': "x7",
                    'des_type': "custom",
                    'embedding_size': 3,
                    'embedder': lambda x: [float(x[0]), float(x[1]),
                                           float(x[2])],
                    'extracter': lambda x: (str(int(x[0])) + str(int(x[1])) +
                                            str(int(x[2])))})
    # Add a raw variable
    moop.addDesign({'name': "x6",
                    'des_type': "raw"})
    # Add a custom variable
    moop.addDesign({'name': "x5",
                    'des_type': "custom",
                    'dtype': "U40",
                    'embedding_size': 1,
                    'embedder': lambda x: float(x),
                    'extracter': lambda x: str(x)})
    # Add an integer variable
    moop.addDesign({'name': "x4",
                    'des_type': "int",
                    'lb': -5,
                    'ub': 5})
    # Add two categorical variables
    moop.addDesign({'name': "x3",
                    'des_type': "categorical",
                    'levels': ["biggie", "shortie", "shmedium"]})
    moop.addDesign({'name': "x2",
                    'des_type': "categorical",
                    'levels': 2})
    # Add two continuous variables and check that they are embedded correctly
    moop.addDesign({'name': "x1",
                    'lb': -1.0,
                    'ub': 0.0})
    moop.addDesign({'name': "x0",
                    'des_type': "integer",
                    'lb': 0,
                    'ub': 1000})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(8)
        xi = np.zeros(1, dtype=[("x0", float), ("x1", float), ("x2", float),
                                ("x3", object), ("x4", int), ("x5", "U5"),
                                ("x6", float), ("x7", "U5")])[0]
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xi["x5"] = str(num[5])
        xi["x6"] = num[6]
        xi["x7"] = "010"
        xxi = moop.__embed__(xi)
        # Check that embedding is legal
        assert (all(xxi >= 0.0) and all(xxi <= 1.0))
        assert (xxi.size == moop.n)
        # Check extraction
        assert (all([abs(moop.__extract__(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4", "x6"]]))
        assert (moop.__extract__(xxi)["x3"] == xi["x3"])
        assert (abs(float(moop.__extract__(xxi)["x5"]) - float(xi["x5"]))
                < 1.0e-8)
        assert (moop.__extract__(xxi)["x7"] == xi["x7"])


if __name__ == "__main__":
    test_MOOP_addDesign_bad_cont()
    test_MOOP_addDesign_bad_cat()
    test_MOOP_addDesign_bad_int()
    test_MOOP_addDesign()
    test_MOOP_embed_extract_unnamed1()
    test_MOOP_embed_extract_unnamed2()
    test_MOOP_embed_extract_named1()
    test_MOOP_embed_extract_named2()
