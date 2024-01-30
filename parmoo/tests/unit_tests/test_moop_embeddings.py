
def test_MOOP_addDesign_bad_cont():
    """ Check that the MOOP class handles adding bad continuous variables.

    Initialize a MOOP objects, and add several bad continuous design
    variables.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    import pytest

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalSurrogate_PS)
    # Try to add some bad design variable types
    with pytest.raises(TypeError):
        moop.addDesign([])
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': 1.0})
    with pytest.raises(ValueError):
        moop.addDesign({'des_type': "hello world"})
    # Add some bad continuous variables
    with pytest.raises(KeyError):
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
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addDesign({'name': "x_1", 'lb': 0.0, 'ub': 1.0})
        moop.addDesign({'name': "x_1",
                        'des_type': "continuous",
                        'lb': 0.0,
                        'ub': 1.0})
    # Add variables out of order
    with pytest.raises(RuntimeError):
        moop1 = MOOP(LocalSurrogate_PS)
        moop1.acquisitions.append(0)
        moop1.addDesign({'des_type': "continuous",
                         'lb': 0.0,
                         'ub': 1.0})
    with pytest.raises(RuntimeError):
        moop2 = MOOP(LocalSurrogate_PS)
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
    from parmoo.optimizers import LocalSurrogate_PS
    import pytest

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalSurrogate_PS)
    # Add some bad categorical variables
    with pytest.raises(KeyError):
        moop.addDesign({'des_type': "categorical"})
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': "categorical",
                        'levels': 1.0})
    with pytest.raises(ValueError):
        moop.addDesign({'des_type': "categorical",
                        'levels': 1})
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
    from parmoo.optimizers import LocalSurrogate_PS
    import pytest

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalSurrogate_PS)
    # Add some bad integer variables
    with pytest.raises(KeyError):
        moop.addDesign({'des_type': "integer"})
    with pytest.raises(TypeError):
        moop.addDesign({'des_type': "integer",
                        'lb': "hello",
                        'ub': "world"})
    with pytest.raises(ValueError):
        moop.addDesign({'des_type': "integer",
                        'lb': 0,
                        'ub': 0})


def test_MOOP_addDesign():
    """ Check that the MOOP class handles adding design variables properly.

    Initialize a MOOP objects, and add several design variables.

    """

    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.embeddings import IdentityEmbedder

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalSurrogate_PS)
    # Now add some continuous and integer design variables
    assert (moop.n_latent == 0)
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n_latent == 1)
    moop.addDesign({'des_type': "integer",
                    'lb': 0,
                    'ub': 4})
    assert (moop.n_latent == 2)
    moop.addDesign({'name': "x_3",
                    'des_type': "continuous",
                    'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n_latent == 3)
    moop.addDesign({'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n_latent == 4)
    moop.addDesign({'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n_latent == 5)
    moop.addDesign({'name': "x_6",
                    'des_tol': 0.01,
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n_latent == 6)
    # Now add some categorical design variables
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    assert (moop.n_latent == 7)
    moop.addDesign({'des_type': "categorical",
                    'levels': 3})
    assert (moop.n_latent == 9)
    moop.addDesign({'name': "x_9",
                    'des_type': "categorical",
                    'levels': 3})
    assert (moop.n_latent == 11)
    moop.addDesign({'name': "x_10",
                    'des_type': "categorical",
                    'levels': ["boy", "girl", "doggo"]})
    assert (moop.n_latent == 13)
    # Now add a custom design variables
    moop.addDesign({'des_type': "custom",
                    'lb': -100.0,
                    'ub': 100.0,
                    'embedder': IdentityEmbedder})
    assert (moop.n_latent == 14)
    moop.addDesign({'des_type': "raw",
                    'lb': -100.0,
                    'ub': 100.0})
    assert (moop.n_latent == 15)
    # Now add more continuous design variables
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n_latent == 16)
    moop.addDesign({'lb': 0.0,
                    'ub': 1.0})
    assert (moop.n_latent == 17)


def test_MOOP_embed_extract_unnamed1():
    """ Test that the MOOP class can embed/extract unnamed design variables.

    Add several design variables and generate an embedding. Then embed and
    extract several inputs, and check that the results match up to the
    design tolerance. This test applies to the three hidden methods:
     * MOOP._embed(x)
     * MOOP._extract(x)

    """

    from jax import config
    config.update("jax_enable_x64", True)
    from parmoo import MOOP
    from parmoo.embeddings import IdentityEmbedder
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np

    # Initialize a MOOP with no hyperparameters
    moop = MOOP(LocalSurrogate_PS)
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
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (abs(moop._extract(xxi)[0] - xi[0]) <= 0.5)
        assert (abs(moop._extract(xxi)[1] - xi[1]) < 1.0e-8)
    # Test upper and lower bounds
    x0 = np.zeros(2)
    x0[0] *= 1000.0
    x0[1] -= 1.0
    xx0 = moop._embed(x0)
    # Check that embedding is legal
    assert (all(xx0 >= -1.0e-8) and all(xx0 <= 1 + 1.0e-8))
    assert (xx0.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx0)[0] - x0[0]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[1] - x0[1]) < 1.0e-8)
    x1 = np.ones(2)
    x1[0] *= 1000.0
    x1[1] -= 1.0
    xx1 = moop._embed(x1)
    # Check that embedding is legal
    assert (all(xx1 >= -1.0e-8) and all(xx1 <= 1 + 1.0e-8))
    assert (xx1.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx1)[0] - x1[0]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[1] - x1[1]) < 1.0e-8)
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
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (abs(moop._extract(xxi)[0] - xi[0]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[1] - xi[1]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[2] - xi[2]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[3] - xi[3]) < 1.0e-8)
    # Test upper and lower bounds
    x0 = np.zeros(4)
    x0[0] *= 1000.0
    x0[1] -= 1.0
    x0[2:] = np.round(x0[2:])
    xx0 = moop._embed(x0)
    # Check that embedding is legal
    assert (all(xx0 >= -1.0e-8) and all(xx0 <= 1 + 1.0e-8))
    assert (xx0.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx0)[0] - x0[0]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[1] - x0[1]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[2] - x0[2]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[3] - x0[3]) < 1.0e-8)
    x1 = np.ones(4)
    x1[0] *= 1000.0
    x1[1] -= 1.0
    x1[2:] = np.round(x1[2:])
    xx1 = moop._embed(x1)
    # Check that embedding is legal
    assert (all(xx1 >= -1.0e-8) and all(xx1 <= 1 + 1.0e-8))
    assert (xx1.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx1)[0] - x1[0]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[1] - x1[1]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[2] - x1[2]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[3] - x1[3]) < 1.0e-8)
    # Add a custom variable and raw variable and check that they embed
    moop.addDesign({'des_type': "custom",
                    'embedder': IdentityEmbedder,
                    'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'des_type': "raw",
                    'lb': 0.0, 'ub': 1.0})
    # Test 5 random variables
    for i in range(5):
        xi = np.random.random_sample(6)
        xi[0] *= 1000
        xi[0] = int(xi[0])
        xi[1] -= 1.0
        xi[2:4] = np.round(xi[2:4])
        xi[5] *= 5.0
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi[:5] >= -1.0e-8) and all(xxi[:5] <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (abs(moop._extract(xxi)[0] - xi[0]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[1] - xi[1]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[2] - xi[2]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[3] - xi[3]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[4] - xi[4]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[5] - xi[5]) < 1.0e-8)
    # Test upper and lower bounds
    x0 = np.zeros(6)
    x0[0] *= 1000.0
    x0[1] -= 1.0
    x0[2:] = np.round(x0[2:])
    x0[2:4] = np.round(xi[2:4])
    x0[5] *= 5.0
    xx0 = moop._embed(x0)
    # Check that embedding is legal
    assert (all(xx0 >= -1.0e-8) and all(xx0 <= 1 + 1.0e-8))
    assert (xx0.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx0)[0] - x0[0]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[1] - x0[1]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[2] - x0[2]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[3] - x0[3]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[4] - x0[4]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[5] - x0[5]) < 1.0e-8)
    x1 = np.ones(6)
    x1[0] *= 1000.0
    x1[1] -= 1.0
    x1[2:] = np.round(x1[2:])
    x1[2:4] = np.round(xi[2:4])
    x1[5] *= 5.0
    xx1 = moop._embed(x1)
    # Check that embedding is legal
    assert (all(xx1[:5] >= -1.0e-8) and all(xx1[:5] <= 1 + 1.0e-8))
    assert (xx1.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx1)[0] - x1[0]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[1] - x1[1]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[2] - x1[2]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[3] - x1[3]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[4] - x1[4]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[5] - x1[5]) < 1.0e-8)


def test_MOOP_embed_extract_unnamed2():
    """ Test that the MOOP class can embed/extract unnamed design variables.

    Add several design variables and generate an embedding. Then embed and
    extract several inputs, and check that the results match up to the
    design tolerance. This test applies to the three hidden methods:
     * MOOP._embed(x)
     * MOOP._extract(x)

    """

    from jax import config
    config.update("jax_enable_x64", True)
    from parmoo import MOOP
    from parmoo.embeddings import IdentityEmbedder
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np

    # Same problem as in test_MOOP_embed_extract_unnamed1(), but reverse order
    moop = MOOP(LocalSurrogate_PS)
    # Add a custom variable and raw variable and check that they embed
    moop.addDesign({'des_type': "raw",
                    'lb': 0.0,
                    'ub': 1.0})
    moop.addDesign({'des_type': "custom",
                    'embedder': IdentityEmbedder,
                    'lb': 0.0, 'ub': 1.0})
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
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (abs(moop._extract(xxi)[0] - xi[0]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[1] - xi[1]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[2] - xi[2]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[3] - xi[3]) < 1.0e-8)
    # Test upper and lower bounds
    x0 = np.zeros(4)
    x0[0] *= 0.5
    x0[2:] = np.round(x0[2:])
    xx0 = moop._embed(x0)
    # Check that embedding is legal
    assert (all(xx0 >= -1.0e-8) and all(xx0 <= 1 + 1.0e-8))
    assert (xx0.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx0)[0] - x0[0]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[1] - x0[1]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[2] - x0[2]) < 1.0e-8)
    assert (abs(moop._extract(xx0)[3] - x0[3]) < 1.0e-8)
    x1 = np.ones(4)
    x1[0] *= 0.5
    x1[2:] = np.round(x1[2:])
    xx1 = moop._embed(x1)
    # Check that embedding is legal
    assert (all(xx1 >= -1.0e-8) and all(xx1 <= 1 + 1.0e-8))
    assert (xx1.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx1)[0] - x1[0]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[1] - x1[1]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[2] - x1[2]) < 1.0e-8)
    assert (abs(moop._extract(xx1)[3] - x1[3]) < 1.0e-8)
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
        xi[5] = round(xi[5])
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (abs(moop._extract(xxi)[0] - xi[0]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[1] - xi[1]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[2] - xi[2]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[3] - xi[3]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[4] - xi[4]) < 1.0e-8)
        assert (abs(moop._extract(xxi)[5] - xi[5]) < 1.0e-8)
    # Test upper and lower bounds
    x0 = np.zeros(6)
    x0[0] *= 0.5
    x0[2:4] = np.round(x0[:2])
    x0[4] -= 1.0
    x0[5] *= 1000.0
    x0[5] = round(x0[5])
    xx0 = moop._embed(x0)
    # Check that embedding is legal
    assert (all(xx0 >= -1.0e-8) and all(xx0 <= 1 + 1.0e-8))
    assert (xx0.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx0)[0] - x0[0] < 1.0e-8))
    assert (abs(moop._extract(xx0)[1] - x0[1] < 1.0e-8))
    assert (abs(moop._extract(xx0)[2] - x0[2] < 1.0e-8))
    assert (abs(moop._extract(xx0)[3] - x0[3] < 1.0e-8))
    assert (abs(moop._extract(xx0)[4] - x0[4] < 1.0e-8))
    assert (abs(moop._extract(xx0)[5] - x0[5] < 1.0e-8))
    x1 = np.ones(6)
    x1[0] *= 0.5
    x1[2:4] = np.round(x1[2:4])
    x1[4] -= 1.0
    x1[5] *= 1000.0
    x1[5] = round(x1[5])
    xx1 = moop._embed(x1)
    # Check that embedding is legal
    assert (all(xx1 >= -1.0e-8) and all(xx1 <= 1 + 1.0e-8))
    assert (xx1.size == moop.n_latent)
    # Check extraction
    assert (abs(moop._extract(xx1)[0] - x1[0] < 1.0e-8))
    assert (abs(moop._extract(xx1)[1] - x1[1] < 1.0e-8))
    assert (abs(moop._extract(xx1)[2] - x1[2] < 1.0e-8))
    assert (abs(moop._extract(xx1)[3] - x1[3] < 1.0e-8))
    assert (abs(moop._extract(xx1)[4] - x1[4] < 1.0e-8))
    assert (abs(moop._extract(xx1)[5] - x1[5] < 1.0e-8))


def test_MOOP_embed_extract_named1():
    """ Test that the MOOP class can embed/extract named design variables.

    Add several design variables and generate an embedding. Then embed and
    extract several inputs, and check that the results match up to the
    design tolerance. This test applies to the three hidden methods:
     * MOOP._embed(x)
     * MOOP._extract(x)

    """

    from jax import config
    config.update("jax_enable_x64", True)
    from parmoo import MOOP
    from parmoo.embeddings import IdentityEmbedder
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np

    # Now, create another MOOP where all variables are labeled
    moop = MOOP(LocalSurrogate_PS)
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
        xi = {}
        xi["x0"] = int(1000.0 * nums[0])
        xi["x1"] = nums[1] - 1.0
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (all([abs(moop._extract(xxi)[key] - xi[key]) < 1.0e-8
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
        xi = {}
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (all([abs(moop._extract(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2"]]))
        assert (moop._extract(xxi)["x3"] == xi["x3"])
    # Add an integer variables and check that it is embedded correctly
    moop.addDesign({'name': "x4",
                    'des_type': "int",
                    'lb': -5,
                    'ub': 5})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(5)
        xi = {}
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (all([abs(moop._extract(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4"]]))
        assert (moop._extract(xxi)["x3"] == xi["x3"])
    # Add a custom variable and check that it is embedded correctly
    moop.addDesign({'name': "x5",
                    'des_type': "custom",
                    'embedder': IdentityEmbedder,
                    'lb': 0.0, 'ub': 1.0})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(6)
        xi = {}
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xi["x5"] = num[5]
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (all([abs(moop._extract(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4"]]))
        assert (moop._extract(xxi)["x3"] == xi["x3"])
        assert (abs(float(moop._extract(xxi)["x5"]) - float(xi["x5"]))
                < 1.0e-8)
    # Add a raw variable
    moop.addDesign({'name': "x6",
                    'des_type': "raw",
                    'lb': 0.0, 'ub': 1.0})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(7)
        xi = {}
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xi["x5"] = num[5]
        xi["x6"] = num[6]
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (all([abs(moop._extract(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4", "x6"]]))
        assert (moop._extract(xxi)["x3"] == xi["x3"])
        assert (abs(float(moop._extract(xxi)["x5"]) - float(xi["x5"]))
                < 1.0e-8)
    # Add another custom variable and check that it is embedded correctly
    moop.addDesign({'name': "x7",
                    'des_type': "custom",
                    'embedder': IdentityEmbedder,
                    'lb': 0.0, 'ub': 1.0})
    # Test 5 random variables
    for i in range(5):
        num = np.random.random_sample(8)
        xi = {}
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xi["x5"] = num[5]
        xi["x6"] = num[6]
        xi["x7"] = 0
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (all([abs(moop._extract(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4", "x6"]]))
        assert (moop._extract(xxi)["x3"] == xi["x3"])
        assert (abs(float(moop._extract(xxi)["x5"]) - float(xi["x5"]))
                < 1.0e-8)
        assert (moop._extract(xxi)["x7"] == xi["x7"])


def test_MOOP_embed_extract_named2():
    """ Test that the MOOP class can embed/extract named design variables.

    Add several design variables and generate an embedding. Then embed and
    extract several inputs, and check that the results match up to the
    design tolerance. This test applies to the three hidden methods:
     * MOOP._embed(x)
     * MOOP._extract(x)

    Define the same problem as above, but add variables in reverse order.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    from parmoo import MOOP
    from parmoo.embeddings import IdentityEmbedder
    from parmoo.optimizers import LocalSurrogate_PS
    import numpy as np

    # Now, create another MOOP where all variables are labeled
    moop = MOOP(LocalSurrogate_PS)
    # Add a custom variable
    moop.addDesign({'name': "x7",
                    'des_type': "custom",
                    'embedder': IdentityEmbedder,
                    'lb': 0.0, 'ub': 1.0})
    # Add a raw variable
    moop.addDesign({'name': "x6",
                    'des_type': "raw",
                    'lb': 0.0, 'ub': 1.0})
    # Add a custom variable
    moop.addDesign({'name': "x5",
                    'des_type': "custom",
                    'embedder': IdentityEmbedder,
                    'lb': 0.0, 'ub': 1.0})
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
        xi = {}
        xi["x0"] = int(1000.0 * num[0])
        xi["x1"] = num[1] - 1.0
        xi["x2"] = np.round(num[2])
        xi["x3"] = np.random.choice(["biggie", "shortie", "shmedium"])
        xi["x4"] = np.random.randint(-5, 5)
        xi["x5"] = num[5]
        xi["x6"] = num[6]
        xi["x7"] = 0
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (all(xxi >= -1.0e-8) and all(xxi <= 1 + 1.0e-8))
        assert (xxi.size == moop.n_latent)
        # Check extraction
        assert (all([abs(moop._extract(xxi)[key] - xi[key]) < 1.0e-8
                     for key in ["x0", "x1", "x2", "x4", "x6"]]))
        assert (moop._extract(xxi)["x3"] == xi["x3"])
        assert (abs(float(moop._extract(xxi)["x5"]) - float(xi["x5"]))
                < 1.0e-8)
        assert (moop._extract(xxi)["x7"] == xi["x7"])


if __name__ == "__main__":
    test_MOOP_addDesign_bad_cont()
    test_MOOP_addDesign_bad_cat()
    test_MOOP_addDesign_bad_int()
    test_MOOP_addDesign()
    test_MOOP_embed_extract_unnamed1()
    test_MOOP_embed_extract_unnamed2()
    test_MOOP_embed_extract_named1()
    test_MOOP_embed_extract_named2()
