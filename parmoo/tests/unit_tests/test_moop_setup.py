
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
    assert (moop.m == 0 and moop.n_feature == 0 and moop.n_latent == 0 and
            moop.s == 0 and moop.o == 0 and moop.p == 0)
    moop = MOOP(LocalSurrogate_PS, hyperparams={'test': 0})
    assert (moop.m == 0 and moop.n_feature == 0 and moop.n_latent == 0 and
            moop.s == 0 and moop.o == 0 and moop.p == 0)
    assert (moop.opt_hp['test'] == 0)


def test_MOOP_addDesign():
    """ Check that the MOOP class handles adding design variables properly.

    Initialize a MOOP objects, and add several design variables.

    """

    from parmoo import MOOP
    from parmoo.embeddings import IdentityEmbedder
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
    assert (moop.n_latent == 0)
    # Now add some continuous and integer design variables
    moop.addDesign({'lb': 0.0,
                    'ub': 1.0})
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1.0})
    assert (moop.n_latent == 1)
    moop.addDesign({'name': "x2",
                    'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0,
                    'des_tol': 0.01})
    assert (moop.n_latent == 2)
    moop.addDesign({'des_type': "integer",
                    'lb': 0,
                    'ub': 4})
    assert (moop.n_latent == 3)
    # Now add some categorical design variables
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    assert (moop.n_latent == 4)
    moop.addDesign({'des_type': "categorical",
                    'levels': 3})
    assert (moop.n_latent == 6)
    moop.addDesign({'name': "x6",
                    'des_type': "categorical",
                    'levels': ["boy", "girl", "doggo"]})
    assert (moop.n_latent == 8)
    # Now add a custom design variables
    moop.addDesign({'des_type': "custom",
                    'lb': -100.0,
                    'ub': 100.0,
                    'embedder': IdentityEmbedder})
    assert (moop.n_latent == 9)
    moop.addDesign({'des_type': "raw",
                    'lb': -100.0,
                    'ub': 100.0})
    assert (moop.n_latent == 10)


def test_MOOP_addSimulation():
    """ Check that the MOOP class handles adding new simulations properly.

    Initialize several MOOPs, and add several simulations. Check that
    the metadata is updated correctly.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

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
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    moop.addSimulation(g1)
    assert (moop.m == 1 and moop.n_latent == 3 and moop.s == 1 and
            moop.o == 0 and moop.p == 0)
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
    g2 = {'m': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    moop.addSimulation(g1, g2)
    assert (moop.m == 3 and moop.n_latent == 3 and moop.s == 2 and
            moop.o == 0 and moop.p == 0)
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
    moop.addSimulation(g3, g4)
    # Try to use a repeated name to test error handling
    with pytest.raises(ValueError):
        moop.addSimulation(g4)
    # Check the names
    assert (moop.sim_schema[0][0] == "sim1")
    assert (moop.sim_schema[1][0] == "sim2")
    assert (moop.sim_schema[2][0] == "Bobo1")
    assert (moop.sim_schema[3][0] == "Bobo2")


def test_MOOP_addObjective():
    """ Check that the MOOP class handles adding objectives properly.

    Initialize a MOOP object and check that the addObjective() function works
    correctly.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Initialize a MOOP with 2 SimGroups, one of which has 2 outputs
    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
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

    import numpy as np
    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Initialize a MOOP with 2 SimGroups, one of which has 2 outputs
    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
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

    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Initialize a MOOP with 3 variables and 3 objectives
    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addObjective({'obj_func': lambda x, s: x[0]},
                      {'obj_func': lambda x, s: x[1]},
                      {'obj_func': lambda x, s: x[2]})
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
    # Check that no acquisitions were added then add 3 good acquisitions
    assert (len(moop.acquisitions) == 0)
    moop.addAcquisition({'acquisition': UniformWeights})
    moop.addAcquisition({'acquisition': UniformWeights},
                        {'acquisition': UniformWeights, 'hyperparams': {}})
    moop.compile()
    assert (len(moop.acquisitions) == 3)


def test_MOOP_getTypes():
    """ Check that the MOOP class handles getting dtypes properly.

    Initialize a MOOP object, add design variables, simulations, objectives,
    and constraints, and get the corresponding types.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF

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
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    moop.addSimulation(g1)
    moop.addObjective({'obj_func': lambda x, s: [sum(s)]})
    moop.addConstraint({'constraint': lambda x, s: [sum(s) - 1]})
    # Check the dtypes
    assert (np.zeros(1, dtype=moop.getDesignType()).size == 1)
    assert (np.zeros(1, dtype=moop.getSimulationType()).size == 1)
    assert (np.zeros(1, dtype=moop.getObjectiveType()).size == 1)
    assert (np.zeros(1, dtype=moop.getConstraintType()).size == 1)


if __name__ == "__main__":
    test_MOOP_init()
    test_MOOP_addDesign()
    test_MOOP_addSimulation()
    test_MOOP_addObjective()
    test_MOOP_addConstraint()
    test_MOOP_addAcquisition()
    test_MOOP_getTypes()
