
def test_MOOP_embed_extract():
    """ Test that the MOOP class can embed/extract design variables.

    Add several design variables and generate an embedding. Then embed and
    extract several inputs, and check that the results match up to the
    design tolerance.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.embeddings import IdentityEmbedder
    from parmoo.optimizers import LocalSurrogate_PS

    # Create a MOOP with 6 variables of mixed types
    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'des_type': "integer",
                    'lb': 0,
                    'ub': 1000})
    moop.addDesign({'des_type': "continuous",
                    'lb': -1.0,
                    'ub': 0.0})
    moop.addDesign({'des_type': "categorical",
                    'levels': 2})
    moop.addDesign({'des_type': "categorical",
                    'levels': ["0guy", "1guy", "2guy"]})
    moop.addDesign({'des_type': "custom",
                    'embedder': IdentityEmbedder,
                    'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'des_type': "raw",
                    'lb': 0.0, 'ub': 5.0})
    moop.addDesign({'des_type': "continuous",
                    'lb': 0.0,
                    'ub': 1.0})
    # Embed/extract lower bounds, upper bounds, and 5 random values
    test_pts = np.zeros((7, 7))
    test_pts[1, :] = 1.0
    test_pts[2:, :] = np.random.sample((5, 7))
    for xi_tmp in test_pts:
        xi_tmp = np.random.sample(7)
        xi = {}
        xi["x1"] = int(1000 * xi_tmp[0])
        xi["x2"] = xi_tmp[1] - 1.0
        xi["x3"] = np.round(xi_tmp[2])
        xi["x4"] = f"{int(np.round(xi_tmp[3]))}guy"
        xi["x5"] = xi_tmp[4]
        xi["x6"] = xi_tmp[5] * 5.0
        xi["x7"] = xi_tmp[6]
        xxi = moop._embed(xi)
        # Check that embedding is legal
        assert (np.all(xxi[:6] >= -1.0e-8) and np.all(xxi[:6] <= 1 + 1.0e-8))
        assert (xxi[7] >= -1.0e-8 and xxi[7] <= 1 + 1.0e-8)
        assert (xxi.size == moop.n_latent)
        # Check extraction matches
        assert (moop._extract(xxi)["x1"] == xi["x1"])
        assert (np.abs(moop._extract(xxi)["x2"] - xi["x2"]) < 1.0e-8)
        assert (np.abs(moop._extract(xxi)["x3"] - xi["x3"]) < 1.0e-8)
        assert (moop._extract(xxi)["x4"] == xi["x4"])
        assert (moop._extract(xxi)["x5"] == xi["x5"])
        assert (np.abs(moop._extract(xxi)["x6"] - xi["x6"]) < 1.0e-8)
        assert (np.abs(moop._extract(xxi)["x7"] - xi["x7"]) < 1.0e-8)


def test_MOOP_pack_unpack_sim():
    """ Check that the MOOP class handles simulation packing correctly.

    Initialize a MOOP objecti with and without design variable names.
    Add 2 simulations and pack/unpack each output.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF

    # Create a continuous MOOP with 2 sims for packing/unpacking
    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1000.0},
                   {'name': "x2", 'lb': -1.0, 'ub': 0.0})
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'m': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-1.0), np.linalg.norm(x-0.5)],
          'surrogate': GaussRBF}
    moop.addSimulation(g1, g2)
    # Create a test vector
    sx = np.array([1.0, 2.0, 3.0])
    sxx = np.zeros(1, dtype=moop.sim_schema)
    sxx[0]['sim1'] = 1.0
    sxx[0]['sim2'][:] = np.array([2.0, 3.0])
    # Check packing
    assert (np.all(moop._pack_sim(sxx) == sx))
    # Check unpacking
    assert (moop._unpack_sim(sx)['sim1'] == sxx[0]['sim1'])
    assert (moop._unpack_sim(sx)['sim2'][0] == sxx[0]['sim2'][0])
    assert (moop._unpack_sim(sx)['sim2'][1] == sxx[0]['sim2'][1])


def test_MOOP_fit_update_surrogates():
    """ Check that the MOOP class handles evaluating surrogate models properly.

    Initialize a MOOP object and check that the _evaluate_surrogates() function
    works correctly.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Initialize a continuous MOOP with 2 sims, 3 objs
    moop1 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i] for i in x])],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i]-1 for i in x]),
                                 np.linalg.norm([x[i]-0.5 for i in x])],
          'surrogate': GaussRBF}
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': lambda x, s: x["x1"]},
                       {'obj_func': lambda x, s: s["sim1"]},
                       {'obj_func': lambda x, s: sum(s["sim2"])})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    # Evaluate some data points and fit the surrogates
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 0}, sn)
        moop1.evaluateSimulation({'x1': 0.5, 'x2': 0.5, 'x3': 0.5}, sn)
        moop1.evaluateSimulation({'x1': 1, 'x2': 0, 'x3': 0}, sn)
        moop1.evaluateSimulation({'x1': 0, 'x2': 1, 'x3': 0}, sn)
        moop1.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 1}, sn)
        moop1.evaluateSimulation({'x1': 1, 'x2': 1, 'x3': 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.ones(3) * 0.5, np.ones(3) * 0.5)
    # Create an identical copy
    moop2 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': lambda x, s: x["x1"]},
                       {'obj_func': lambda x, s: s["sim1"]},
                       {'obj_func': lambda x, s: sum(s["sim2"])})
    moop2.addAcquisition({'acquisition': UniformWeights})
    moop2.compile()
    # Fit with half the training data used by moop1
    for sn in ["sim1", "sim2"]:
        moop2.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 0}, sn)
        moop2.evaluateSimulation({'x1': 0.5, 'x2': 0.5, 'x3': 0.5}, sn)
        moop2.evaluateSimulation({'x1': 1, 'x2': 0, 'x3': 0}, sn)
    moop2._fit_surrogates()
    moop2._set_surrogate_tr(np.ones(3) * 0.5, np.ones(3) * 0.5)
    # Update with the other half of the training data
    for sn in ["sim1", "sim2"]:
        moop2.evaluateSimulation({'x1': 0, 'x2': 1, 'x3': 0}, sn)
        moop2.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 1}, sn)
        moop2.evaluateSimulation({'x1': 1, 'x2': 1, 'x3': 1}, sn)
    moop2._update_surrogates()
    moop2._set_surrogate_tr(np.ones(3) * 0.5, np.ones(3) * 0.5)
    # Do 5 random tests and make sure the outputs are (near) identical
    for xi in np.random.sample((5, 3)):
        s1i = moop1._evaluate_surrogates(xi)
        s2i = moop2._evaluate_surrogates(xi)
        assert (np.linalg.norm(s1i - s2i) < 1.0e-8)


def test_MOOP_evaluate_surrogates():
    """ Check that the MOOP class handles evaluating surrogate models properly.

    Initialize a MOOP object and check that the _evaluate_surrogates() function
    works correctly.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Initialize a continuous MOOP with 2 sims, 3 objs
    moop1 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i] for i in x])],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i]-1 for i in x]),
                                 np.linalg.norm([x[i]-0.5 for i in x])],
          'surrogate': GaussRBF}
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': lambda x, s: x["x1"]},
                       {'obj_func': lambda x, s: s["sim1"]},
                       {'obj_func': lambda x, s: sum(s["sim2"])})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    # Try some bad evaluations
    with pytest.raises(ValueError):
        moop1.evaluateSimulation(np.zeros(3), -1)
    # Evaluate some data points and fit the surrogates
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 0}, sn)
        moop1.evaluateSimulation({'x1': 0.5, 'x2': 0.5, 'x3': 0.5}, sn)
        moop1.evaluateSimulation({'x1': 1, 'x2': 0, 'x3': 0}, sn)
        moop1.evaluateSimulation({'x1': 0, 'x2': 1, 'x3': 0}, sn)
        moop1.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 1}, sn)
        moop1.evaluateSimulation({'x1': 1, 'x2': 1, 'x3': 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.ones(3) * 0.5, np.ones(3) * 0.5)
    # Now do some test evaluations and check the results
    test_cases = [
        (np.zeros(3), np.array([0, np.sqrt(3), np.sqrt(0.75)]), 0),
        (np.ones(3) / 2, np.array([np.sqrt(0.75), np.sqrt(0.75), 0]), 0),
        (np.eye(3)[0], np.array([1, np.sqrt(2), np.sqrt(0.75)]), 0),
        (np.eye(3)[1], np.array([1, np.sqrt(2),  np.sqrt(0.75)]), 0),
        (np.eye(3)[2], np.array([1, np.sqrt(2),  np.sqrt(0.75)]), 0),
        (np.ones(3), np.array([np.sqrt(3), 0.0, np.sqrt(0.75)]), 0)
    ]
    for xi, si, sdi in test_cases:
        assert (np.linalg.norm(moop1._evaluate_surrogates(xi) - si) < 1.0e-8)
        assert (np.linalg.norm(moop1._surrogate_uncertainty(xi) - sdi) < 1.0e-4)
    # Evaluate one point not in the training set and check that std_dev > 0
    xi = np.ones(3) * 0.75
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
    moop2.addAcquisition({'acquisition': UniformWeights})
    moop2.compile()
    for sn in ["sim1", "sim2"]:
        moop2.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 0}, sn)
        moop2.evaluateSimulation({'x1': 0.5, 'x2': 0.5, 'x3': 0.5}, sn)
        moop2.evaluateSimulation({'x1': 1, 'x2': 0, 'x3': 0}, sn)
        moop2.evaluateSimulation({'x1': 0, 'x2': 1, 'x3': 0}, sn)
        moop2.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 1}, sn)
        moop2.evaluateSimulation({'x1': 1, 'x2': 1, 'x3': 1}, sn)
    moop2._fit_surrogates()
    moop2._set_surrogate_tr(np.zeros(3), np.infty)
    # Now compare evaluations against the original surrogate
    x = moop1._embed({'x1': 0, 'x2': 0, 'x3': 0})
    xx = moop2._embed({'x1': 0, 'x2': 0, 'x3': 0})
    assert (np.linalg.norm(moop1._evaluate_surrogates(x) -
                           moop2._evaluate_surrogates(xx)) < 1.0e-8)
    x = moop1._embed({'x1': 1, 'x2': 1, 'x3': 1})
    xx = moop2._embed({'x1': 1, 'x2': 1, 'x3': 1})
    assert (np.linalg.norm(moop1._evaluate_surrogates(x) -
                           moop2._evaluate_surrogates(xx)) < 1.0e-8)


def test_MOOP_evaluate_objectives():
    """ Check that the MOOP class handles evaluating objectives properly.

    Initialize a MOOP object and check that the _evaluate_objectives() function
    works correctly.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Initialize a continuous MOOP with 2 sims, 3 objs
    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i] for i in x])],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i]-1 for i in x]),
                                 np.linalg.norm([x[i]-0.5 for i in x])],
          'surrogate': GaussRBF}
    moop.addSimulation(g1, g2)
    moop.addObjective({'obj_func': lambda x, s: x["x1"]},
                      {'obj_func': lambda x, s: s["sim1"][0]},
                      {'obj_func': lambda x, s: s["sim2"][0] + s["sim2"][1]})
    moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    # Try some bad evaluations
    with pytest.raises(ValueError):
        moop.evaluateSimulation(np.zeros(3), -1)
    # Evaluate some data points and fit the surrogates
    for sn in ["sim1", "sim2"]:
        moop.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 0}, sn)
        moop.evaluateSimulation({'x1': 0.5, 'x2': 0.5, 'x3': 0.5}, sn)
        moop.evaluateSimulation({'x1': 1, 'x2': 0, 'x3': 0}, sn)
        moop.evaluateSimulation({'x1': 0, 'x2': 1, 'x3': 0}, sn)
        moop.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 1}, sn)
        moop.evaluateSimulation({'x1': 1, 'x2': 1, 'x3': 1}, sn)
    moop._fit_surrogates()
    moop._set_surrogate_tr(np.ones(3) * 0.5, np.ones(3) * 0.5)
    # Now do some test evaluations and check the results
    test_cases = [
        (np.zeros(3), np.array([0.0, 0.0, np.sqrt(3) + np.sqrt(0.75)])),
        (np.ones(3) * 0.5, np.array([0.5, np.sqrt(0.75), np.sqrt(0.75)])),
        (np.eye(3)[0], np.array([1.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
        (np.eye(3)[1], np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
        (np.eye(3)[2], np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
        (np.ones(3), np.array([1.0, np.sqrt(3), np.sqrt(0.75)]))
    ]
    for xi, fi in test_cases:
        sxi = moop._evaluate_surrogates(xi)
        fxi = moop._evaluate_objectives(xi, sxi)
        assert (np.linalg.norm(fi - fxi) < 1.0e-8)


def test_MOOP_evaluate_constraints():
    """ Check that the MOOP class handles evaluating constraints properly.

    Initialize a MOOP object and check that the _evaluate_constraints() function
    works correctly.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Initialize a continuous MOOP with 2 sims, 3 cons
    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i] for i in x])],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i]-1 for i in x]),
                                 np.linalg.norm([x[i]-0.5 for i in x])],
          'surrogate': GaussRBF}
    moop.addSimulation(g1, g2)
    moop.addObjective({'obj_func': lambda x, s: x["x1"]})
    moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    # Evaluate an empty constraint and check that a zero array is returned
    assert (np.all(moop._evaluate_constraints(np.zeros(3), np.zeros(3))
            == np.zeros(1)))
    # Now add 3 constraints
    moop.addConstraint({'constraint': lambda x, s: x["x1"]})
    moop.addConstraint({'constraint': lambda x, s: s["sim1"][0]})
    moop.addConstraint({'constraint': lambda x, s: s["sim2"][0] + s["sim2"][1]})
    moop.compile()
    # Evaluate some data points and fit the surrogates
    for sn in ["sim1", "sim2"]:
        moop.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 0}, sn)
        moop.evaluateSimulation({'x1': 0.5, 'x2': 0.5, 'x3': 0.5}, sn)
        moop.evaluateSimulation({'x1': 1, 'x2': 0, 'x3': 0}, sn)
        moop.evaluateSimulation({'x1': 0, 'x2': 1, 'x3': 0}, sn)
        moop.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 1}, sn)
        moop.evaluateSimulation({'x1': 1, 'x2': 1, 'x3': 1}, sn)
    moop._fit_surrogates()
    moop._set_surrogate_tr(np.zeros(3), np.infty)
    # Now do some test evaluations and check the results
    test_cases = [
        (np.zeros(3), np.array([0.0, 0.0, np.sqrt(3) + np.sqrt(0.75)])),
        (np.ones(3) * 0.5, np.array([0.5, np.sqrt(0.75), np.sqrt(0.75)])),
        (np.eye(3)[0], np.array([1.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
        (np.eye(3)[1], np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
        (np.eye(3)[2], np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
        (np.ones(3), np.array([1.0, np.sqrt(3), np.sqrt(0.75)]))
    ]
    for xi, ci in test_cases:
        sxi = moop._evaluate_surrogates(xi)
        cxi = moop._evaluate_constraints(xi, sxi)
        assert (np.linalg.norm(ci - cxi) < 1.0e-8)


def test_MOOP_evaluate_penalty():
    """ Check that the MOOP class handles evaluating penalty function properly.

    Initialize a MOOP object and check that the _evaluate_penalty() function
    works correctly.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Initialize a continuous MOOP with 2 sims, 3 objs, 1 cons
    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i] for i in x])],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[i]-1 for i in x]),
                                 np.linalg.norm([x[i]-0.5 for i in x])],
          'surrogate': GaussRBF}
    moop.addSimulation(g1, g2)
    moop.addObjective({'obj_func': lambda x, s: x["x1"]},
                      {'obj_func': lambda x, s: s["sim1"][0]},
                      {'obj_func': lambda x, s: s["sim2"][0] + s["sim2"][1]})
    moop.addConstraint({'constraint': lambda x, s: x["x1"] - 0.5})
    moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    # Try some bad evaluations
    with pytest.raises(ValueError):
        moop.evaluateSimulation(np.zeros(3), -1)
    # Evaluate some data points and fit the surrogates
    for sn in ["sim1", "sim2"]:
        moop.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 0}, sn)
        moop.evaluateSimulation({'x1': 0.5, 'x2': 0.5, 'x3': 0.5}, sn)
        moop.evaluateSimulation({'x1': 1, 'x2': 0, 'x3': 0}, sn)
        moop.evaluateSimulation({'x1': 0, 'x2': 1, 'x3': 0}, sn)
        moop.evaluateSimulation({'x1': 0, 'x2': 0, 'x3': 1}, sn)
        moop.evaluateSimulation({'x1': 1, 'x2': 1, 'x3': 1}, sn)
    moop._fit_surrogates()
    moop._set_surrogate_tr(np.ones(3) * 0.5, np.ones(3) * 0.5)
    # Now do some test evaluations and check the results
    test_cases = [
        (np.zeros(3), np.array([0.0, 0.0, np.sqrt(3) + np.sqrt(0.75)])),
        (np.ones(3) * 0.5, np.array([0.5, np.sqrt(0.75), np.sqrt(0.75)])),
        (np.eye(3)[0], np.array([1.0, 1.0, np.sqrt(2) + np.sqrt(0.75)]) + 0.5),
        (np.eye(3)[1], np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
        (np.eye(3)[2], np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
        (np.ones(3), np.array([1.0, np.sqrt(3), np.sqrt(0.75)]) + 0.5)
    ]
    for xi, pi in test_cases:
        sxi = moop._evaluate_surrogates(xi)
        pxi = moop._evaluate_penalty(xi, sxi)
        assert (np.linalg.norm(pi - pxi) < 1.0e-8)


if __name__ == "__main__":
    test_MOOP_embed_extract()
    test_MOOP_pack_unpack_sim()
    test_MOOP_fit_update_surrogates()
    test_MOOP_evaluate_surrogates()
    test_MOOP_evaluate_objectives()
    test_MOOP_evaluate_constraints()
    test_MOOP_evaluate_penalty()
