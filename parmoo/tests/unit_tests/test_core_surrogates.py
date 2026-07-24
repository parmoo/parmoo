""" Unit tests for MOOP_base surrogate fitting, updating, and evaluation.
"""

import pytest


def test_MOOP_base_fit_update_surrogates():
    """ Check that the MOOP_base class handles evaluating surrogate models
    properly.

    Initialize a MOOP_base object and check that the _evaluate_surrogates()
    function works correctly.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF

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


def test_MOOP_base_evaluate_surrogates():
    """ Check that the MOOP_base class handles evaluating surrogate models
    properly.

    Initialize a MOOP_base object and check that the _evaluate_surrogates()
    function works correctly.

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
        assert (np.linalg.norm(moop1._evaluate_surrogates(xi) - si) < 1.e-7)
        assert (np.linalg.norm(moop1._surrogate_uncertainty(xi) - sdi) < 1.e-3)
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
    moop2._set_surrogate_tr(np.zeros(3), np.inf)
    # Now compare evaluations against the original surrogate
    x = moop1._embed({'x1': 0, 'x2': 0, 'x3': 0})
    xx = moop2._embed({'x1': 0, 'x2': 0, 'x3': 0})
    assert (np.linalg.norm(moop1._evaluate_surrogates(x) -
                           moop2._evaluate_surrogates(xx)) < 1.0e-8)
    x = moop1._embed({'x1': 1, 'x2': 1, 'x3': 1})
    xx = moop2._embed({'x1': 1, 'x2': 1, 'x3': 1})
    assert (np.linalg.norm(moop1._evaluate_surrogates(x) -
                           moop2._evaluate_surrogates(xx)) < 1.0e-8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
