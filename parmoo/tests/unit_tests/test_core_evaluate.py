""" Unit tests for MOOP_base objective, constraint, and penalty evaluation.
"""

import pytest


def test_MOOP_base_evaluate_objectives():
    """ Check that the MOOP_base class handles evaluating objectives properly.

    Initialize a MOOP_base object and check that the _evaluate_objectives()
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


def test_MOOP_base_evaluate_constraints():
    """ Check that the MOOP_base class handles evaluating constraints properly.

    Initialize a MOOP_base object and check that the _evaluate_constraints()
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
    moop.addConstraint({'constraint':
                        lambda x, s: s["sim2"][0] + s["sim2"][1]})
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
    moop._set_surrogate_tr(np.zeros(3), np.inf)
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


def test_MOOP_base_evaluate_penalty():
    """ Check that the MOOP_base class handles evaluating penalty function
    properly.

    Initialize a MOOP_base object and check that the _evaluate_penalty()
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
    raise SystemExit(pytest.main([__file__]))
