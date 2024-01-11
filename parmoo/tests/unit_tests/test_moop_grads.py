
def test_MOOP_evaluatePenalty():
    """ Check that the MOOP class handles evaluating penalty function properly.

    Initialize a MOOP object and check that the evaluatePenalty() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import GlobalSurrogate_PS
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
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addObjective({'obj_func': f1})
    assert (np.all(moop1.evaluatePenalty(np.zeros(3)) == np.zeros(1)))
    assert (np.all(moop1.evaluatePenalty(np.ones(3)) == 3.0 * np.ones(1)))
    moop1.addConstraint({'constraint': c1})
    assert (np.all(moop1.evaluatePenalty(np.zeros(3)) == np.zeros(1)))
    assert (np.all(moop1.evaluatePenalty(np.ones(3)) == 3.75 * np.ones(1)))
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.evaluateSimulation(np.ones(3), 0)
    moop1.evaluateSimulation(np.ones(3), 1)
    moop1.fitSurrogates()
    moop1.addObjective({'obj_func': f1})
    moop1.addObjective({'obj_func': f1})
    assert (np.all(moop1.evaluatePenalty(np.zeros(3)) == np.zeros(1)))
    assert (np.all(moop1.evaluatePenalty(np.ones(3)) == 3.0 * np.ones(1)))
    moop1.addConstraint({'constraint': c1})
    assert (np.all(moop1.evaluatePenalty(np.zeros(3)) == np.zeros(1)))
    assert (np.all(moop1.evaluatePenalty(np.ones(3)) == 3.75 * np.ones(1)))
    # Now try some bad evaluations
    with pytest.raises(TypeError):
        moop1.evaluatePenalty(10.0)
    with pytest.raises(ValueError):
        moop1.evaluatePenalty(np.zeros(1))
    # Adjust the scaling and compare
    moop2 = MOOP(GlobalSurrogate_PS)
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
    assert (np.linalg.norm(moop1.evaluatePenalty(x) -
                           moop2.evaluatePenalty(xx)) < 0.00000001)


def test_MOOP_evaluateGradients_1():
    """ Check that the MOOP class handles evaluating gradients properly.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import GlobalSurrogate_PS
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
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addObjective({'obj_func': f1})
    assert (np.all(moop1.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert (np.all(moop1.evaluateGradients(np.ones(3)) ==
                   2.0 * np.ones((1, 3))))
    moop1.addConstraint({'constraint': c1})
    assert (np.all(moop1.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    result = 2.0 * np.ones((1, 3))
    result[0, 0] = 3.0
    assert (np.all(moop1.evaluateGradients(np.ones(3)) == result))
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.evaluateSimulation(np.ones(3), 0)
    moop1.evaluateSimulation(np.ones(3), 1)
    moop1.fitSurrogates()
    moop1.addObjective({'obj_func': f1})
    assert (np.all(moop1.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert (np.all(moop1.evaluateGradients(np.ones(3)) ==
                   2.0 * np.ones((1, 3))))
    moop1.addConstraint({'constraint': c1})
    assert (np.all(moop1.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert (np.all(moop1.evaluateGradients(np.ones(3)) == result))
    result = np.zeros((2, 3))
    result[1, 0] = 1.0
    result[0, :] = 2.0
    result[0, 0] = 3.0
    moop1.addObjective({'obj_func': f2})
    assert (np.all(moop1.evaluateGradients(np.ones(3)) == result))
    moop1.addConstraint({'constraint': c2})
    assert (np.all(moop1.evaluateGradients(np.ones(3)) == result))
    # Now try some bad evaluations
    with pytest.raises(TypeError):
        moop1.evaluateGradients(10.0)
    with pytest.raises(ValueError):
        moop1.evaluateGradients(np.zeros(1))
    # Adjust the scaling and try again
    moop2 = MOOP(GlobalSurrogate_PS)
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
    assert (np.linalg.norm(moop1.evaluatePenalty(x) -
                           moop2.evaluatePenalty(xx)) < 0.00000001)


def test_MOOP_evaluateGradients_2():
    """ Check that the MOOP class handles evaluating gradients properly.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import GlobalSurrogate_PS
    import numpy as np

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

    moop3 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop3.addDesign({'name': ('x' + str(i + 1)), 'lb': 0.0, 'ub': 1.0})
    moop3.addObjective({'obj_func': f3})
    assert (np.all(moop3.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert (np.all(moop3.evaluateGradients(np.ones(3)) ==
                   2.0 * np.ones((1, 3))))
    moop3.addConstraint({'constraint': c3})
    assert (np.all(moop3.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    result = 2.0 * np.ones((1, 3))
    result[0, 0] = 3.0
    assert (np.all(moop3.evaluateGradients(np.ones(3)) == result))
    moop3 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop3.addDesign({'name': ('x' + str(i + 1)), 'lb': 0.0, 'ub': 1.0})
    moop3.addSimulation(g3, g4)
    moop3.evaluateSimulation(np.ones(1, dtype=[("x1", float), ("x2", float),
                                               ("x3", float)]), 0)
    moop3.evaluateSimulation(np.ones(1, dtype=[("x1", float), ("x2", float),
                                               ("x3", float)]), 1)
    moop3.fitSurrogates()
    moop3.addObjective({'obj_func': f3})
    assert (np.all(moop3.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert (np.all(moop3.evaluateGradients(np.ones(3)) ==
                   2.0 * np.ones((1, 3))))
    moop3.addConstraint({'constraint': c3})
    assert (np.all(moop3.evaluateGradients(np.zeros(3)) == np.zeros((1, 3))))
    assert (np.all(moop3.evaluateGradients(np.ones(3)) == result))
    result = np.zeros((2, 3))
    result[1, 0] = 1.0
    result[0, :] = 2.0
    result[0, 0] = 3.0
    moop3.addObjective({'obj_func': f4})
    assert (np.all(moop3.evaluateGradients(np.ones(3)) == result))
    moop3.addConstraint({'constraint': c4})
    assert (np.all(moop3.evaluateGradients(np.ones(3)) == result))
    # Adjust the scaling and try again
    moop4 = MOOP(GlobalSurrogate_PS)
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
    # Check for a double-sized step due to step in rescaled input space
    assert (np.linalg.norm(moop3.evaluateGradients(x) -
                           moop4.evaluateGradients(xx) * 2) < 0.00000001)


def test_MOOP_evaluateGradients_3():
    """ Check that the MOOP class handles evaluating gradients properly.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import GlobalSurrogate_BFGS, GlobalSurrogate_PS
    from parmoo.acquisitions import FixedWeights
    import numpy as np

    # Create 2 SimGroups for later
    g1 = {'n': 3,
          'm': 1,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF,
          'hyperparams': {'search_budget': 100}}
    g2 = {'n': 3,
          'm': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm(x-0.5), np.linalg.norm(x-1.0)],
          'surrogate': GaussRBF,
          'hyperparams': {'search_budget': 100}}

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

    # Initialize a MOOP with 1 design var, 2 SimGroups, and 3 objectives
    moop1 = MOOP(GlobalSurrogate_BFGS, hyperparams={'opt_restarts': 20})
    moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': f1})
    moop1.addObjective({'obj_func': f2})
    moop1.addConstraint({'constraint': c1})
    moop1.addConstraint({'constraint': c2})
    moop1.addAcquisition({'acquisition': FixedWeights,
                          'hyperparams': {'weights': np.ones(2) / 2}})
    np.random.seed(0)
    moop1.solve(0)
    # Adjust the scaling and try again
    moop2 = MOOP(GlobalSurrogate_PS)
    moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': f1})
    moop2.addObjective({'obj_func': f2})
    moop2.addConstraint({'constraint': c1})
    moop2.addConstraint({'constraint': c2})
    moop2.addAcquisition({'acquisition': FixedWeights,
                          'hyperparams': {'weights': np.ones(2) / 2}})
    np.random.seed(0)
    moop2.solve(0)
    np.random.seed(0)
    b1 = moop1.iterate(1)
    np.random.seed(0)
    b2 = moop2.iterate(1)
    # Check that same solutions were found
    for x1, x2 in zip(b1, b2):
        assert (np.all(np.abs(x1[0] - x2[0]) < 0.1))


if __name__ == "__main__":
    test_MOOP_evaluatePenalty()
    test_MOOP_evaluateGradients_1()
    test_MOOP_evaluateGradients_2()
    test_MOOP_evaluateGradients_3()
