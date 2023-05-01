
def test_MOOP_evaluateExpectedValue_unnamed():
    """ Check that the MOOP class handles evaluating objective EVs properly.

    Initialize a MOOP object and check that the evaluateObjectives() function
    works correctly for expected values.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            return np.eye(x.size)[0]
        elif der == 2:
            return np.zeros(s.size)
        else:
            return x[0]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * np.eye(ev_s.size)[0] * ev_s[0]
        elif der == 3:
            return 2.0 * np.eye(sd_s.size)[0] * sd_s[0]
        else:
            return ev_s[0] ** 2 + sd_s[0] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * (np.eye(ev_s.size)[1] * ev_s[1] +
                          np.eye(ev_s.size)[2] * ev_s[2])
        elif der == 3:
            return 2.0 * (np.eye(sd_s.size)[1] * sd_s[1] +
                          np.eye(sd_s.size)[2] * sd_s[2])
        else:
            return ev_s[1] ** 2 + sd_s[1] ** 2 + ev_s[2] ** 2 + sd_s[2] ** 2

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
    moop1.addObjective({'obj_func': obj_1},
                       {'exp_func': ev_1},
                       {'exp_func': ev_2})
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
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': obj_1},
                       {'obj_func': lambda x, s, der=0: ev_1(x, s,
                                                             np.zeros(s.size),
                                                             der=der)},
                       {'obj_func': lambda x, s, der=0: ev_2(x, s,
                                                             np.zeros(s.size),
                                                             der=der)})
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), 0)
    moop2.evaluateSimulation(np.zeros(3), 1)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 0)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 1)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 1)
    moop2.evaluateSimulation(np.ones(3), 0)
    moop2.evaluateSimulation(np.ones(3), 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)
    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluateObjectives(np.zeros(3)) -
                           np.asarray([0.0, 0.0, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.asarray([0.5,
                                                                0.5, 0.5]))
                           - np.asarray([0.5, 0.75, 0.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.asarray([1.0, 0.0,
                                                                0.0]))
                           - np.asarray([1.0, 1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.asarray([0.0, 1.0,
                                                                0.0]))
                           - np.asarray([0.0, 1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.asarray([0.0, 0.0,
                                                                1.0]))
                           - np.asarray([0.0, 1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.ones(3)) -
                           np.asarray([1.0, 3.0, 0.75])) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        assert (np.all(moop1.evaluateObjectives(xi) >=
                       moop2.evaluateObjectives(xi)))


def test_MOOP_evaluateExpectedValuePenalty_unnamed():
    """ Check that the MOOP class handles evaluating penalized EVs properly.

    Initialize a MOOP object and check that the evaluatePenalty() function
    works correctly for expected values.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            return np.eye(x.size)[0]
        elif der == 2:
            return np.zeros(s.size)
        else:
            return x[0]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * np.eye(ev_s.size)[0] * ev_s[0]
        elif der == 3:
            return 2.0 * np.eye(sd_s.size)[0] * sd_s[0]
        else:
            return ev_s[0] ** 2 + sd_s[0] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * (np.eye(ev_s.size)[1] * ev_s[1] +
                          np.eye(ev_s.size)[2] * ev_s[2])
        elif der == 3:
            return 2.0 * (np.eye(sd_s.size)[1] * sd_s[1] +
                          np.eye(sd_s.size)[2] * sd_s[2])
        else:
            return ev_s[1] ** 2 + sd_s[1] ** 2 + ev_s[2] ** 2 + sd_s[2] ** 2

    def c2(x, s, der=0): return ev_2(x, s, np.zeros(s.size), der=der)

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
    moop1.addObjective({'obj_func': obj_1},
                       {'exp_func': ev_1})
    moop1.addConstraint({'exp_func': ev_2})
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
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': obj_1},
                       {'obj_func': lambda x, s, der=0: ev_1(x, s,
                                                             np.zeros(s.size),
                                                             der=der)})
    moop2.addConstraint({'constraint': c2})
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), 0)
    moop2.evaluateSimulation(np.zeros(3), 1)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 0)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 1)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 1)
    moop2.evaluateSimulation(np.ones(3), 0)
    moop2.evaluateSimulation(np.ones(3), 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)
    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluatePenalty(np.zeros(3)) -
                           np.asarray([3.75, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.asarray([0.5, 0.5, 0.5]))
                           - np.asarray([1.25, 1.5])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.asarray([1.0, 0.0, 0.0]))
                           - np.asarray([3.75, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.asarray([0.0, 1.0, 0.0]))
                           - np.asarray([2.75, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.asarray([0.0, 0.0, 1.0]))
                           - np.asarray([2.75, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.ones(3)) -
                           np.asarray([1.75, 3.75])) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        assert (np.all(moop1.evaluatePenalty(xi) >=
                       moop2.evaluatePenalty(xi)))


def test_MOOP_expectedValueGradient_unnamed():
    """ Check that the MOOP class evaluates the expected value gradient.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly for objectives.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            return np.eye(x.size)[0]
        elif der == 2:
            return np.zeros(s.size)
        else:
            return x[0]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * np.eye(ev_s.size)[0] * ev_s[0]
        elif der == 3:
            return 2.0 * np.eye(sd_s.size)[0] * sd_s[0]
        else:
            return ev_s[0] ** 2 + sd_s[0] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * (np.eye(ev_s.size)[1] * ev_s[1] +
                          np.eye(ev_s.size)[2] * ev_s[2])
        elif der == 3:
            return 2.0 * (np.eye(sd_s.size)[1] * sd_s[1] +
                          np.eye(sd_s.size)[2] * sd_s[2])
        else:
            return ev_s[1] ** 2 + sd_s[1] ** 2 + ev_s[2] ** 2 + sd_s[2] ** 2

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
    moop1.addObjective({'obj_func': obj_1},
                       {'exp_func': ev_1},
                       {'exp_func': ev_2})
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
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': obj_1},
                       {'obj_func': lambda x, s, der=0: ev_1(x, s,
                                                             np.zeros(s.size),
                                                             der=der)},
                       {'obj_func': lambda x, s, der=0: ev_2(x, s,
                                                             np.zeros(s.size),
                                                             der=der)})
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), 0)
    moop2.evaluateSimulation(np.zeros(3), 1)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 0)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 1)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 1)
    moop2.evaluateSimulation(np.ones(3), 0)
    moop2.evaluateSimulation(np.ones(3), 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)
    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluateGradients(np.zeros(3)) -
                           moop2.evaluateGradients(np.zeros(3))) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(0.5 * np.ones(3)) -
                           moop2.evaluateGradients(0.5 * np.ones(3)))
            < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[0]) -
                           moop2.evaluateGradients(np.eye(3)[0])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[1]) -
                           moop2.evaluateGradients(np.eye(3)[1])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[2]) -
                           moop2.evaluateGradients(np.eye(3)[2])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.ones(3)) -
                           moop2.evaluateGradients(np.ones(3))) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        dsig_dxi = moop1.surrogateUncertainty(xi, grad=True)
        df_dsig = np.zeros((3, 3))
        df_dsig[1, :] = ev_1(xi, np.zeros(3),
                             moop1.surrogateUncertainty(xi), der=3)
        df_dsig[2, :] = ev_2(xi, np.zeros(3),
                             moop1.surrogateUncertainty(xi), der=3)
        assert (np.linalg.norm(moop1.evaluateGradients(xi) -
                               np.dot(df_dsig, dsig_dxi) -
                               moop2.evaluateGradients(xi)) < 1.0e-4)


def test_MOOP_evaluateConstraintExpectedValue_unnamed():
    """ Check that the MOOP class handles evaluating constraint EVs properly.

    Initialize a MOOP object and check that the evaluateConstraints() function
    works correctly for expected-values.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            return np.eye(x.size)[0]
        elif der == 2:
            return np.zeros(s.size)
        else:
            return x[0]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * np.eye(ev_s.size)[0] * ev_s[0]
        elif der == 3:
            return 2.0 * np.eye(sd_s.size)[0] * sd_s[0]
        else:
            return ev_s[0] ** 2 + sd_s[0] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * (np.eye(ev_s.size)[1] * ev_s[1] +
                          np.eye(ev_s.size)[2] * ev_s[2])
        elif der == 3:
            return 2.0 * (np.eye(sd_s.size)[1] * sd_s[1] +
                          np.eye(sd_s.size)[2] * sd_s[2])
        else:
            return ev_s[1] ** 2 + sd_s[1] ** 2 + ev_s[2] ** 2 + sd_s[2] ** 2

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
    moop1.addObjective({'obj_func': obj_1})
    moop1.addConstraint({'exp_func': ev_1},
                        {'exp_func': ev_2})
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
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': obj_1})
    moop2.addConstraint({'constraint': lambda x, s, der=0: ev_1(x, s,
                                                                np.zeros(3),
                                                                der=der)},
                        {'constraint': lambda x, s, der=0: ev_2(x, s,
                                                                np.zeros(3),
                                                                der=der)})
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), 0)
    moop2.evaluateSimulation(np.zeros(3), 1)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 0)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 1)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 1)
    moop2.evaluateSimulation(np.ones(3), 0)
    moop2.evaluateSimulation(np.ones(3), 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)
    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluateConstraints(np.zeros(3)) -
                           np.asarray([0.0, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.asarray([0.5,
                                                                 0.5, 0.5]))
                           - np.asarray([0.75, 0.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.asarray([1.0, 0.0,
                                                                 0.0]))
                           - np.asarray([1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.asarray([0.0, 1.0,
                                                                 0.0]))
                           - np.asarray([1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.asarray([0.0, 0.0,
                                                                 1.0]))
                           - np.asarray([1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.ones(3)) -
                           np.asarray([3.0, 0.75])) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        assert (np.all(moop1.evaluateConstraints(xi) >=
                       moop2.evaluateConstraints(xi)))


def test_MOOP_expectedValueConstraintGradient_unnamed():
    """ Check that the MOOP class evaluates the expected value gradient.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly for constraint expected values.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            return np.eye(x.size)[0]
        elif der == 2:
            return np.zeros(s.size)
        else:
            return x[0]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * np.eye(ev_s.size)[0] * ev_s[0]
        elif der == 3:
            return 2.0 * np.eye(sd_s.size)[0] * sd_s[0]
        else:
            return ev_s[0] ** 2 + sd_s[0] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return 2.0 * (np.eye(ev_s.size)[1] * ev_s[1] +
                          np.eye(ev_s.size)[2] * ev_s[2])
        elif der == 3:
            return 2.0 * (np.eye(sd_s.size)[1] * sd_s[1] +
                          np.eye(sd_s.size)[2] * sd_s[2])
        else:
            return ev_s[1] ** 2 + sd_s[1] ** 2 + ev_s[2] ** 2 + sd_s[2] ** 2

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
    moop1.addObjective({'obj_func': obj_1})
    moop1.addConstraint({'exp_func': ev_1},
                        {'exp_func': ev_2})
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
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': obj_1})
    moop2.addConstraint({'constraint': lambda x, s, der=0: ev_1(x, s,
                                                                np.zeros(3),
                                                                der=der)},
                        {'constraint': lambda x, s, der=0: ev_2(x, s,
                                                                np.zeros(3),
                                                                der=der)})
    # Evaluate some data points and fit the surrogates
    moop2.evaluateSimulation(np.zeros(3), 0)
    moop2.evaluateSimulation(np.zeros(3), 1)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 0)
    moop2.evaluateSimulation(np.array([0.5, 0.5, 0.5]), 1)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([1.0, 0.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 1.0, 0.0]), 1)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 0)
    moop2.evaluateSimulation(np.array([0.0, 0.0, 1.0]), 1)
    moop2.evaluateSimulation(np.ones(3), 0)
    moop2.evaluateSimulation(np.ones(3), 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)
    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluateGradients(np.zeros(3)) -
                           moop2.evaluateGradients(np.zeros(3))) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(0.5 * np.ones(3)) -
                           moop2.evaluateGradients(0.5 * np.ones(3)))
            < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[0]) -
                           moop2.evaluateGradients(np.eye(3)[0])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[1]) -
                           moop2.evaluateGradients(np.eye(3)[1])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[2]) -
                           moop2.evaluateGradients(np.eye(3)[2])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.ones(3)) -
                           moop2.evaluateGradients(np.ones(3))) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        dsig_dxi = moop1.surrogateUncertainty(xi, grad=True)
        df_dsig = np.zeros((3, 3))
        df_dsig[1, :] = ev_1(xi, np.zeros(3),
                             moop1.surrogateUncertainty(xi), der=3)
        df_dsig[2, :] = ev_2(xi, np.zeros(3),
                             moop1.surrogateUncertainty(xi), der=3)
        if all(moop1.evaluateConstraints(xi) > 0) and \
           all(moop2.evaluateConstraints(xi) > 0):
            assert (np.linalg.norm(moop1.evaluateGradients(xi) -
                                   np.sum(np.dot(df_dsig, dsig_dxi), axis=0) -
                                   moop2.evaluateGradients(xi)) < 1.0e-4)


def test_MOOP_evaluateExpectedValue_named():
    """ Check that the MOOP class handles evaluating objective EVs properly.

    Initialize a MOOP object and check that the evaluateObjectives() function
    works correctly for expected values.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            dx = np.zeros(1, dtype=x.dtype)[0]
            dx["x1"] = 1
            return dx
        elif der == 2:
            return np.zeros(1, dtype=s.dtype)[0]
        else:
            return x["x1"]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)[0]
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s1"] = 2.0 * ev_s["s1"]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=sd_s.dtype)[0]
            dsd["s1"] = 2.0 * sd_s["s1"]
            return dsd
        else:
            return ev_s["s1"] ** 2 + sd_s["s1"] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s2"][0] = 2.0 * ev_s["s2"][0]
            ds["s2"][1] = 2.0 * ev_s["s2"][1]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=ev_s.dtype)[0]
            dsd["s2"][0] = 2.0 * sd_s["s2"][0]
            dsd["s2"][1] = 2.0 * sd_s["s2"][1]
            return dsd
        else:
            return (ev_s["s2"][0] ** 2 + sd_s["s2"][0] ** 2 +
                    ev_s["s2"][1] ** 2 + sd_s["s2"][1] ** 2)

    # Create 2 sims for later

    def sim1(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx)]

    def sim2(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx-1.0), np.linalg.norm(xx-0.5)]

    # Create 2 SimGroups for later
    g1 = {'name': "s1",
          'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'name': "s2",
          'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'name': "f1", 'obj_func': obj_1},
                       {'name': "f2", 'exp_func': ev_1},
                       {'name': "f3", 'exp_func': ev_2})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop1.getDesignType())[0]
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    moop1.fitSurrogates()
    moop1.resetSurrogates(np.ones(3) * 0.5)
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    sds = np.zeros(1, dtype=moop2.getSimulationType())[0]
    moop2.addObjective({'name': "f1", 'obj_func': obj_1},
                       {'name': "f2",
                        'obj_func': lambda x, s, der=0: ev_1(x, s, sds,
                                                             der=der)},
                       {'name': "f3",
                        'obj_func': lambda x, s, der=0: ev_2(x, s, sds,
                                                             der=der)})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop2.getDesignType())[0]
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)
    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluateObjectives(np.zeros(3)) -
                           np.asarray([0.0, 0.0, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.asarray([0.5,
                                                                0.5, 0.5]))
                           - np.asarray([0.5, 0.75, 0.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.asarray([1.0, 0.0,
                                                                0.0]))
                           - np.asarray([1.0, 1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.asarray([0.0, 1.0,
                                                                0.0]))
                           - np.asarray([0.0, 1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.asarray([0.0, 0.0,
                                                                1.0]))
                           - np.asarray([0.0, 1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateObjectives(np.ones(3)) -
                           np.asarray([1.0, 3.0, 0.75])) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        assert (np.all(moop1.evaluateObjectives(xi) >=
                       moop2.evaluateObjectives(xi)))


def test_MOOP_evaluateExpectedValuePenalty_named():
    """ Check that the MOOP class handles evaluating penalized EVs properly.

    Initialize a MOOP object and check that the evaluatePenalty() function
    works correctly for expected values.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            dx = np.zeros(1, dtype=x.dtype)[0]
            dx["x1"] = 1
            return dx
        elif der == 2:
            return np.zeros(1, dtype=s.dtype)[0]
        else:
            return x["x1"]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)[0]
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s1"] = 2.0 * ev_s["s1"]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=sd_s.dtype)[0]
            dsd["s1"] = 2.0 * sd_s["s1"]
            return dsd
        else:
            return ev_s["s1"] ** 2 + sd_s["s1"] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s2"][0] = 2.0 * ev_s["s2"][0]
            ds["s2"][1] = 2.0 * ev_s["s2"][1]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=ev_s.dtype)[0]
            dsd["s2"][0] = 2.0 * sd_s["s2"][0]
            dsd["s2"][1] = 2.0 * sd_s["s2"][1]
            return dsd
        else:
            return (ev_s["s2"][0] ** 2 + sd_s["s2"][0] ** 2 +
                    ev_s["s2"][1] ** 2 + sd_s["s2"][1] ** 2)

    def c2(x, s, der=0): return ev_2(x, s, sds, der=der)

    # Create 2 sims for later

    def sim1(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx)]

    def sim2(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx-1.0), np.linalg.norm(xx-0.5)]

    # Create 2 SimGroups for later
    g1 = {'name': "s1",
          'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'name': "s2",
          'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'name': "f1", 'obj_func': obj_1},
                       {'name': "f2", 'exp_func': ev_1})
    moop1.addConstraint({'name': "f3", 'exp_func': ev_2})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop1.getDesignType())[0]
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    moop1.fitSurrogates()
    moop1.resetSurrogates(np.ones(3) * 0.5)
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    sds = np.zeros(1, dtype=moop2.getSimulationType())[0]
    moop2.addObjective({'name': "f1", 'obj_func': obj_1},
                       {'name': "f2",
                        'obj_func': lambda x, s, der=0: ev_1(x, s, sds,
                                                             der=der)})
    moop2.addConstraint({'name': "f3", 'constraint': c2})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop2.getDesignType())[0]
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)
    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluatePenalty(np.zeros(3)) -
                           np.asarray([3.75, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.asarray([0.5, 0.5, 0.5]))
                           - np.asarray([1.25, 1.5])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.asarray([1.0, 0.0, 0.0]))
                           - np.asarray([3.75, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.asarray([0.0, 1.0, 0.0]))
                           - np.asarray([2.75, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.asarray([0.0, 0.0, 1.0]))
                           - np.asarray([2.75, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluatePenalty(np.ones(3)) -
                           np.asarray([1.75, 3.75])) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        assert (np.all(moop1.evaluatePenalty(xi) >=
                       moop2.evaluatePenalty(xi)))


def test_MOOP_expectedValueGradient_named():
    """ Check that the MOOP class evaluates the expected value gradient.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly for objectives.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            dx = np.zeros(1, dtype=x.dtype)[0]
            dx["x1"] = 1
            return dx
        elif der == 2:
            return np.zeros(1, dtype=s.dtype)[0]
        else:
            return x["x1"]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)[0]
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s1"] = 2.0 * ev_s["s1"]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=sd_s.dtype)[0]
            dsd["s1"] = 2.0 * sd_s["s1"]
            return dsd
        else:
            return ev_s["s1"] ** 2 + sd_s["s1"] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s2"][0] = 2.0 * ev_s["s2"][0]
            ds["s2"][1] = 2.0 * ev_s["s2"][1]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=ev_s.dtype)[0]
            dsd["s2"][0] = 2.0 * sd_s["s2"][0]
            dsd["s2"][1] = 2.0 * sd_s["s2"][1]
            return dsd
        else:
            return (ev_s["s2"][0] ** 2 + sd_s["s2"][0] ** 2 +
                    ev_s["s2"][1] ** 2 + sd_s["s2"][1] ** 2)

    # Create 2 sims for later

    def sim1(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx)]

    def sim2(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx-1.0), np.linalg.norm(xx-0.5)]

    # Create 2 SimGroups for later
    g1 = {'name': "s1",
          'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'name': "s2",
          'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'name': "f1", 'obj_func': obj_1},
                       {'name': "f2", 'exp_func': ev_1},
                       {'name': "f3", 'exp_func': ev_2})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop1.getDesignType())[0]
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    moop1.fitSurrogates()
    moop1.resetSurrogates(np.ones(3) * 0.5)
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    sds = np.zeros(1, dtype=moop2.getSimulationType())[0]
    moop2.addObjective({'name': "f1", 'obj_func': obj_1},
                       {'name': "f2",
                        'obj_func': lambda x, s, der=0: ev_1(x, s, sds,
                                                             der=der)},
                       {'name': "f3",
                        'obj_func': lambda x, s, der=0: ev_2(x, s, sds,
                                                             der=der)})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop2.getDesignType())[0]
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)

    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluateGradients(np.zeros(3)) -
                           moop2.evaluateGradients(np.zeros(3))) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(0.5 * np.ones(3)) -
                           moop2.evaluateGradients(0.5 * np.ones(3)))
            < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[0]) -
                           moop2.evaluateGradients(np.eye(3)[0])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[1]) -
                           moop2.evaluateGradients(np.eye(3)[1])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[2]) -
                           moop2.evaluateGradients(np.eye(3)[2])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.ones(3)) -
                           moop2.evaluateGradients(np.ones(3))) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        xxi = moop1.__extract__(xi)
        sds = moop1.surrogateUncertainty(xi)
        dsig_dxi = moop1.surrogateUncertainty(xi, grad=True)
        df_dsig = np.zeros((3, 3))
        df_dsig_tmp = ev_1(xxi, moop1.__unpack_sim__(np.zeros(3)),
                           moop1.__unpack_sim__(sds), der=3)
        df_dsig[1, :] = moop1.__pack_sim__(df_dsig_tmp)
        df_dsig_tmp = ev_2(xxi, moop1.__unpack_sim__(np.zeros(3)),
                           moop1.__unpack_sim__(sds), der=3)
        df_dsig[2, :] = moop1.__pack_sim__(df_dsig_tmp)
        assert (np.linalg.norm(moop1.evaluateGradients(xi) -
                               np.dot(df_dsig, dsig_dxi) -
                               moop2.evaluateGradients(xi)) < 1.0e-4)


def test_MOOP_evaluateConstraintExpectedValue_named():
    """ Check that the MOOP class handles evaluating constraint EVs properly.

    Initialize a MOOP object and check that the evaluateConstraints() function
    works correctly for expected-values.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            dx = np.zeros(1, dtype=x.dtype)[0]
            dx["x1"] = 1
            return dx
        elif der == 2:
            return np.zeros(1, dtype=s.dtype)[0]
        else:
            return x["x1"]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)[0]
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s1"] = 2.0 * ev_s["s1"]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=sd_s.dtype)[0]
            dsd["s1"] = 2.0 * sd_s["s1"]
            return dsd
        else:
            return ev_s["s1"] ** 2 + sd_s["s1"] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s2"][0] = 2.0 * ev_s["s2"][0]
            ds["s2"][1] = 2.0 * ev_s["s2"][1]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=ev_s.dtype)[0]
            dsd["s2"][0] = 2.0 * sd_s["s2"][0]
            dsd["s2"][1] = 2.0 * sd_s["s2"][1]
            return dsd
        else:
            return (ev_s["s2"][0] ** 2 + sd_s["s2"][0] ** 2 +
                    ev_s["s2"][1] ** 2 + sd_s["s2"][1] ** 2)

    # Create 2 sims for later

    def sim1(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx)]

    def sim2(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx-1.0), np.linalg.norm(xx-0.5)]

    # Create 2 SimGroups for later
    g1 = {'name': "s1",
          'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'name': "s2",
          'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'name': "f1", 'obj_func': obj_1})
    moop1.addConstraint({'name': "c1", 'exp_func': ev_1},
                        {'name': "c2", 'exp_func': ev_2})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop1.getDesignType())[0]
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    moop1.fitSurrogates()
    moop1.resetSurrogates(np.ones(3) * 0.5)
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    sds = np.zeros(1, dtype=moop2.getSimulationType())[0]
    moop2.addObjective({'name': "f1", 'obj_func': obj_1})
    moop2.addConstraint({'name': "c1",
                         'constraint': lambda x, s, der=0: ev_1(x, s, sds,
                                                                der=der)},
                        {'name': "c2",
                         'constraint': lambda x, s, der=0: ev_2(x, s, sds,
                                                                der=der)})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop2.getDesignType())[0]
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)

    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluateConstraints(np.zeros(3)) -
                           np.asarray([0.0, 3.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.asarray([0.5,
                                                                 0.5, 0.5]))
                           - np.asarray([0.75, 0.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.asarray([1.0, 0.0,
                                                                 0.0]))
                           - np.asarray([1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.asarray([0.0, 1.0,
                                                                 0.0]))
                           - np.asarray([1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.asarray([0.0, 0.0,
                                                                 1.0]))
                           - np.asarray([1.0, 2.75])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateConstraints(np.ones(3)) -
                           np.asarray([3.0, 0.75])) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        assert (np.all(moop1.evaluateConstraints(xi) >=
                       moop2.evaluateConstraints(xi)))


def test_MOOP_expectedValueConstraintGradient_named():
    """ Check that the MOOP class evaluates the expected value gradient.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly for constraint expected values.

    """

    from parmoo import MOOP
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalGPS
    import numpy as np

    # Create 3 objectives for later

    def obj_1(x, s, der=0):
        if der == 1:
            dx = np.zeros(1, dtype=x.dtype)[0]
            dx["x1"] = 1
            return dx
        elif der == 2:
            return np.zeros(1, dtype=s.dtype)[0]
        else:
            return x["x1"]

    def ev_1(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)[0]
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s1"] = 2.0 * ev_s["s1"]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=sd_s.dtype)[0]
            dsd["s1"] = 2.0 * sd_s["s1"]
            return dsd
        else:
            return ev_s["s1"] ** 2 + sd_s["s1"] ** 2

    def ev_2(x, ev_s, sd_s, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)
        elif der == 2:
            ds = np.zeros(1, dtype=ev_s.dtype)[0]
            ds["s2"][0] = 2.0 * ev_s["s2"][0]
            ds["s2"][1] = 2.0 * ev_s["s2"][1]
            return ds
        elif der == 3:
            dsd = np.zeros(1, dtype=ev_s.dtype)[0]
            dsd["s2"][0] = 2.0 * sd_s["s2"][0]
            dsd["s2"][1] = 2.0 * sd_s["s2"][1]
            return dsd
        else:
            return (ev_s["s2"][0] ** 2 + sd_s["s2"][0] ** 2 +
                    ev_s["s2"][1] ** 2 + sd_s["s2"][1] ** 2)

    # Create 2 sims for later

    def sim1(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx)]

    def sim2(x):
        xx = np.zeros(3)
        xx[0] = x["x1"]
        xx[1] = x["x2"]
        xx[2] = x["x3"]
        return [np.linalg.norm(xx-1.0), np.linalg.norm(xx-0.5)]

    # Create 2 SimGroups for later
    g1 = {'name': "s1",
          'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'name': "s2",
          'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    # Initialize a MOOP with 2 SimGroups and 3 objectives
    moop1 = MOOP(LocalGPS)
    for i in range(3):
        moop1.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'name': "f1", 'obj_func': obj_1})
    moop1.addConstraint({'name': "c1", 'exp_func': ev_1},
                        {'name': "c2", 'exp_func': ev_2})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop1.getDesignType())[0]
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop1.evaluateSimulation(xi, 0)
    moop1.evaluateSimulation(xi, 1)
    moop1.fitSurrogates()
    moop1.resetSurrogates(np.ones(3) * 0.5)
    # Initialize a second identical MOOP without UQ functions for comparison
    moop2 = MOOP(LocalGPS)
    for i in range(3):
        moop2.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    sds_tt = np.zeros(1, dtype=moop2.getSimulationType())[0]
    moop2.addObjective({'name': "f1", 'obj_func': obj_1})
    moop2.addConstraint({'name': "c1",
                         'constraint': lambda x, s, der=0: ev_1(x, s, sds_tt,
                                                                der=der)},
                        {'name': "c2",
                         'constraint': lambda x, s, der=0: ev_2(x, s, sds_tt,
                                                                der=der)})
    # Evaluate some data points and fit the surrogates
    xi = np.zeros(1, dtype=moop2.getDesignType())[0]
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0.5
    xi['x2'] = 0.5
    xi['x3'] = 0.5
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 0
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 1
    xi['x3'] = 0
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 0
    xi['x2'] = 0
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    xi['x1'] = 1
    xi['x2'] = 1
    xi['x3'] = 1
    moop2.evaluateSimulation(xi, 0)
    moop2.evaluateSimulation(xi, 1)
    moop2.fitSurrogates()
    moop2.resetSurrogates(np.ones(3) * 0.5)

    # Do some objective evaluations and check the results
    assert (np.linalg.norm(moop1.evaluateGradients(np.zeros(3)) -
                           moop2.evaluateGradients(np.zeros(3))) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(0.5 * np.ones(3)) -
                           moop2.evaluateGradients(0.5 * np.ones(3)))
            < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[0]) -
                           moop2.evaluateGradients(np.eye(3)[0])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[1]) -
                           moop2.evaluateGradients(np.eye(3)[1])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.eye(3)[2]) -
                           moop2.evaluateGradients(np.eye(3)[2])) < 1.0e-4)
    assert (np.linalg.norm(moop1.evaluateGradients(np.ones(3)) -
                           moop2.evaluateGradients(np.ones(3))) < 1.0e-4)
    # Try some random evals
    for i in range(10):
        xi = np.random.random_sample(3)
        xxi = moop1.__extract__(xi)
        sds = moop1.surrogateUncertainty(xi)
        dsig_dxi = moop1.surrogateUncertainty(xi, grad=True)
        df_dsig = np.zeros((3, 3))
        df_dsig_tmp = ev_1(xxi, moop1.__unpack_sim__(np.zeros(3)),
                           moop1.__unpack_sim__(sds), der=3)
        df_dsig[1, :] = moop1.__pack_sim__(df_dsig_tmp)
        df_dsig_tmp = ev_2(xxi, moop1.__unpack_sim__(np.zeros(3)),
                           moop1.__unpack_sim__(sds), der=3)
        df_dsig[2, :] = moop1.__pack_sim__(df_dsig_tmp)
        if all(moop1.evaluateConstraints(xi) > 0) and \
           all(moop2.evaluateConstraints(xi) > 0):
            assert (np.linalg.norm(moop1.evaluateGradients(xi) -
                                   np.sum(np.dot(df_dsig, dsig_dxi), axis=0) -
                                   moop2.evaluateGradients(xi)) < 1.0e-4)


if __name__ == "__main__":
    test_MOOP_evaluateExpectedValue_unnamed()
    test_MOOP_evaluateExpectedValuePenalty_unnamed()
    test_MOOP_expectedValueGradient_unnamed()
    test_MOOP_evaluateConstraintExpectedValue_unnamed()
    test_MOOP_expectedValueConstraintGradient_unnamed()
    test_MOOP_evaluateExpectedValue_named()
    test_MOOP_evaluateExpectedValuePenalty_named()
    test_MOOP_expectedValueGradient_named()
    test_MOOP_evaluateConstraintExpectedValue_named()
    test_MOOP_expectedValueConstraintGradient_named()
