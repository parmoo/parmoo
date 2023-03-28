
def test_MOOP_evaluateExpectedValue():
    """ Check that the MOOP class handles evaluating objectives properly.

    Initialize a MOOP object and check that the evaluateObjectives() function
    works correctly.

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


def test_MOOP_expectedValueGradient():
    """ Check that the MOOP class evaluates the expected value gradient.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly.

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


if __name__ == "__main__":
    test_MOOP_evaluateExpectedValue()
    test_MOOP_expectedValueGradient()
