
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
                       {'exp_func': lambda x, ev_s, sd_s: ev_s[0]**2 + \
                                                          sd_s[0]**2},
                       {'exp_func': lambda x, ev_s, sd_s: ev_s[1]**2 + \
                                                          sd_s[1]**2 + \
                                                          ev_s[2]**2 + \
                                                          sd_s[2]**2})
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


if __name__ == "__main__":
    test_MOOP_evaluateExpectedValue()
