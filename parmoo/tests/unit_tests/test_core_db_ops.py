""" Unit tests for MOOP simulation/objective database operations.
"""


import pytest


def test_MOOP_evaluateSimulation():
    """ Check that the MOOP class handles evaluating simulations properly.

    Initialize a MOOP object and check that the evaluateSimulation() function
    works correctly.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Initialize a continuous MOOP with 2 sims and 3 objs
    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'name': "x" + str(i+1), 'lb': 0.0, 'ub': 1.0})
    g1 = {'name': "g1",
          'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.sqrt(sum([x[xi] ** 2 for xi in x]))],
          'surrogate': GaussRBF}
    g2 = {'name': "g2",
          'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [sum([(x[xi]-1)**2 for xi in x])],
          'surrogate': GaussRBF}
    moop.addSimulation(g1, g2)
    moop.addObjective({'obj_func': lambda x, s: s["g1"]})
    moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    x = {"x1": 0, "x2": 0, "x3": 0}
    y = {"x1": 1, "x2": 1, "x3": 1}
    sx = np.zeros(1, dtype=moop.getSimulationType())[0]
    # Check/update database with bad values
    with pytest.raises(ValueError):
        moop.checkSimDb(x, "hello world")
    with pytest.raises(ValueError):
        moop.updateSimDb(x, sx, -1)
    with pytest.raises(ValueError):
        moop.evaluateSimulation(x, "g6")
    # Place 2 items in "g1" DB, 3 in "g2"
    moop.evaluateSimulation(x, "g1")
    moop.evaluateSimulation(y, "g1")
    moop.evaluateSimulation(x, "g2")
    assert (moop.checkSimDb(x, "g1") is not None)
    assert (moop.checkSimDb(y, "g1") is not None)
    assert (moop.checkSimDb(x, "g2") is not None)
    assert (moop.checkSimDb(y, "g2") is None)
    return


def test_MOOP_addObjData():
    """ Check that the MOOP class is able to add data to its internal database.

    Initialize a MOOP object and check that the addObjData(s, sx) function
    works correctly.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    from parmoo.optimizers import LocalSurrogate_PS

    # Initialize a continuous MOOP with 2 sims and 3 objs
    moop1 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    g1 = {'n': 3,
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.sqrt(sum([x[i] ** 2 for i in x]))],
          'surrogate': GaussRBF}
    g2 = {'n': 3,
          'm': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.sqrt(sum([(x[i]-1)**2 for i in x])),
                                 np.sqrt(sum([x[i-0.5]**2 for i in x]))],
          'surrogate': GaussRBF}
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': lambda x, s: s["sim2"][0]})
    moop1.addObjective({'obj_func': lambda x, s: s["sim1"]})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    # Test adding some data
    x0 = moop1._extract(np.zeros(3))
    s0 = moop1._unpack_sim(np.zeros(3))
    x1 = moop1._extract(np.ones(3))
    s1 = moop1._unpack_sim(np.ones(3))
    xe2 = moop1._extract(np.eye(3)[2])
    moop1.addObjData(x0, s0)
    moop1.addObjData(x0, s0)
    moop1.addObjData(x1, s1)
    assert len(moop1.getObjectiveData()) == 2
    # Initialize another continuous MOOP with some constraints
    moop2 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': lambda x, s: s["sim2"][0]})
    moop2.addObjective({'obj_func': lambda x, s: s["sim1"]})
    moop2.addConstraint({'constraint': lambda x, s: x["x1"]})
    moop2.addConstraint({'constraint': lambda x, s: s["sim1"]})
    moop2.addConstraint({'constraint': lambda x, s: sum(s["sim2"])})
    moop2.addAcquisition({'acquisition': UniformWeights})
    moop2.compile()
    # Test adding some data
    moop2.addObjData(x0, s0)
    moop2.addObjData(x0, s0)
    moop2.addObjData(xe2, s0)
    moop2.addObjData(x1, s1)
    assert len(moop2.getObjectiveData()) == 3
    # Initialize another MOOP with mixed variables
    moop3 = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})
    moop3.addDesign({'des_type': "categorical", 'levels': ["L1", "L2", "L3"]})
    moop3.addSimulation(g1, g2)
    moop3.addObjective({'obj_func': lambda x, s: s["sim2"][0]})
    moop3.addObjective({'obj_func': lambda x, s: s["sim1"]})
    moop3.addConstraint({'constraint': lambda x, s: x["x1"]})
    moop3.addConstraint({'constraint': lambda x, s: s["sim1"]})
    moop3.addConstraint({'constraint': lambda x, s: sum(s["sim2"])})
    moop3.addAcquisition({'acquisition': UniformWeights})
    moop3.compile()
    # Test adding some data
    x1 = moop3._extract(np.ones(5))
    moop3.addObjData(x1, s1)
    assert len(moop3.getObjectiveData()) == 1


def test_MOOP_getPF():
    """ Test the getPF function.

    Create several MOOPs, evaluate simulations, and check the final Pareto
    front for correctness.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS

    # Create a toy problem with 4 variables, 3 objectives, 1 constraint
    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})

    def f1(x, s):
        return (np.sqrt(sum([x[f"x{i}"]**2 for i in [2, 3, 4]])
                + (x["x1"] - 1)**2))

    def f2(x, s):
        return (np.sqrt(sum([x[f"x{i}"]**2 for i in [1, 3, 4]])
                + (x["x2"] - 1)**2))

    def f3(x, s):
        return (np.sqrt(sum([x[f"x{i}"]**2 for i in [1, 2, 4]])
                + (x["x3"] - 1)**2))

    def c1(x, s): return -sum([x[i] for i in ["x1", "x2", "x3", "x4"]])
    moop.addObjective({'obj_func': f1})
    moop.addObjective({'obj_func': f2})
    moop.addObjective({'obj_func': f3})
    moop.addConstraint({'constraint': c1})
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    # Directly set the MOOP's database to produce a known Pareto front
    sx = np.zeros(0)
    for data in [
        {"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
        {"x1": 1.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
        {"x1": 0.0, "x2": 1.0, "x3": 0.0, "x4": 0.0},
        {"x1": 0.0, "x2": 0.0, "x3": 1.0, "x4": 0.0},
        {"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 1.0},
    ]:
        data["f1"] = f1(data, sx)
        data["f2"] = f2(data, sx)
        data["f3"] = f3(data, sx)
        data["c1"] = c1(data, sx)
        moop.database.updateObjDb(data, data, data)
    soln = moop.getPF()
    assert (soln.shape[0] == 4)
    assert (soln['f1'].size == 4)
    assert (soln['f2'].size == 4)
    assert (soln['f3'].size == 4)


def test_MOOP_getSimulationData():
    """ Test the getSimulationData function.

    Create several MOOPs, evaluate simulations, and check the simulation
    database.

    """

    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF

    # Create a toy problem with 4 variables, 2 sims
    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'name': ("x" + str(i + 1)), 'lb': 0.0, 'ub': 1.0})
    g1 = {'name': "Bobo1",
          'm': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [sum([x[i]**2 for i in x])],
          'surrogate': GaussRBF}
    g2 = {'name': "Bobo2",
          'm': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [sum([(x[i] - 1)**2 for i in x]),
                                 sum([(x[i] - 0.5)**2 for i in x])],
          'surrogate': GaussRBF}
    moop.addSimulation(g1, g2)
    moop.addObjective({'obj_func': lambda x, s: s["Bobo2"][0]})
    moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    soln = moop.getSimulationData()
    assert (soln['Bobo1']['out'].size == 0)
    assert (soln['Bobo2']['out'].size == 0)
    # Evaluate 5 simulations
    sample_x = {"x1": 0, "x2": 0, "x3": 0, "x4": 0}
    moop.evaluateSimulation(sample_x, "Bobo1")
    moop.evaluateSimulation(sample_x, "Bobo2")
    sample_x["x1"] = 1.0
    moop.evaluateSimulation(sample_x, "Bobo1")
    moop.evaluateSimulation(sample_x, "Bobo2")
    sample_x["x1"] = 0.0
    sample_x["x2"] = 1.0
    moop.evaluateSimulation(sample_x, "Bobo1")
    moop.evaluateSimulation(sample_x, "Bobo2")
    sample_x["x2"] = 0.0
    sample_x["x3"] = 1.0
    moop.evaluateSimulation(sample_x, "Bobo1")
    moop.evaluateSimulation(sample_x, "Bobo2")
    sample_x["x3"] = 0.0
    sample_x["x4"] = 1.0
    moop.evaluateSimulation(sample_x, "Bobo1")
    moop.evaluateSimulation(sample_x, "Bobo2")
    soln = moop.getSimulationData()
    assert (soln['Bobo1']['out'].shape == (5,))
    assert (soln['Bobo2']['out'].shape == (5, 2))


def test_MOOP_getObjectiveData():
    """ Test the getObjectiveData function.

    Create several MOOPs, evaluate simulations, and check the objective
    database.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS

    # Create a toy problem with 4 variables, 3 objectives
    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'name': ('x' + str(i+1)), 'lb': 0.0, 'ub': 1.0})

    def f1(x, s):
        return (np.sqrt(sum([x[f"x{i}"]**2 for i in [2, 3, 4]])
                + (x["x1"] - 1)**2))

    def f2(x, s):
        return (np.sqrt(sum([x[f"x{i}"]**2 for i in [1, 3, 4]])
                + (x["x2"] - 1)**2))

    def f3(x, s):
        return (np.sqrt(sum([x[f"x{i}"]**2 for i in [1, 2, 4]])
                + (x["x3"] - 1)**2))

    def c1(x, s): return -sum([x[f"x{i}"] for i in [1, 2, 3, 4]])
    moop.addObjective({'obj_func': f1})
    moop.addObjective({'obj_func': f2})
    moop.addObjective({'obj_func': f3})
    moop.addConstraint({'constraint': c1})
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    # Directly set the MOOP's database to produce a known output
    sx = np.zeros(0)
    for data in [
        {"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
        {"x1": 1.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
        {"x1": 0.0, "x2": 1.0, "x3": 0.0, "x4": 0.0},
        {"x1": 0.0, "x2": 0.0, "x3": 1.0, "x4": 0.0},
        {"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 1.0},
    ]:
        data["f1"] = f1(data, sx)
        data["f2"] = f2(data, sx)
        data["f3"] = f3(data, sx)
        data["c1"] = c1(data, sx)
        moop.database.updateObjDb(data, data, data)
    soln = moop.getObjectiveData()
    assert (soln.shape[0] == 5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
