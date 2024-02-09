
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
    assert (moop1.data['f_vals'].shape == (2, 2))
    assert (moop1.data['x_vals'].shape == (2, 3))
    assert (moop1.data['c_vals'].shape == (2, 1))
    assert (moop1.n_dat == 2)
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
    assert (moop2.data['f_vals'].shape == (3, 2))
    assert (moop2.data['x_vals'].shape == (3, 3))
    assert (moop2.data['c_vals'].shape == (3, 3))
    assert (moop2.n_dat == 3)
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
    assert (moop3.data['f_vals'].shape == (1, 2))
    assert (moop3.data['x_vals'].shape == (1, 5))
    assert (moop3.data['c_vals'].shape == (1, 3))
    assert (moop3.n_dat == 1)


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
    def f1(x, s): return np.sqrt(sum([x[f"x{i}"]**2 for i in [2, 3, 4]]) + (x["x1"] - 1)**2)
    def f2(x, s): return np.sqrt(sum([x[f"x{i}"]**2 for i in [1, 3, 4]]) + (x["x2"] - 1)**2)
    def f3(x, s): return np.sqrt(sum([x[f"x{i}"]**2 for i in [1, 2, 4]]) + (x["x3"] - 1)**2)
    def c1(x, s): return -sum([x[i] for i in ["x1", "x2", "x3", "x4"]])
    moop.addObjective({'obj_func': f1})
    moop.addObjective({'obj_func': f2})
    moop.addObjective({'obj_func': f3})
    moop.addConstraint({'constraint': c1})
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    # Directly set the MOOP's database to produce a known Pareto front
    moop.data = {'x_vals': np.zeros((5, 4)),
                 'f_vals': np.zeros((5, 3)),
                 'c_vals': np.zeros((5, 1))}
    sx = np.zeros(0)
    moop.data['x_vals'][0, :] = np.array([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 0.0]), sx)
    moop.data['x_vals'][1, :] = np.array([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop._evaluate_objectives(
                                   np.array([1.0, 0.0, 0.0, 0.0]), sx)
    moop.data['x_vals'][2, :] = np.array([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop._evaluate_objectives(
                                   np.array([0.0, 1.0, 0.0, 0.0]), sx)
    moop.data['x_vals'][3, :] = np.array([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 1.0, 0.0]), sx)
    moop.data['x_vals'][4, :] = np.array([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 1.0]), sx)
    moop.n_dat = 5
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

    import numpy as np
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
    def f1(x, s): return np.sqrt(sum([x[f"x{i}"]**2 for i in [2, 3, 4]]) + (x["x1"] - 1)**2)
    def f2(x, s): return np.sqrt(sum([x[f"x{i}"]**2 for i in [1, 3, 4]]) + (x["x2"] - 1)**2)
    def f3(x, s): return np.sqrt(sum([x[f"x{i}"]**2 for i in [1, 2, 4]]) + (x["x3"] - 1)**2)
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
    moop.data = {'x_vals': np.zeros((5, 4)),
                 'f_vals': np.zeros((5, 3)),
                 'c_vals': np.zeros((5, 1))}
    moop.data['x_vals'][0, :] = np.array([0.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][0, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 0.0]), sx)
    moop.data['c_vals'][0, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 0.0, 0.0]), sx)
    moop.data['x_vals'][1, :] = np.array([1.0, 0.0, 0.0, 0.0])
    moop.data['f_vals'][1, :] = moop._evaluate_objectives(
                                   np.array([1.0, 0.0, 0.0, 0.0]), sx)
    moop.data['c_vals'][1, :] = moop._evaluate_constraints(
                                   np.array([1.0, 0.0, 0.0, 0.0]), sx)
    moop.data['x_vals'][2, :] = np.array([0.0, 1.0, 0.0, 0.0])
    moop.data['f_vals'][2, :] = moop._evaluate_objectives(
                                   np.array([0.0, 1.0, 0.0, 0.0]), sx)
    moop.data['c_vals'][2, :] = moop._evaluate_constraints(
                                   np.array([0.0, 1.0, 0.0, 0.0]), sx)
    moop.data['x_vals'][3, :] = np.array([0.0, 0.0, 1.0, 0.0])
    moop.data['f_vals'][3, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 1.0, 0.0]), sx)
    moop.data['c_vals'][3, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 1.0, 0.0]), sx)
    moop.data['x_vals'][4, :] = np.array([0.0, 0.0, 0.0, 1.0])
    moop.data['f_vals'][4, :] = moop._evaluate_objectives(
                                   np.array([0.0, 0.0, 0.0, 1.0]), sx)
    moop.data['c_vals'][4, :] = moop._evaluate_constraints(
                                   np.array([0.0, 0.0, 0.0, 1.0]), sx)
    moop.n_dat = 5
    soln = moop.getObjectiveData()
    assert (soln.shape[0] == 5)


def test_MOOP_save_load_functions():
    """ Check that a MOOP object can be correctly saved/reloaded.

    Create and save a MOOP object, then reload and check that it is the same.

    """

    import numpy as np
    import os
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Functions sim1, sim2, f1, f2, c1 need to be global for save/load to work
    global sim1, sim2, f1, f2, c1
    def sim1(x): return [np.sqrt(sum([x[i]**2 for i in x]))]
    def sim2(x): return [np.sqrt(sum([(x[i] - 1)**2 for i in x]))]
    def f1(x, sim): return sim["sim1"]
    def f2(x, sim): return sim["sim2"]
    def c1(x, sim): return x["x1"] - 0.5
    # Create a MOOP with 3 variables, 2 sims, 2 objs, and 1 constraint
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    # Empty save
    moop1.save()
    # Add MOOP variables, sims, objectives, etc.
    for i in range(2):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addDesign({'des_type': "categorical", 'levels': 3})
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 25,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    moop1.addSimulation(g1, g2)
    # Add 2 objectives
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    # Add 1 constraint
    moop1.addConstraint({'constraint': c1})
    # Add 3 acquisition functions
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    batch = moop1.iterate(0)
    batch = moop1.filterBatch(batch)
    for (xi, i) in batch:
        moop1.evaluateSimulation(xi, i)
    moop1.updateAll(0, batch)
    # Test save
    moop1.save()
    # Test load
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.load()
    # Check that save/load are correct
    check_moops(moop1, moop2)
    # Create a new MOOP with same specs
    moop3 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    for i in range(2):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})
    moop3.addDesign({'des_type': "categorical", 'levels': 3})
    moop3.addSimulation(g1, g2)
    moop3.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    moop3.addConstraint({'constraint': c1})
    for i in range(3):
        moop3.addAcquisition({'acquisition': UniformWeights})
    moop3.compile()
    # Try to save and overwrite old data
    with pytest.raises(OSError):
        moop3.save()
    # Save a data point with moop1
    moop1.savedata(np.zeros(1, dtype=moop3.getDesignType())[0],
                   np.zeros(1), "sim1")
    # Try to overwrite with moop3
    with pytest.raises(OSError):
        moop3.savedata(np.zeros(1, dtype=moop3.getDesignType())[0],
                       np.zeros(1), "sim1")
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.simdb.json")
    os.remove("parmoo.surrogate.1")
    os.remove("parmoo.surrogate.2")
    os.remove("parmoo.optimizer")


def test_MOOP_save_load_classes():
    """ Check that a MOOP object can be correctly saved/reloaded.

    Create and save a MOOP object, then reload and check that it is the same.

    Use simulation/objective callable objects from the library.

    """

    import os
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.constraints import single_sim_bound
    from parmoo.objectives import SingleSimObjective
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.simulations.dtlz import dtlz2_sim

    # Initialize the simulation group with 3 outputs
    f1 = SingleSimObjective(3, 2, 0)
    f2 = SingleSimObjective(3, 2, 1)
    c1 = single_sim_bound(3, 2, 1)
    # Create a mixed-variable MOOP with 3 variables, 2 sims, 2 objs, 1 const
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    # Test empty save
    moop1.save()
    for i in range(2):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addDesign({'des_type': "categorical", 'levels': 3})
    g1 = {'m': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': dtlz2_sim(moop1.getDesignType(), num_obj=2),
          'surrogate': GaussRBF}
    moop1.addSimulation(g1)
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    moop1.addConstraint({'constraint': c1})
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    # Test save and reload
    moop1.save()
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.load()
    check_moops(moop1, moop2)
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.surrogate.1")
    os.remove("parmoo.optimizer")


def test_MOOP_checkpoint():
    """ Check that the MOOP object performs checkpointing correctly.

    Run 1 iteration of ParMOO, with checkpointing on.

    """

    import numpy as np
    import os
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF

    # Functions sim1, sim2, f1, f2, c1 need to be global for save/load to work
    global sim1, sim2, f1, f2, c1
    def sim1(x): return [np.sqrt(sum([x[i] ** 2 for i in x]))]
    def sim2(x): return [np.sqrt(sum([(x[i] - 1) ** 2 for i in x]))]
    def f1(x, sim): return sim["sim1"]
    def f2(x, sim): return sim["sim2"]
    def c1(x, sim): return x["x1"] - 0.5
    # Create a mixed-variable MOOP with 3 variables, 2 sims, 3 objs, 1 const
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    for i in range(2):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addDesign({'des_type': "categorical", 'levels': 3})
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': sim1,
          'surrogate': GaussRBF}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 25,
          'sim_func': sim2,
          'surrogate': GaussRBF}
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    moop1.addConstraint({'constraint': c1})
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    # Turn on checkpointing
    moop1.setCheckpoint(True)
    # One iteration
    batch = moop1.iterate(0)
    batch = moop1.filterBatch(batch)
    for (xi, i) in batch:
        moop1.evaluateSimulation(xi, i)
    moop1.updateAll(0, batch)
    # Test load
    moop2 = MOOP(LocalSurrogate_PS)
    moop2.load()
    check_moops(moop1, moop2)
    # Clean up test directory
    os.remove("parmoo.moop")
    os.remove("parmoo.simdb.json")
    os.remove("parmoo.surrogate.1")
    os.remove("parmoo.surrogate.2")
    os.remove("parmoo.optimizer")


def check_moops(moop1, moop2):
    """ Auxiliary function for checking that 2 moops are equal.

    Check that all entries in moop1 = moop2

    Args:
        moop1 (MOOP): First moop to compare

        moop2 (MOOP): Second moop to compare

    """

    import numpy as np

    # Check scalars
    assert (moop2.m == moop1.m and
            moop2.n_feature == moop1.n_feature and
            moop2.n_latent == moop1.n_latent and
            moop2.o == moop1.o and moop2.p == moop1.p and
            moop2.s == moop1.s and moop2.n_dat == moop1.n_dat and
            moop2.lam == moop1.lam and
            moop2.iteration == moop1.iteration)
    # Check lists
    assert (all([dt2i == dt1i for dt2i, dt1i in zip(moop2.latent_des_tols,
                                                    moop1.latent_des_tols)]))
    assert (all([lb2i == lb1i for lb2i, lb1i in zip(moop2.latent_lb,
                                                    moop1.latent_lb)]))
    assert (all([ub2i == ub1i for ub2i, ub1i in zip(moop2.latent_ub,
                                                    moop1.latent_ub)]))
    assert (all([m2i == m1i for m2i, m1i in zip(moop2.m_list, moop1.m_list)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.sim_schema,
                                                      moop1.sim_schema)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.des_schema,
                                                      moop1.des_schema)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.obj_schema,
                                                      moop1.obj_schema)]))
    assert (all([n2i[0] == n1i[0] for n2i, n1i in zip(moop2.con_schema,
                                                      moop1.con_schema)]))
    # Check dictionaries
    assert (all([moop2.data[ki].shape == moop1.data[ki].shape
                 for ki in moop2.data.keys()]))
    assert (all([all([moop2.sim_db[j][ki].shape == moop1.sim_db[j][ki].shape
                      for ki in ["x_vals", "s_vals"]])
                 for j in range(len(moop1.sim_db))]))
    for obj1, obj2 in zip(moop1.obj_funcs, moop2.obj_funcs):
        if hasattr(obj1, "__name__"):
            assert (obj1.__name__ == obj2.__name__)
        else:
            assert (obj1.__class__.__name__ == obj2.__class__.__name__)
    for sim1, sim2 in zip(moop1.sim_funcs, moop2.sim_funcs):
        if hasattr(sim1, "__name__"):
            assert (sim1.__name__ == sim2.__name__)
        else:
            assert (sim1.__class__.__name__ == sim2.__class__.__name__)
    for const1, const2 in zip(moop1.con_funcs, moop2.con_funcs):
        if hasattr(const1, "__name__"):
            assert (const1.__name__ == const2.__name__)
        else:
            assert (const1.__class__.__name__ == const2.__class__.__name__)
    # Check functions
    assert (moop2.optimizer.__class__.__name__ ==
            moop1.optimizer.__class__.__name__)
    assert (all([s1.__class__.__name__ == s2.__class__.__name__
                 for s1, s2 in zip(moop1.searches, moop2.searches)]))
    assert (all([s1.__class__.__name__ == s2.__class__.__name__
                 for s1, s2 in zip(moop1.surrogates, moop2.surrogates)]))
    assert (all([s1.__class__.__name__ == s2.__class__.__name__
                 for s1, s2 in zip(moop1.acquisitions, moop2.acquisitions)]))


if __name__ == "__main__":
    test_MOOP_evaluateSimulation()
    test_MOOP_addObjData()
    test_MOOP_getPF()
    test_MOOP_getSimulationData()
    test_MOOP_getObjectiveData()
    test_MOOP_save_load_functions()
    test_MOOP_save_load_classes()
    test_MOOP_checkpoint()
