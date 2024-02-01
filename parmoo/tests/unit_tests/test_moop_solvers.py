
def test_MOOP_iterate():
    """ Test the MOOP class's iterator in objectives.py.

    Initialize several MOOP objects and perform iterations to produce
    a batch of candidate solutions.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS, LocalSurrogate_BFGS
    import numpy as np
    import pytest

    # Initialize two simulation groups with 1 output each
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[str(key)] for key in [0, 1, 2]])],
          'surrogate': GaussRBF,
          'search_budget': 20}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[str(key)] - 1 for key in [0, 1, 2]])],
          'surrogate': GaussRBF,
          'search_budget': 20}
    # Create a MOOP with 3 design variables and 2 simulations
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    with pytest.raises(AttributeError):
        moop1.iterate(1)
    for i in range(3):
        moop1.addDesign({'name': str(i), 'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    with pytest.raises(AttributeError):
        moop1.iterate(1)
    # Now add the two objectives
    def f1(x, sim): return sim["sim1"]
    def f2(x, sim): return sim["sim2"]
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    # Add 3 acquisition functions
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    # Try some invalid iterations
    with pytest.raises(ValueError):
        moop1.iterate(-1)
    with pytest.raises(TypeError):
        moop1.iterate(2.0)
    # Solve the MOOP with 1 iteration
    batch = moop1.iterate(0)
    batch = moop1.filterBatch(batch)
    for (x, i) in batch:
        moop1.evaluateSimulation(x, i)
    moop1.updateAll(0, batch)
    batch = moop1.iterate(1)
    batch = moop1.filterBatch(batch)
    for (x, i) in batch:
        moop1.evaluateSimulation(x, i)
    moop1.updateAll(1, batch)
    soln = moop1.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    for si in soln:
        assert (np.abs(g1['sim_func'](si) - si['f1']) < 1.0e-8)
        assert (np.linalg.norm(g2['sim_func'](si) - si['f2']) < 1.0e-8)

    g3 = {'m': 4,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [x[i] for i in range(4)],
          'surrogate': GaussRBF,
          'search_budget': 500}
    # Create a three objective toy problem, with one simulation
    moop2 = MOOP(LocalSurrogate_BFGS, hyperparams={'opt_budget': 100})
    for i in range(4):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0, 'des_tol': 0.1})
    moop2.addSimulation(g3)

    # Now add the three objectives
    def f3(x, sim, der=0):
        if der == 1:
            x_out = x.copy()
            for key in x:
                x_out[key] = 0.0
            return x_out
        elif der == 2:
            s_out = sim.copy()
            s_out["sim1"] *= 2.0
            s_out["sim1"][0] -= 0.2
            return s_out
        else:
            return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[0, :]) ** 2.0

    def f4(x, sim, der=0):
        if der == 1:
            x_out = x.copy()
            for key in x:
                x_out[key] = 0.0
            return x_out
        elif der == 2:
            s_out = sim.copy()
            s_out["sim1"] *= 2.0
            s_out["sim1"][1] -= 0.2
            return s_out
        else:
            return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[1, :]) ** 2.0

    def f5(x, sim, der=0):
        if der == 1:
            x_out = x.copy()
            for key in x:
                x_out[key] = 0.0
            return x_out
        elif der == 2:
            s_out = sim.copy()
            s_out["sim1"] *= 2.0
            s_out["sim1"][2] -= 0.2
            return s_out
        else:
            return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[2, :]) ** 2.0

    moop2.addObjective({'obj_func': f3},
                       {'obj_func': f4},
                       {'obj_func': f5})
    # Add 3 acquisition functions
    for i in range(3):
        moop2.addAcquisition({'acquisition': UniformWeights})
    # Do 2 iterates of the MOOP and extract the final database
    moop2.iterate(0)
    batch = [(0.1 * xi, "sim1") for xi in np.eye(4)]
    batch.append((0.1 * np.ones(4), "sim1"))
    for (x, i) in batch:
        moop2.evaluateSimulation(x, i)
    moop2.updateAll(0, batch)
    batch = moop2.iterate(1)
    batch = moop2.filterBatch(batch)
    for (x, i) in batch:
        moop2.evaluateSimulation(x, i)
    moop2.updateAll(1, batch)
    soln = moop2.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(4)
    for i in range(np.shape(soln['x_vals'])[0]):
        sim = soln['x_vals'][i]
        assert (np.linalg.norm(np.array([f3(soln['x_vals'][i], sim),
                                         f4(soln['x_vals'][i], sim),
                                         f5(soln['x_vals'][i], sim)]
                                        ).flatten()
                               - soln['f_vals'][i])
                < 0.00000001)
        assert (all(soln['x_vals'][i, :4] <= 0.2))

    g4 = {'m': 4,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: x[0:4] + abs(x[4] - 1.0),
          'surrogate': GaussRBF,
          'search_budget': 500}
    # Create a three objective toy problem, with one simulation
    moop3 = MOOP(LocalSurrogate_BFGS, hyperparams={})
    for i in range(4):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})
    moop3.addDesign({'des_type': "categorical", 'levels': 3})
    moop3.addSimulation(g4)

    # Now add the three objectives
    def f6(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.array([2.0 * sim[0] - 2.0,
                             2.0 * sim[1],
                             2.0 * sim[2],
                             2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - np.eye(4)[0, :]) ** 2.0

    def f7(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.array([2.0 * sim[0],
                             2.0 * sim[1] - 2.0,
                             2.0 * sim[2],
                             2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - np.eye(4)[1, :]) ** 2.0

    def f8(x, sim, der=0):
        if der == 1:
            return np.zeros(x.size)
        elif der == 2:
            return np.array([2.0 * sim[0],
                             2.0 * sim[1],
                             2.0 * sim[2] - 2.0,
                             2.0 * sim[3]])
        else:
            return np.linalg.norm(sim - np.eye(4)[2, :]) ** 2.0

    moop3.addObjective({'obj_func': f6},
                       {'obj_func': f7},
                       {'obj_func': f8})
    # Add 3 acquisition functions
    for i in range(3):
        moop3.addAcquisition({'acquisition': UniformWeights})
    # Do 2 iterates of the MOOP and extract the final database
    moop3.iterate(0)
    batch = [(xi, 0) for xi in np.eye(5)]
    batch.append((np.ones(5), 0))
    for (x, i) in batch:
        moop3.evaluateSimulation(x, i)
    moop3.updateAll(0, batch)
    batch = moop3.iterate(1)
    batch = moop3.filterBatch(batch)
    for (x, i) in batch:
        moop3.evaluateSimulation(x, i)
    moop3.updateAll(1, batch)
    soln = moop3.getPF()
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(4)
    for i in range(np.shape(soln['x_vals'])[0]):
        sim = soln['x_vals'][i, :4] - abs(soln['x_vals'][i, 4] - 1.0)
        assert (np.linalg.norm(np.array([f6(soln['x_vals'][i], sim),
                                         f7(soln['x_vals'][i], sim),
                                         f8(soln['x_vals'][i], sim)]
                                        ).flatten()
                               - soln['f_vals'][i])
                < 0.00000001)
        assert (soln['x_vals'][i, 3] <= 0.1 and soln['x_vals'][i, 4] == 1.0)

    x_entry = np.zeros(1, dtype=np.dtype([("x0", float), ("x1", float),
                                          ("x2", object)]))
    x_entry[0]["x2"] = "0"
    g5 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [(x["x0"] - 1.0) * (x["x0"] - 1.0) +
                                 (x["x1"]) * (x["x1"]) + float(x["x2"])],
          'surrogate': GaussRBF,
          'search_budget': 100}
    # Solve a MOOP with categorical variables
    moop4 = MOOP(LocalSurrogate_BFGS, hyperparams={})
    moop4.addDesign({'name': "x0", 'lb': 0.0, 'ub': 1.0})
    moop4.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1.0})
    moop4.addDesign({'name': "x2", 'des_type': "categorical",
                     'levels': ["0", "1"]})
    moop4.addSimulation(g5)

    # Now add the two objectives
    def f9(x, sim, der=0):
        if der == 1:
            return np.zeros(1, dtype=x.dtype)[0]
        elif der == 2:
            result = np.ones(1, dtype=sim.dtype)
            return result[0]
        else:
            return sim[0]

    def f10(x, sim, der=0):
        if der == 1:
            out = np.zeros(1, dtype=x.dtype)
            out['x0'] = 2.0 * x["x0"]
            out['x1'] = 2.0 * x["x1"] - 2.0
            out['x2'] = 0.0
            return out[0]
        elif der == 2:
            return np.zeros(1, dtype=sim.dtype)[0]
        else:
            return ((x["x0"]) * (x["x0"]) +
                    (x["x1"] - 1.0) * (x["x1"] - 1.0) + float(x["x2"]))

    moop4.addObjective({'obj_func': f9},
                       {'obj_func': f10})
    # Add 3 acquisition functions
    for i in range(3):
        moop4.addAcquisition({'acquisition': UniformWeights})
    # Do 2 iterates of the MOOP and extract the final database
    batch = moop4.iterate(0)
    batch = moop4.filterBatch(batch)
    for (x, i) in batch:
        moop4.evaluateSimulation(x, i)
    moop4.updateAll(0, batch)
    batch = moop4.iterate(1)
    batch = moop4.filterBatch(batch)
    for (x, i) in batch:
        moop4.evaluateSimulation(x, i)
    moop4.updateAll(1, batch)
    soln = moop4.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(1)
    for i, xi in enumerate(soln):
        sim[0] = ((xi["x0"] - 1.0) * (xi["x0"] - 1.0) +
                  (xi["x1"]) * (xi["x1"]) + float(xi["x2"]))
        assert (f9(soln[i], sim) - soln['f1'][i] < 1.0e-8 and
                f10(soln[i], sim) - soln['f2'][i] < 1.0e-8)
        assert (xi["x2"] == "0")


def test_MOOP_solve():
    """ Test the MOOP class's solver in objectives.py.

    Perform a test of the MOOP solver class by minimizing a 5 variable,
    biobjective convex function s.t. $x in [0, 1]^n$.

    The correctness of the solutions is difficult to assert , but we can
    assert  that the efficient points map onto the Pareto front, as
    expected.

    """

    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights, RandomConstraint
    from parmoo.optimizers import LocalSurrogate_PS, GlobalSurrogate_BFGS
    import numpy as np
    import pytest

    # Initialize two simulation groups with 1 output each
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 20,
          'sim_func': lambda x: [np.linalg.norm(x)],
          'surrogate': GaussRBF}
    g2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'search_budget': 25,
          'sim_func': lambda x: [np.linalg.norm(x-1.0)],
          'surrogate': GaussRBF}
    # Create a MOOP with 4 design variables and 2 simulations
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100})
    for i in range(4):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    # Now add 2 objectives
    def f1(x, sim): return sim[0]
    def f2(x, sim): return sim[1]
    moop1.addObjective({'obj_func': f1},
                       {'obj_func': f2})
    # Add 3 acquisition functions
    for i in range(3):
        moop1.addAcquisition({'acquisition': UniformWeights})
    # Try to solve several invalid problems/budgets to test error handling
    with pytest.raises(ValueError):
        moop1.solve(-1)
    with pytest.raises(TypeError):
        moop1.solve(2.0)
    # Solve the MOOP with 6 iterations
    moop1.solve(6)
    soln = moop1.data
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    for i in range(np.shape(soln['x_vals'])[0]):
        assert (np.linalg.norm(np.array([g1['sim_func'](soln['x_vals'][i]),
                                         g2['sim_func'](soln['x_vals'][i])]
                                        ).flatten() - soln['f_vals'][i])
               < 0.00000001)
    # Create new single objective toy problem
    g3 = {'m': 1,
          'sim_func': lambda x: [x[0] + x[1]],
          'surrogate': GaussRBF,
          'search': LatinHypercube,
          'hyperparams': {'search_budget': 10}}
    g4 = {'m': 1,
          'sim_func': lambda x: [x[2] + x[3]],
          'surrogate': GaussRBF,
          'search': LatinHypercube,
          'hyperparams': {'search_budget': 20}}
    moop2 = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g3, g4)
    # Now add 1 objective
    def f3(x, sim): return sim[0] + sim[1]
    moop2.addObjective({'obj_func': f3})
    # Add 3 acquisition functions
    for i in range(3):
        moop2.addAcquisition({'acquisition': RandomConstraint})
    # Solve the MOOP and extract the final database with 6 iterations
    moop2.solve(6)
    soln = moop2.data
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    for i in range(np.shape(soln['x_vals'])[0]):
        assert (np.linalg.norm(np.array(g3['sim_func'](soln['x_vals'][i])) +
                               np.array(g4['sim_func'](soln['x_vals'][i])) -
                               soln['f_vals'][i]) < 0.00000001)

    # Create a 3 objective toy problem, with no simulations
    moop3 = MOOP(GlobalSurrogate_BFGS, hyperparams={})
    for i in range(4):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})

    # Now add the three objectives
    def f4(x, sim, der=0):
        if der == 1:
            return np.array([2.0 * x[0] - 2.0,
                             2.0 * x[1],
                             2.0 * x[2],
                             2.0 * x[3]])
        elif der == 2:
            return np.zeros(sim.size)
        else:
            return np.linalg.norm(x - np.eye(x.size)[0, :]) ** 2.0

    def f5(x, sim, der=0):
        if der == 1:
            return np.array([2.0 * x[0],
                             2.0 * x[1] - 2.0,
                             2.0 * x[2],
                             2.0 * x[3]])
        elif der == 2:
            return np.zeros(sim.size)
        else:
            return np.linalg.norm(x - np.eye(x.size)[1, :]) ** 2.0

    def f6(x, sim, der=0):
        if der == 1:
            return np.array([2.0 * x[0],
                             2.0 * x[1],
                             2.0 * x[2] - 2.0,
                             2.0 * x[3],
                             0.0])
        elif der == 2:
            return np.zeros(sim.size)
        else:
            return np.linalg.norm(x - np.eye(x.size)[2, :]) ** 2.0

    moop3.addObjective({'obj_func': f4},
                       {'obj_func': f5},
                       {'obj_func': f6})
    # Add 3 acquisition functions
    for i in range(3):
        moop3.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop3.solve(6)
    soln = moop3.data
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(0)
    for i in range(np.shape(soln['x_vals'])[0]):
        assert (np.linalg.norm(np.array([f4(soln['x_vals'][i], sim),
                                         f5(soln['x_vals'][i], sim),
                                         f6(soln['x_vals'][i], sim)]
                                        ).flatten()
                               - soln['f_vals'][i])
                < 0.00000001)

    # Create a 3 objective toy problem, with no simulations and 1 categorical
    moop4 = MOOP(GlobalSurrogate_BFGS, hyperparams={})
    for i in range(3):
        moop4.addDesign({'lb': 0.0, 'ub': 1.0})
    moop4.addDesign({'des_type': "categorical", 'levels': 3})
    moop4.addObjective({'obj_func': f4},
                       {'obj_func': f5},
                       {'obj_func': f6})
    # Add 3 acquisition functions
    for i in range(3):
        moop4.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop4.solve(6)
    soln = moop4.getPF()
    # Assert that solutions were found
    assert (np.size(soln['x_vals']) > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(0)
    for i in range(np.shape(soln['x_vals'])[0]):
        assert (np.linalg.norm(np.array([f4(soln['x_vals'][i], sim),
                                        f5(soln['x_vals'][i], sim),
                                         f6(soln['x_vals'][i], sim)]
                                        ).flatten()
                               - soln['f_vals'][i])
                < 0.00000001)


if __name__ == "__main__":
    test_MOOP_iterate()
    test_MOOP_solve()
