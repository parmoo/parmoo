
def test_MOOP_iterate():
    """ Test the MOOP class's iterator in objectives.py.

    Initialize several MOOP objects and perform iterations to produce
    a batch of candidate solutions.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo import MOOP
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_PS, LocalSurrogate_BFGS
    import pytest

    # Initialize two simulation groups with 1 output each
    s1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[f"x{key}"]
                                 for key in [1, 2, 3]])],
          'surrogate': GaussRBF,
          'search_budget': 20}
    s2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[f"x{key}"] - 1
                                 for key in [1, 2, 3]])],
          'surrogate': GaussRBF,
          'search_budget': 20}
    # Create a MOOP with 3 design variables and 2 simulations
    moop1 = MOOP(LocalSurrogate_PS,
                 hyperparams={'opt_budget': 100, 'np_random_gen': 0})
    with pytest.raises(AttributeError):
        moop1.iterate(1)
    for i in range(3):
        moop1.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(s1, s2)
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
        assert (np.abs(s1['sim_func'](si) - si['f1']) < 1.0e-8)
        assert (np.linalg.norm(s2['sim_func'](si) - si['f2']) < 1.0e-8)

    s3 = {'m': 4,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [x[i] for i in x],
          'surrogate': GaussRBF,
          'search_budget': 500}
    # Create a three objective toy problem, with one simulation
    moop2 = MOOP(LocalSurrogate_BFGS,
                 hyperparams={'opt_budget': 100, 'np_random_gen': 0})
    for i in range(4):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0, 'des_tol': 0.1})
    moop2.addSimulation(s3)

    # Now add the three objectives
    def f3(x, sim):
        return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[0, :]) ** 2.0

    def g3(x, sim):
        x_out = x.copy()
        for key in x:
            x_out[key] = 0.0
        s_out = sim.copy()
        s_out["sim1"] *= 2.0
        s_out["sim1"] = s_out["sim1"].at[0].set(s_out["sim1"][0] - 0.2)
        return x_out, s_out

    def f4(x, sim):
        return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[1, :]) ** 2.0

    def g4(x, sim):
        x_out = x.copy()
        for key in x:
            x_out[key] = 0.0
        s_out = sim.copy()
        s_out["sim1"] *= 2.0
        s_out["sim1"] = s_out["sim1"].at[1].set(s_out["sim1"][1] - 0.2)
        return x_out, s_out

    def f5(x, sim):
        return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[2, :]) ** 2.0

    def g5(x, sim):
        x_out = x.copy()
        for key in x:
            x_out[key] = 0.0
        s_out = sim.copy()
        s_out["sim1"] *= 2.0
        s_out["sim1"] = s_out["sim1"].at[2].set(s_out["sim1"][2] - 0.2)
        return x_out, s_out

    moop2.addObjective({'obj_func': f3, 'obj_grad': g3},
                       {'obj_func': f4, 'obj_grad': g4},
                       {'obj_func': f5, 'obj_grad': g5})
    # Add 3 acquisition functions
    for i in range(3):
        moop2.addAcquisition({'acquisition': UniformWeights})
    # Do 2 iterates of the MOOP and extract the final database
    moop2.iterate(0)
    batch = []
    xi = {"x1": 0, "x2": 0, "x3": 0, "x4": 0}
    for i in range(1, 5):
        xi[f"x{i}"] = 0.1
        batch.append((xi.copy(), "sim1"))
        xi[f"x{i}"] = 0
    batch.append(({"x1": 0.1, "x2": 0.1, "x3": 0.1, "x4": 0.1}, "sim1"))
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
    si = {}
    for xsi in soln:
        xi = {"x1": xsi["x1"],
              "x2": xsi["x2"],
              "x3": xsi["x3"],
              "x4": xsi["x4"]}
        si["sim1"] = s3["sim_func"](xi)
        fi = [xsi["f1"], xsi["f2"], xsi["f3"]]
        assert (np.linalg.norm(np.array([f3(xi, si), f4(xi, si), f5(xi, si)]
                                        ).flatten() - fi) < 1.0e-8)
        for j in xi:
            assert (xi[j] <= 0.2)

    s4 = {'m': 4,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: abs(x["x5"] - 1) + np.array([x["x1"], x["x2"],
                                                             x["x3"], x["x4"]]),
          'surrogate': GaussRBF,
          'search_budget': 500}
    # Create a three objective toy problem, with one simulation
    moop3 = MOOP(LocalSurrogate_BFGS, hyperparams={'np_random_gen': 0})
    for i in range(4):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})
    moop3.addDesign({'des_type': "categorical", 'levels': 3})
    moop3.addSimulation(s4)

    # Now add the three objectives
    def f6(x, sim):
        return np.linalg.norm(sim["sim1"] - np.eye(4)[0, :]) ** 2

    def g6(x, sim):
        dx = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "x5": 0}
        ds = {"sim1": np.array([2.0 * sim["sim1"][0] - 2.0,
                                2.0 * sim["sim1"][1],
                                2.0 * sim["sim1"][2],
                                2.0 * sim["sim1"][3]])}
        return dx, ds

    def f7(x, sim):
        return np.linalg.norm(sim["sim1"] - np.eye(4)[1, :]) ** 2

    def g7(x, sim):
        dx = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "x5": 0}
        ds = {"sim1": np.array([2.0 * sim["sim1"][0],
                                2.0 * sim["sim1"][1] - 2.0,
                                2.0 * sim["sim1"][2],
                                2.0 * sim["sim1"][3]])}
        return dx, ds

    def f8(x, sim):
        return np.linalg.norm(sim["sim1"] - np.eye(4)[2, :]) ** 2

    def g8(x, sim):
        dx = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "x5": 0}
        ds = {"sim1": np.array([2.0 * sim["sim1"][0],
                                2.0 * sim["sim1"][1],
                                2.0 * sim["sim1"][2] - 2.0,
                                2.0 * sim["sim1"][3]])}
        return dx, ds

    moop3.addObjective({'obj_func': f6, 'obj_grad': g6},
                       {'obj_func': f7, 'obj_grad': g7},
                       {'obj_func': f8, 'obj_grad': g8})
    # Add 3 acquisition functions
    for i in range(3):
        moop3.addAcquisition({'acquisition': UniformWeights})
    # Do 2 iterates of the MOOP and extract the final database
    moop3.iterate(0)
    batch = []
    for i in range(1, 6):
        xi = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "x5": 0}
        xi[f"x{i}"] = 1
        batch.append((xi.copy(), "sim1"))
    batch.append(({"x1": 1, "x2": 1, "x3": 1, "x4": 1, "x5": 1}, "sim1"))
    batch.append(({"x1": 1, "x2": 1, "x3": 1, "x4": 1, "x5": 2}, "sim1"))
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
    assert (np.size(soln) > 0)
    # Assert that the x_vals and f_vals match
    si = {}
    for xsi in soln:
        xi = {"x1": xsi["x1"],
              "x2": xsi["x2"],
              "x3": xsi["x3"],
              "x4": xsi["x4"],
              "x5": xsi["x5"]}
        si["sim1"] = s4["sim_func"](xi)
        fi = [xsi["f1"], xsi["f2"], xsi["f3"]]
        assert (np.linalg.norm(np.array([f6(xi, si), f7(xi, si), f8(xi, si)]
                                        ).flatten() - fi) < 1.0e-8)
        assert (abs(xi["x4"]) <= 0.1 and abs(xi["x5"] - 1) <= 0.1)

    s5 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [(x["x0"] - 1.0) * (x["x0"] - 1.0) +
                                 (x["x1"]) * (x["x1"]) + float(x["x2"])],
          'surrogate': GaussRBF,
          'search_budget': 100}
    # Solve a MOOP with categorical variables
    moop4 = MOOP(LocalSurrogate_BFGS, hyperparams={'np_random_gen': 0})
    moop4.addDesign({'name': "x0", 'lb': 0.0, 'ub': 1.0})
    moop4.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1.0})
    moop4.addDesign({'name': "x2", 'des_type': "categorical",
                     'levels': ["0", "1"]})
    moop4.addSimulation(s5)

    # Now add the two objectives
    def f9(x, sim):
        return sim["sim1"]

    def g9(x, sim):
        dx = {"x0": 0, "x1": 0}
        ds = {"sim1": 1}
        return dx, ds

    def f10(x, sim):
        return ((x["x0"]) * (x["x0"]) +
                (x["x1"] - 1.0) * (x["x1"] - 1.0) + float(x["x2"]))

    def g10(x, sim):
        dx = {}
        dx['x0'] = 2.0 * x["x0"]
        dx['x1'] = 2.0 * x["x1"] - 2.0
        ds = {"sim1": 0}
        return dx, ds

    moop4.addObjective({'obj_func': f9, 'obj_grad': g9},
                       {'obj_func': f10, 'obj_grad': g10})
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
    sim = {}
    for i, xi in enumerate(soln):
        sim["sim1"] = ((xi["x0"] - 1.0) * (xi["x0"] - 1.0) +
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
    s1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[f"x{key}"]
                                                 for key in [1, 2, 3, 4]])],
          'surrogate': GaussRBF,
          'search_budget': 20}
    s2 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.linalg.norm([x[f"x{key}"] - 1
                                                 for key in [1, 2, 3, 4]])],
          'surrogate': GaussRBF,
          'search_budget': 20}
    # Create a MOOP with 4 design variables and 2 simulations
    moop1 = MOOP(LocalSurrogate_PS, hyperparams={'opt_budget': 100,
                                                 'np_random_gen': 0})
    for i in range(4):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(s1, s2)
    # Now add 2 objectives
    def f1(x, sim): return sim["sim1"]
    def f2(x, sim): return sim["sim2"]
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
    soln = moop1.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    for i in range(soln.shape[0]):
        assert (np.linalg.norm(np.array([s1['sim_func'](soln[i]),
                                         s2['sim_func'](soln[i])]
                                        ).flatten() -
                                        np.array([soln['f1'][i],
                                                  soln['f2'][i]]))
               < 0.00000001)
    # Create new single objective toy problem
    s3 = {'m': 1,
          'sim_func': lambda x: [x["x1"] + x["x2"]],
          'surrogate': GaussRBF,
          'search': LatinHypercube,
          'hyperparams': {'search_budget': 10}}
    s4 = {'m': 1,
          'sim_func': lambda x: [x["x3"] + x["x4"]],
          'surrogate': GaussRBF,
          'search': LatinHypercube,
          'hyperparams': {'search_budget': 20}}
    moop2 = MOOP(LocalSurrogate_PS, hyperparams={'np_random_gen': 0})
    for i in range(4):
        moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(s3, s4)
    # Now add 1 objective
    def f3(x, sim): return sim["sim1"][0] + sim["sim2"][0]
    moop2.addObjective({'obj_func': f3})
    # Add 3 acquisition functions
    for i in range(3):
        moop2.addAcquisition({'acquisition': RandomConstraint})
    # Solve the MOOP and extract the final database with 6 iterations
    moop2.solve(6)
    soln = moop2.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    for i in range(soln.shape[0]):
        assert (np.linalg.norm(np.array(s3['sim_func'](soln[i])) +
                               np.array(s4['sim_func'](soln[i])) -
                               soln['f1'][i]) < 0.00000001)

    # Create a 3 objective toy problem, with no simulations
    moop3 = MOOP(GlobalSurrogate_BFGS, hyperparams={'np_random_gen': 0})
    for i in range(4):
        moop3.addDesign({'lb': 0.0, 'ub': 1.0})

    # Now add the three objectives
    def f4(x, sim):
        return np.linalg.norm([x["x1"] - 1, x["x2"], x["x3"], x["x4"]]) ** 2.0

    def g4(x, sim):
        dx = {"x1": 2 * x["x1"] - 2,
              "x2": 2 * x["x2"],
              "x3": 2 * x["x3"],
              "x4": 2 * x["x4"]}
        return dx, {}

    def f5(x, sim):
        return np.linalg.norm([x["x1"], x["x2"] - 1, x["x3"], x["x4"]]) ** 2.0

    def g5(x, sim):
        dx = {"x1": 2 * x["x1"],
              "x2": 2 * x["x2"] - 2,
              "x3": 2 * x["x3"],
              "x4": 2 * x["x4"]}
        return dx, {}

    def f6(x, sim):
        return np.linalg.norm([x["x1"], x["x2"], x["x3"] - 1, x["x4"]]) ** 2.0

    def g6(x, sim):
        dx = {"x1": 2 * x["x1"],
              "x2": 2 * x["x2"],
              "x3": 2 * x["x3"] - 2,
              "x4": 2 * x["x4"]}
        return dx, {}

    moop3.addObjective({'obj_func': f4, 'obj_grad': g4},
                       {'obj_func': f5, 'obj_grad': g5},
                       {'obj_func': f6, 'obj_grad': g6})
    # Add 3 acquisition functions
    for i in range(3):
        moop3.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop3.solve(6)
    soln = moop3.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(0)
    for i in range(soln.shape[0]):
        assert (np.linalg.norm(np.array([f4(soln[i], sim),
                                         f5(soln[i], sim),
                                         f6(soln[i], sim)]
                                        ).flatten() -
                               np.array([soln[i]["f1"], soln[i]["f2"],
                                         soln[i]["f3"]])) < 0.00000001)

    # Create a 3 objective toy problem, with no simulations and 1 categorical
    moop4 = MOOP(GlobalSurrogate_BFGS, hyperparams={'np_random_gen': 0})
    for i in range(3):
        moop4.addDesign({'lb': 0.0, 'ub': 1.0})
    moop4.addDesign({'des_type': "categorical", 'levels': 3})
    moop4.addObjective({'obj_func': f4, 'obj_grad': g4},
                       {'obj_func': f5, 'obj_grad': g5},
                       {'obj_func': f6, 'obj_grad': g6})
    # Add 3 acquisition functions
    for i in range(3):
        moop4.addAcquisition({'acquisition': UniformWeights})
    # Solve the MOOP and extract the final database with 6 iterations
    moop4.solve(6)
    soln = moop4.getPF()
    # Assert that solutions were found
    assert (soln.size > 0)
    # Assert that the x_vals and f_vals match
    sim = np.zeros(0)
    for i in range(soln.shape[0]):
        assert (np.linalg.norm(np.array([f4(soln[i], sim),
                                         f5(soln[i], sim),
                                         f6(soln[i], sim)]
                                        ).flatten() -
                               np.array([soln[i]["f1"], soln[i]["f2"],
                                         soln[i]["f3"]])) < 0.00000001)


if __name__ == "__main__":
    test_MOOP_iterate()
    test_MOOP_solve()
