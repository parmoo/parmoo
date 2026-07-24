""" Unit tests for the MOOP.solve() driver, with and without gradients.
"""

import pytest
from jax import numpy as jnp


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
                                         s2['sim_func'](soln[i])]).flatten() -
                               np.array([soln['f1'][i], soln['f2'][i]]))
                < 1.e-8)
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


def test_MOOP_solve_with_grads():
    """ Check that the MOOP class propagates gradients correctly to solvers.

    Initialize a simple convex MOOP and check that the gradient-based solver
    matches the gradient-free solver's solutions.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import FixedWeights
    from parmoo.optimizers import GlobalSurrogate_BFGS, GlobalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF

    # Create several differentiable simulation groups
    g1 = {'m': 1,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.sqrt(sum([x[i]**2 for i in x]))],
          'surrogate': GaussRBF,
          'hyperparams': {'search_budget': 100}}
    g2 = {'m': 2,
          'search': LatinHypercube,
          'sim_func': lambda x: [np.sqrt(sum([(x[i]-0.5)**2 for i in x])),
                                 np.sqrt(sum([(x[i]-1.0)**2 for i in x]))],
          'surrogate': GaussRBF,
          'hyperparams': {'search_budget': 100}}

    # Create several differentiable functions and constraints
    def f1(x, s):
        return x["x1"] ** 2

    def df1(x, s):
        return ({"x1": 2*x["x1"]},
                {"sim1": 0, "sim2": jnp.zeros(2)})

    def f2(x, s):
        names = ["sim1", "sim2"]
        return np.sum([jnp.dot(s[i] - 0.5, s[i] - 0.5) for i in names])

    def df2(x, s):
        return ({"x1": 0},
                {"sim1": 2*s["sim1"] - 1, "sim2": 2*s["sim2"] - jnp.ones(2)})

    def c1(x, s):
        return x["x1"] - 0.25

    def dc1(x, s):
        return {"x1": 1}, {"sim1": 0, "sim2": jnp.zeros(2)}

    def c2(x, s):
        return s["sim1"] - 0.25

    def dc2(x, s):
        return {"x1": 0}, {"sim1": 1, "sim2": jnp.zeros(2)}

    # Initialize 2 continuous MOOPs with 1 design var, 2 sims, and 3 objs
    moop1 = MOOP(GlobalSurrogate_BFGS, hyperparams={'opt_restarts': 2,
                                                    'np_random_gen': 0})
    moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addObjective({'obj_func': f2, 'obj_grad': df2})
    moop1.addConstraint({'constraint': c1, 'con_grad': dc1})
    moop1.addConstraint({'constraint': c2, 'con_grad': dc2})
    moop1.addAcquisition({'acquisition': FixedWeights,
                          'hyperparams': {'weights': np.ones(2) / 2}})
    moop1.solve(0)
    moop2 = MOOP(GlobalSurrogate_PS, hyperparams={'np_random_gen': 0})
    moop2.addDesign({'lb': 0.0, 'ub': 1.0})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': f1})
    moop2.addObjective({'obj_func': f2})
    moop2.addConstraint({'constraint': c1})
    moop2.addConstraint({'constraint': c2})
    moop2.addAcquisition({'acquisition': FixedWeights,
                          'hyperparams': {'weights': np.ones(2) / 2}})
    moop2.solve(0)
    b1 = moop1.iterate(1)
    b2 = moop2.iterate(1)
    # Check that same solutions were found
    for x1, x2 in zip(b1, b2):
        assert (np.abs(x1[0]["x1"] - x2[0]["x1"]) < 0.1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
