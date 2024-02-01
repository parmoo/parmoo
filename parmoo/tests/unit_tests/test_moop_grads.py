from jax import config
config.update("jax_enable_x64", True)
from jax import jacrev
from jax import numpy as jnp

def eval_pen_jac(moop, x):
    """ Helper for testing penalty fwd/bwd evaluations """

    sx = moop._evaluate_surrogates(x)
    dsdx = jacrev(moop._evaluate_surrogates)(x)
    _, res = moop._pen_fwd(x, sx)
    dfdx, dfds = moop._pen_bwd(res, jnp.ones(moop.o))
    return dfdx + jnp.dot(dfds, dsdx)

def eval_obj_jac(moop, x, sx):
    """ Helper for testing objective fwd/bwd evaluations """

    sx = moop._evaluate_surrogates(x)
    dsdx = jacrev(moop._evaluate_surrogates)(x)
    _, res = moop._obj_fwd(x, sx)
    dfdx, dfds = moop._obj_bwd(res, jnp.ones(moop.o))
    return dfdx + jnp.dot(dfds, dsdx)

def eval_con_jac(moop, x, sx):
    """ Helper for testing constraint fwd/bwd evaluations """

    sx = moop._evaluate_surrogates(x)
    dsdx = jacrev(moop._evaluate_surrogates)(x)
    _, res = moop._con_fwd(x, sx)
    dcdx, dcds = moop._con_bwd(res, jnp.ones(moop.o))
    return dcdx + jnp.dot(dcds, dsdx)

def test_MOOP_evaluate_penalty_grads():
    """ Check that the MOOP class handles evaluating gradients properly.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.optimizers import GlobalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Create several differentiable functions and constraints.
    def f1(x, s, der=0):
        names = ["x1", "x2", "x3"]
        if der == 0:
            return np.sum([x[i] * x[i] for i in names])
        if der == 1:
            return {"x1": 2*x["x1"], "x2": 2*x["x2"], "x3": 2*x["x3"]}
        if der == 2:
            return {"sim1": 0, "sim2": np.zeros(2)}

    def f2(x, s, der=0):
        names = ["sim1", "sim2"]
        if der == 0:
            return np.sum([np.dot(s[i] - 0.5, s[i] - 0.5) for i in names])
        if der == 1:
            return {"x1": 0, "x2": 0, "x3": 0}
        if der == 2:
            return {"sim1": 2*s["sim1"] - 1, "sim2": 2*s["sim2"] - np.ones(2)}

    def c1(x, s, der=0):
        if der == 0:
            return x["x1"] - 0.25
        if der == 1:
            return {"x1": 1, "x2": 0, "x3": 0}
        if der == 2:
            return {"sim1": 0, "sim2": np.zeros(2)}

    def c2(x, s, der=0):
        if der == 0:
            return s["sim1"] - 0.25
        if der == 1:
            return {"x1": 0, "x2": 0, "x3": 0}
        if der == 2:
            return {"sim1": 1, "sim2": np.zeros(2)}

    # Initialize a continuous MOOP with 3 variables, no sims, 1 objective
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addObjective({'obj_func': f1})
    # Check the shape and values of the penalty jacobian
    assert (eval_pen_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.zeros(3))) < 1.0e-8))
    fx1 = 2.0 * np.ones((1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    # Add a constraint and make sure that the penalty appears in the jacobian
    moop1.addConstraint({'constraint': c1})
    assert (eval_pen_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.zeros(3))) < 1.0e-8))
    fx1[0, 0] = 3.0
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    # Create a new continuous MOOP as before but with 2 sims
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    g1 = {'m': 1,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: np.sqrt(sum([x[i]**2 for i in x])),
          'surrogate': GaussRBF}
    g2 = {'m': 2,
          'hyperparams': {},
          'search': LatinHypercube,
          'sim_func': lambda x: [np.sqrt(sum([(x[i] - 1.0)**2 for i in x])),
                                 np.sqrt(sum([(x[i] - 0.5)**2 for i in x]))],
          'surrogate': GaussRBF}
    moop1.addSimulation(g1, g2)
    # Add some data and set the surrogates
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.ones(3) * np.infty)
    moop1.addObjective({'obj_func': f1})
    # Check the jacobian outputs with the same test cases as above
    assert (eval_pen_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.zeros(3))) < 1.0e-8))
    fx1 = 2.0 * np.ones((1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    moop1.addConstraint({'constraint': c1})
    assert (eval_pen_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.zeros(3))) < 1.0e-8))
    fx1[0, 0] = 3.0
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    # Now add an additional objective and constraint that depend upon the sim
    fx0 = np.zeros((2, 3))
    fx0[1, 0] = 1.0
    fx0[0, :] = 2.0
    fx0[0, 0] = 3.0
    moop1.addObjective({'obj_func': f2})
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx0) < 1.0e-8))
    moop1.addConstraint({'constraint': c2})
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx0) < 1.0e-8))
    # Create a duplicate MOOP but adjust the design space scaling
    moop2 = MOOP(GlobalSurrogate_PS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': f1})
    moop2.addObjective({'obj_func': f2})
    moop2.addConstraint({'constraint': c1})
    moop2.addConstraint({'constraint': c2})
    for sn in ["sim1", "sim2"]:
        moop2.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop2._fit_surrogates()
    moop2._set_surrogate_tr(np.zeros(3), np.ones(3) * np.infty)
    # After embedding inputs, the outputs should be the same for evaluations
    # at the interpolation nodes...
    x = moop1._embed({'x1': 1, 'x2': 1, 'x3': 1})
    xx = moop2._embed({'x1': 1, 'x2': 1, 'x3': 1})
    assert (np.linalg.norm(eval_pen_jac(moop1, x) - eval_pen_jac(moop2, xx) < 1.0e-8))


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

    # Create several differentiable sims, objecives, and constraints.
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

    def f1(x, s, der=0):
        if der == 0:
            return sum([x[i] * x[i] for i in x])
        if der == 1:
            return {"x1": 2 * x["x1"]}
        if der == 2:
            return {"sim1": 0, "sim2": np.zeros(2)}

    def f2(x, s, der=0):
        if der == 0:
            return sum([np.dot(s[i] - 0.5, s[i] - 0.5) for i in s])
        if der == 1:
            return {"x1": 0}
        if der == 2:
            return {"sim1": 2 * s["sim1"] - 1, "sim2": 2 * s["sim2"] - np.ones(2)}

    def c1(x, s, der=0):
        if der == 0:
            return x["x1"] - 0.25
        if der == 1:
            return {"x1": 1}
        if der == 2:
            return {"sim1": 0, "sim2": np.zeros(2)}

    def c2(x, s, der=0):
        if der == 0:
            return s["sim1"] - 0.25
        if der == 1:
            return {"x1": 0}
        if der == 2:
            return {"sim1": 1, "sim2": np.zeros(2)}

    # Initialize 2 continuous MOOPs with 1 design var, 2 sims, and 3 objs
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
        assert (np.abs(x1[0]["x1"] - x2[0]["x1"]) < 0.1)


if __name__ == "__main__":
    test_MOOP_evaluate_penalty_grads()
    test_MOOP_solve_with_grads()
