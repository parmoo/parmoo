from jax import config
config.update("jax_enable_x64", True)
from jax import jacrev
from jax import numpy as jnp

def eval_pen_jac(moop, x):
    """ Helper for testing penalty fwd/bwd evaluations """

    sx = moop._evaluate_surrogates(x)
    dsdx = jacrev(moop._evaluate_surrogates)(x)
    _, res = moop._pen_fwd(x, sx)
    dfdx = jnp.zeros((moop.o, moop.n_latent))
    for i, ei in enumerate(jnp.eye(moop.o)):
        dfdxi, dfdsi = moop._pen_bwd(res, ei)
        dfdx = dfdx.at[i].set(dfdxi + jnp.dot(dfdsi, dsdx))
    return dfdx

def eval_obj_jac(moop, x):
    """ Helper for testing objective fwd/bwd evaluations """

    sx = moop._evaluate_surrogates(x)
    dsdx = jacrev(moop._evaluate_surrogates)(x)
    _, res = moop._obj_fwd(x, sx)
    dfdx = jnp.zeros((moop.o, moop.n_latent))
    for i, ei in enumerate(jnp.eye(moop.o)):
        dfdxi, dfdsi = moop._obj_bwd(res, ei)
        dfdx = dfdx.at[i].set(dfdxi + jnp.dot(dfdsi, dsdx))
    return dfdx

def eval_con_jac(moop, x):
    """ Helper for testing constraint fwd/bwd evaluations """

    sx = moop._evaluate_surrogates(x)
    dsdx = jacrev(moop._evaluate_surrogates)(x)
    _, res = moop._con_fwd(x, sx)
    dcdx = jnp.zeros((moop.p, moop.n_latent))
    for i, ei in enumerate(jnp.eye(moop.p)):
        dcdxi, dcdsi = moop._con_bwd(res, ei)
        dcdx = dcdx.at[i].set(dcdxi + jnp.dot(dcdsi, dsdx))
    return dcdx

def test_MOOP_evaluate_penalty_grads():
    """ Check that the MOOP class handles evaluating gradients properly.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import GlobalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Create several differentiable functions and constraints.
    def f1(x, s):
        names = ["x1", "x2", "x3"]
        return np.sum([x[i] * x[i] for i in names])

    def df1(x, s):
        return ({"x1": 2*x["x1"], "x2": 2*x["x2"], "x3": 2*x["x3"]},
                {"sim1": 0, "sim2": jnp.zeros(2)})

    def f2(x, s):
        names = ["sim1", "sim2"]
        return np.sum([jnp.dot(s[i] - 0.5, s[i] - 0.5) for i in names])

    def df2(x, s):
        return ({"x1": 0, "x2": 0, "x3": 0},
                {"sim1": 2*s["sim1"] - 1, "sim2": 2*s["sim2"] - jnp.ones(2)})

    def c1(x, s):
        return x["x1"] - 0.25

    def dc1(x, s):
        return {"x1": 1, "x2": 0, "x3": 0}, {"sim1": 0, "sim2": jnp.zeros(2)}

    def c2(x, s):
        return s["sim1"] - 0.25

    def dc2(x, s):
        return {"x1": 0, "x2": 0, "x3": 0}, {"sim1": 1, "sim2": jnp.zeros(2)}

    # Initialize a continuous MOOP with 3 variables, no sims, 1 objective
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    # Check the shape and values of the penalty Jacobian
    assert (eval_pen_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.zeros(3))) < 1.0e-8))
    fx1 = 2.0 * np.ones((1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    # Add a constraint and make sure that the penalty appears in the Jacobian
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop1.compile()
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
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    # Add some data and set the surrogates
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    # Check the Jacobian outputs with the same test cases as above
    assert (eval_pen_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.zeros(3))) < 1.0e-8))
    fx1 = 2.0 * np.ones((1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    # Re-define the MOOP but add a constraint
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    assert (eval_pen_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_pen_jac(moop1, np.zeros(3))) < 1.0e-8))
    fx1[0, 0] = 3.0
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    # Re-define the MOOP but add objectives and constraints that use the sim
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addObjective({'obj_func': f2, 'obj_grad': df2})
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop1.addConstraint({'con_func': c2, 'con_grad': dc2})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    fx0 = np.zeros((2, 3))
    fx0[1, 0] = 1.0
    fx0[0, :] = 2.0
    fx0[0, 0] = 3.0
    assert (np.all(np.abs(eval_pen_jac(moop1, np.ones(3)) - fx0) < 1.0e-8))
    # Create a duplicate MOOP but adjust the design space scaling
    moop2 = MOOP(GlobalSurrogate_PS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop2.addObjective({'obj_func': f2, 'obj_grad': df2})
    moop2.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop2.addConstraint({'con_func': c2, 'con_grad': dc2})
    moop2.addAcquisition({'acquisition': UniformWeights})
    moop2.compile()
    for sn in ["sim1", "sim2"]:
        moop2.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop2._fit_surrogates()
    moop2._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    # After embedding inputs, the outputs should be the same for evaluations
    # at the interpolation nodes.
    x = moop1._embed({'x1': 1, 'x2': 1, 'x3': 1})
    xx = moop2._embed({'x1': 1, 'x2': 1, 'x3': 1})
    assert (np.linalg.norm(eval_pen_jac(moop1, x) - eval_pen_jac(moop2, xx) < 1.0e-8))
    # Now check that after compiling, jax correctly propagates pen_jac
    _, _, eval_pen1 = moop1._link()
    def pen_jac(x):
        return eval_pen1(x, moop1._evaluate_surrogates(x))
    moop1_pen_jac = jacrev(pen_jac)
    for xi in np.random.sample((5, 3)):
        dfdxi = moop1_pen_jac(xi)
        assert (np.all(np.abs(eval_pen_jac(moop1, xi) - dfdxi) < 1.0e-8))


def test_MOOP_evaluate_objective_grads():
    """ Check that the MOOP class handles evaluating gradients properly.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import GlobalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Create several differentiable functions and constraints.
    def f1(x, s):
        names = ["x1", "x2", "x3"]
        return np.sum([x[i] * x[i] for i in names])

    def df1(x, s):
        return ({"x1": 2*x["x1"], "x2": 2*x["x2"], "x3": 2*x["x3"]},
                {"sim1": 0, "sim2": jnp.zeros(2)})

    def f2(x, s):
        names = ["sim1", "sim2"]
        return np.sum([jnp.dot(s[i] - 0.5, s[i] - 0.5) for i in names])

    def df2(x, s):
        return ({"x1": 0, "x2": 0, "x3": 0},
                {"sim1": 2*s["sim1"] - 1, "sim2": 2*s["sim2"] - jnp.ones(2)})

    def c1(x, s):
        return x["x1"] - 0.25

    def dc1(x, s):
        return {"x1": 1, "x2": 0, "x3": 0}, {"sim1": 0, "sim2": jnp.zeros(2)}

    def c2(x, s):
        return s["sim1"] - 0.25

    def dc2(x, s):
        return {"x1": 0, "x2": 0, "x3": 0}, {"sim1": 1, "sim2": jnp.zeros(2)}

    # Initialize a continuous MOOP with 3 variables, no sims, 1 objective
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    # Check the shape and values of the objective Jacobian
    assert (eval_obj_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_obj_jac(moop1, np.zeros(3))) < 1.0e-8))
    fx1 = 2.0 * np.ones((1, 3))
    assert (np.all(np.abs(eval_obj_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    # Add a constraint and make sure that the Jacobian is unchanged
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop1.compile()
    assert (eval_obj_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_obj_jac(moop1, np.zeros(3))) < 1.0e-8))
    assert (np.all(np.abs(eval_obj_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
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
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    # Add some data and set the surrogates
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    # Check the Jacobian outputs with the same test cases as above
    assert (eval_obj_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_obj_jac(moop1, np.zeros(3))) < 1.0e-8))
    fx1 = 2.0 * np.ones((1, 3))
    assert (np.all(np.abs(eval_obj_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    # Re-define the MOOP but add a constraint
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    assert (eval_obj_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_obj_jac(moop1, np.zeros(3))) < 1.0e-8))
    assert (np.all(np.abs(eval_obj_jac(moop1, np.ones(3)) - fx1) < 1.0e-8))
    # Re-define the MOOP but add objectives and constraints that use the sim
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addObjective({'obj_func': f2, 'obj_grad': df2})
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop1.addConstraint({'con_func': c2, 'con_grad': dc2})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    fx0 = np.zeros((2, 3))
    fx0[0, :] = 2.0
    assert (np.all(np.abs(eval_obj_jac(moop1, np.ones(3)) - fx0) < 1.0e-8))
    # Create a duplicate MOOP but adjust the design space scaling
    moop2 = MOOP(GlobalSurrogate_PS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop2.addObjective({'obj_func': f2, 'obj_grad': df2})
    moop2.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop2.addConstraint({'con_func': c2, 'con_grad': dc2})
    moop2.addAcquisition({'acquisition': UniformWeights})
    moop2.compile()
    for sn in ["sim1", "sim2"]:
        moop2.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop2._fit_surrogates()
    moop2._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    # After embedding inputs, the outputs should be the same for evaluations
    # at the interpolation nodes.
    x = moop1._embed({'x1': 1, 'x2': 1, 'x3': 1})
    xx = moop2._embed({'x1': 1, 'x2': 1, 'x3': 1})
    assert (np.linalg.norm(eval_obj_jac(moop1, x) - eval_obj_jac(moop2, xx) < 1.0e-8))
    eval_obj1, _, _ = moop1._link()
    # Now check that after compiling, jax correctly propagates obj_jac
    def obj_jac(x):
        return eval_obj1(x, moop1._evaluate_surrogates(x))
    moop1_obj_jac = jacrev(obj_jac)
    for xi in np.random.sample((5, 3)):
        dfdxi = moop1_obj_jac(xi)
        assert (np.all(np.abs(eval_obj_jac(moop1, xi) - dfdxi) < 1.0e-8))


def test_MOOP_evaluate_constraint_grads():
    """ Check that the MOOP class handles evaluating gradients properly.

    Initialize a MOOP object and check that the evaluateGradients() function
    works correctly.

    """

    import numpy as np
    from parmoo import MOOP
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import GlobalSurrogate_PS
    from parmoo.searches import LatinHypercube
    from parmoo.surrogates import GaussRBF
    import pytest

    # Create several differentiable functions and constraints.
    def f1(x, s):
        names = ["x1", "x2", "x3"]
        return np.sum([x[i] * x[i] for i in names])

    def df1(x, s):
        return ({"x1": 2*x["x1"], "x2": 2*x["x2"], "x3": 2*x["x3"]},
                {"sim1": 0, "sim2": jnp.zeros(2)})

    def f2(x, s):
        names = ["sim1", "sim2"]
        return np.sum([jnp.dot(s[i] - 0.5, s[i] - 0.5) for i in names])

    def df2(x, s):
        return ({"x1": 0, "x2": 0, "x3": 0},
                {"sim1": 2*s["sim1"] - 1, "sim2": 2*s["sim2"] - jnp.ones(2)})

    def c1(x, s):
        return x["x1"] - 0.25

    def dc1(x, s):
        return {"x1": 1, "x2": 0, "x3": 0}, {"sim1": 0, "sim2": jnp.zeros(2)}

    def c2(x, s):
        return s["sim1"] - 0.25

    def dc2(x, s):
        return {"x1": 0, "x2": 0, "x3": 0}, {"sim1": 1, "sim2": jnp.zeros(2)}

    # Initialize a continuous MOOP with 3 variables, no sims, 1 objective
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    # Check the shape and values of the constraint Jacobian
    assert (eval_con_jac(moop1, np.zeros(3)).size == 0)
    # Add a constraint and make sure it appears in the Jacobian
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop1.compile()
    assert (eval_con_jac(moop1, np.zeros(3)).shape == (1, 3))
    cx1 = np.zeros((1, 3))
    cx1[0, 0] = 1.0
    assert (np.all(np.abs(eval_con_jac(moop1, np.zeros(3)) - cx1) < 1.0e-8))
    assert (np.all(np.abs(eval_con_jac(moop1, np.ones(3)) - cx1) < 1.0e-8))
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
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    assert (eval_con_jac(moop1, np.zeros(3)).shape == (1, 3))
    assert (np.all(np.abs(eval_con_jac(moop1, np.zeros(3)) - cx1) < 1.0e-8))
    assert (np.all(np.abs(eval_con_jac(moop1, np.ones(3)) - cx1) < 1.0e-8))
    # Re-define the MOOP but add objectives and constraints that use the sim
    moop1 = MOOP(GlobalSurrogate_PS)
    for i in range(3):
        moop1.addDesign({'lb': 0.0, 'ub': 1.0})
    moop1.addSimulation(g1, g2)
    moop1.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop1.addObjective({'obj_func': f2, 'obj_grad': df2})
    moop1.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop1.addConstraint({'con_func': c2, 'con_grad': dc2})
    moop1.addAcquisition({'acquisition': UniformWeights})
    moop1.compile()
    for sn in ["sim1", "sim2"]:
        moop1.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop1._fit_surrogates()
    moop1._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    cx2 = np.zeros((2, 3))
    cx2[0, 0] = 1.0
    assert (np.all(np.abs(eval_con_jac(moop1, np.ones(3)) - cx2) < 1.0e-8))
    # Create a duplicate MOOP but adjust the design space scaling
    moop2 = MOOP(GlobalSurrogate_PS)
    moop2.addDesign({'lb': -1.0, 'ub': 1.0},
                    {'lb': 0.0, 'ub': 2.0},
                    {'lb': -0.5, 'ub': 1.5})
    moop2.addSimulation(g1, g2)
    moop2.addObjective({'obj_func': f1, 'obj_grad': df1})
    moop2.addObjective({'obj_func': f2, 'obj_grad': df2})
    moop2.addConstraint({'con_func': c1, 'con_grad': dc1})
    moop2.addConstraint({'con_func': c2, 'con_grad': dc2})
    moop2.addAcquisition({'acquisition': UniformWeights})
    moop2.compile()
    for sn in ["sim1", "sim2"]:
        moop2.evaluateSimulation({"x1": 1, "x2": 1, "x3": 1}, sn)
    moop2._fit_surrogates()
    moop2._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    # After embedding inputs, the outputs should be the same for evaluations
    # at the interpolation nodes.
    x = moop1._embed({'x1': 1, 'x2': 1, 'x3': 1})
    xx = moop2._embed({'x1': 1, 'x2': 1, 'x3': 1})
    assert (np.linalg.norm(eval_con_jac(moop1, x) - eval_con_jac(moop2, xx) < 1.0e-8))
    _, eval_con1, _ = moop1._link()
    # Now check that after compiling, jax correctly propagates con_jac
    def con_jac(x):
        return eval_con1(x, moop1._evaluate_surrogates(x))
    moop1_con_jac = jacrev(con_jac)
    for xi in np.random.sample((5, 3)):
        dfdxi = moop1_con_jac(xi)
        assert (np.all(np.abs(eval_con_jac(moop1, xi) - dfdxi) < 1.0e-8))


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
    test_MOOP_evaluate_penalty_grads()
    test_MOOP_evaluate_objective_grads()
    test_MOOP_evaluate_constraint_grads()
    test_MOOP_solve_with_grads()
