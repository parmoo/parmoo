""" Unit tests for the MOOP.solve() driver, with and without gradients.

Correctness of the returned solutions is hard to assert directly, so these
tests assert the property that must hold regardless of what the solver
converges to: every point on the returned Pareto front must have objective
values consistent with re-evaluating the objectives at its own design point.

"""

import numpy as np
import pytest
from jax import numpy as jnp

from parmoo import MOOP
from parmoo.acquisitions import FixedWeights, RandomConstraint, UniformWeights
from parmoo.optimizers import (
    GlobalSurrogate_BFGS,
    GlobalSurrogate_PS,
    LocalSurrogate_PS,
)
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.tests.unit_tests.helpers import seeded

ITERATIONS = 6


def sim_dict(m, sim_func, search_budget=None):
    """ Build a simulation dict with the budget in the hyperparams. """

    hyperparams = {} if search_budget is None \
        else {'search_budget': search_budget}
    return {'m': m,
            'search': LatinHypercube,
            'sim_func': sim_func,
            'surrogate': GaussRBF,
            'hyperparams': hyperparams}


def assert_front_is_self_consistent(soln, funcs, sim=None):
    """ Check that each front point's objectives match its own design point.

    Args:
        soln (ndarray): The structured array returned by getPF().

        funcs (list of callable): The objective functions, in schema order.

        sim (any, optional): The simulation argument to pass to each
            objective.  Defaults to an empty array, for simulation-free
            problems.

    """

    if sim is None:
        sim = np.zeros(0)
    for i in range(soln.shape[0]):
        expected = np.array([f(soln[i], sim) for f in funcs]).flatten()
        got = np.array([soln[f"f{j + 1}"][i] for j in range(len(funcs))])
        assert (np.linalg.norm(expected - got) < 1.0e-8)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget, error", [(-1, ValueError), (2.0, TypeError)])
def test_solve_rejects_bad_budget(budget, error):
    """ Check that solve() validates its iteration budget. """

    moop = MOOP(LocalSurrogate_PS, hyperparams=seeded(opt_budget=100))
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addObjective({'obj_func': obj_x1_squared})
    moop.addAcquisition({'acquisition': UniformWeights})
    with pytest.raises(error):
        moop.solve(budget)


# ---------------------------------------------------------------------------
# Problem 1: two simulations, two objectives, no gradients
# ---------------------------------------------------------------------------


def sim_norm_4(x):
    """ The norm of (x1, ..., x4). """

    return [np.linalg.norm([x[f"x{k}"] for k in [1, 2, 3, 4]])]


def sim_norm_4_shifted(x):
    """ The norm of (x1, ..., x4) shifted by one. """

    return [np.linalg.norm([x[f"x{k}"] - 1 for k in [1, 2, 3, 4]])]


def obj_sim1(x, sim):
    return sim["sim1"]


def obj_sim2(x, sim):
    return sim["sim2"]


def test_solve_two_simulation_biobjective():
    """ Check solve() on a biobjective problem with one sim per objective. """

    moop = MOOP(LocalSurrogate_PS, hyperparams=seeded(opt_budget=100))
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(sim_dict(1, sim_norm_4),
                       sim_dict(1, sim_norm_4_shifted))
    moop.addObjective({'obj_func': obj_sim1}, {'obj_func': obj_sim2})
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    moop.solve(ITERATIONS)

    soln = moop.getPF()
    assert (soln.size > 0)
    for i in range(soln.shape[0]):
        expected = np.array([sim_norm_4(soln[i]),
                             sim_norm_4_shifted(soln[i])]).flatten()
        got = np.array([soln['f1'][i], soln['f2'][i]])
        assert (np.linalg.norm(expected - got) < 1.0e-8)


# ---------------------------------------------------------------------------
# Problem 2: one objective summing two simulations, epsilon-constraint
# acquisitions
# ---------------------------------------------------------------------------


def sim_x1_plus_x2(x):
    return [x["x1"] + x["x2"]]


def sim_x3_plus_x4(x):
    return [x["x3"] + x["x4"]]


def obj_sum_of_sims(x, sim):
    return sim["sim1"][0] + sim["sim2"][0]


def test_solve_single_objective_two_simulations():
    """ Check solve() on a single-objective problem with two simulations.

    A single objective degenerates the Pareto front to one point, and the
    RandomConstraint acquisitions must cope with that.

    """

    moop = MOOP(LocalSurrogate_PS, hyperparams=seeded())
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(sim_dict(1, sim_x1_plus_x2, 10),
                       sim_dict(1, sim_x3_plus_x4, 20))
    moop.addObjective({'obj_func': obj_sum_of_sims})
    for i in range(3):
        moop.addAcquisition({'acquisition': RandomConstraint})
    moop.solve(ITERATIONS)

    soln = moop.getPF()
    assert (soln.size > 0)
    for i in range(soln.shape[0]):
        expected = (np.array(sim_x1_plus_x2(soln[i])) +
                    np.array(sim_x3_plus_x4(soln[i])))
        assert (np.linalg.norm(expected - soln['f1'][i]) < 1.0e-8)


# ---------------------------------------------------------------------------
# Problems 3 and 4: three simulation-free objectives, with user gradients
# ---------------------------------------------------------------------------


def _distance_to_vertex(x, index):
    """ The squared distance from x to the unit vector at position index. """

    point = np.array([x[f"x{i + 1}"] for i in range(4)])
    return np.linalg.norm(point - np.eye(4)[index]) ** 2.0


def _distance_grad(x, index):
    """ The gradient of _distance_to_vertex: 2(x - e_index). """

    dx = {f"x{i + 1}": 2 * x[f"x{i + 1}"] for i in range(4)}
    dx[f"x{index + 1}"] = dx[f"x{index + 1}"] - 2
    return dx, {}


def f_vertex0(x, sim):
    return _distance_to_vertex(x, 0)


def g_vertex0(x, sim):
    return _distance_grad(x, 0)


def f_vertex1(x, sim):
    return _distance_to_vertex(x, 1)


def g_vertex1(x, sim):
    return _distance_grad(x, 1)


def f_vertex2(x, sim):
    return _distance_to_vertex(x, 2)


def g_vertex2(x, sim):
    return _distance_grad(x, 2)


VERTEX_OBJECTIVES = [
    {'obj_func': f_vertex0, 'obj_grad': g_vertex0},
    {'obj_func': f_vertex1, 'obj_grad': g_vertex1},
    {'obj_func': f_vertex2, 'obj_grad': g_vertex2},
]


@pytest.mark.parametrize("categorical", [False, True],
                         ids=["continuous", "with_categorical"])
def test_solve_three_objectives_without_simulations(categorical):
    """ Check solve() on a gradient-based problem with no simulations at all.

    The three objectives pull the design toward three different vertices of
    the unit cube.  The variant with a categorical fourth variable checks that
    the gradient-based solver handles a design space that is not all
    continuous.

    """

    moop = MOOP(GlobalSurrogate_BFGS, hyperparams=seeded())
    for i in range(3 if categorical else 4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    if categorical:
        moop.addDesign({'des_type': "categorical", 'levels': 3})
    moop.addObjective(*VERTEX_OBJECTIVES)
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    moop.solve(ITERATIONS)

    soln = moop.getPF()
    assert (soln.size > 0)
    assert_front_is_self_consistent(soln,
                                    [f_vertex0, f_vertex1, f_vertex2])


# ---------------------------------------------------------------------------
# Gradient-based and gradient-free solvers must agree
# ---------------------------------------------------------------------------


def sim_norm_all(x):
    """ The norm of every design variable. """

    return [np.sqrt(sum([x[i] ** 2 for i in x]))]


def sim_norms_shifted(x):
    """ The norms of the design point shifted by 0.5 and by 1.0. """

    return [np.sqrt(sum([(x[i] - 0.5) ** 2 for i in x])),
            np.sqrt(sum([(x[i] - 1.0) ** 2 for i in x]))]


def obj_x1_squared(x, s):
    return x["x1"] ** 2


def grad_x1_squared(x, s):
    return ({"x1": 2 * x["x1"]}, {"sim1": 0, "sim2": jnp.zeros(2)})


def obj_sim_distance(x, s):
    return np.sum([jnp.dot(s[i] - 0.5, s[i] - 0.5) for i in ["sim1", "sim2"]])


def grad_sim_distance(x, s):
    return ({"x1": 0},
            {"sim1": 2 * s["sim1"] - 1, "sim2": 2 * s["sim2"] - jnp.ones(2)})


def con_x1_quarter(x, s):
    return x["x1"] - 0.25


def grad_con_x1_quarter(x, s):
    return {"x1": 1}, {"sim1": 0, "sim2": jnp.zeros(2)}


def con_sim1_quarter(x, s):
    return s["sim1"] - 0.25


def grad_con_sim1_quarter(x, s):
    return {"x1": 0}, {"sim1": 1, "sim2": jnp.zeros(2)}


def build_agreement_moop(opt, with_grads):
    """ Build the 1-variable, 2-simulation, 2-objective agreement problem.

    Args:
        opt (SurrogateOptimizer): The optimizer class to drive it with.

        with_grads (bool): Whether to supply the analytic gradients.

    Returns:
        MOOP: The solved MOOP, after solve(0).

    """

    hyperparams = seeded(opt_restarts=2) if with_grads else seeded()
    moop = MOOP(opt, hyperparams=hyperparams)
    moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(sim_dict(1, sim_norm_all, 100),
                       sim_dict(2, sim_norms_shifted, 100))
    if with_grads:
        moop.addObjective({'obj_func': obj_x1_squared,
                           'obj_grad': grad_x1_squared})
        moop.addObjective({'obj_func': obj_sim_distance,
                           'obj_grad': grad_sim_distance})
        moop.addConstraint({'constraint': con_x1_quarter,
                            'con_grad': grad_con_x1_quarter})
        moop.addConstraint({'constraint': con_sim1_quarter,
                            'con_grad': grad_con_sim1_quarter})
    else:
        moop.addObjective({'obj_func': obj_x1_squared})
        moop.addObjective({'obj_func': obj_sim_distance})
        moop.addConstraint({'constraint': con_x1_quarter})
        moop.addConstraint({'constraint': con_sim1_quarter})
    moop.addAcquisition({'acquisition': FixedWeights,
                         'hyperparams': {'weights': np.ones(2) / 2}})
    moop.solve(0)
    return moop


def test_gradient_and_gradient_free_solvers_agree():
    """ Check that supplying gradients does not change where the solver goes.

    Two MOOPs describe the same problem, one with analytic gradients driven by
    L-BFGS-B and one without, driven by pattern search.  Given the same seed
    and the same fixed acquisition weights, the candidates they propose for
    the next iteration must agree.

    """

    with_grads = build_agreement_moop(GlobalSurrogate_BFGS, True)
    without_grads = build_agreement_moop(GlobalSurrogate_PS, False)
    for x1, x2 in zip(with_grads.iterate(1), without_grads.iterate(1)):
        assert (np.abs(x1[0]["x1"] - x2[0]["x1"]) < 0.1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
