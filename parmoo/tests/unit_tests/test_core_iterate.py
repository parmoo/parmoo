""" Unit tests for the MOOP iterate/filterBatch/updateAll solve loop steps.

Each test drives the loop by hand -- iterate(k), filterBatch, evaluate each
candidate, updateAll(k) -- rather than calling solve(), so that the individual
steps can be checked.  The four problems exercised here differ in ways that
matter to the loop: whether user gradients are available, whether the design
space contains categorical variables, and whether those categories are
integers or strings.

Note: these sim dicts carry 'search_budget' as a top-level key rather than
inside 'hyperparams', which is where LatinHypercube reads it from.  That is
preserved from the original tests, so the searches actually run at the default
budget of 100 points.

"""

import numpy as np
import pytest

from parmoo import MOOP
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import LocalSurrogate_BFGS, LocalSurrogate_PS
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF
from parmoo.tests.unit_tests.helpers import seeded


def sim_dict(m, sim_func, search_budget):
    """ Build a simulation dict, preserving the original key placement. """

    return {'m': m,
            'hyperparams': {},
            'search': LatinHypercube,
            'sim_func': sim_func,
            'surrogate': GaussRBF,
            'search_budget': search_budget}


def run_loop(moop, iterations=2, seed_batch=None):
    """ Drive the solve loop by hand for the given number of iterations.

    Args:
        moop (MOOP): The MOOP to advance.

        iterations (int): How many iterations to run.

        seed_batch (list, optional): When given, iteration 0 discards the
            search batch and uses this hand-picked batch instead, so that the
            surrogates are fit to a known set of points.

    """

    for k in range(iterations):
        if k == 0 and seed_batch is not None:
            moop.iterate(0)
            batch = seed_batch
        else:
            batch = moop.filterBatch(moop.iterate(k))
        for (x, name) in batch:
            moop.evaluateSimulation(x, name)
        moop.updateAll(k, batch)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_iterate_requires_a_complete_problem():
    """ Check that iterate() refuses to run an incompletely defined MOOP.

    A MOOP needs design variables and at least one objective before it can be
    iterated, and neither omission is caught until iterate() is called.

    """

    moop = MOOP(LocalSurrogate_PS, hyperparams=seeded(opt_budget=100))
    # No design variables yet
    with pytest.raises(AttributeError):
        moop.iterate(1)
    for i in range(3):
        moop.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(sim_dict(1, sim_norm_123, 20))
    # Still no objectives
    with pytest.raises(AttributeError):
        moop.iterate(1)


@pytest.mark.parametrize("k, error", [(-1, ValueError), (2.0, TypeError)])
def test_iterate_rejects_bad_iteration_index(k, error):
    """ Check that iterate() validates the iteration counter. """

    moop = MOOP(LocalSurrogate_PS, hyperparams=seeded(opt_budget=100))
    for i in range(3):
        moop.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(sim_dict(1, sim_norm_123, 20))
    moop.addObjective({'obj_func': obj_sim1})
    moop.addAcquisition({'acquisition': UniformWeights})
    with pytest.raises(error):
        moop.iterate(k)


# ---------------------------------------------------------------------------
# Problem 1: two simulations, no user gradients
# ---------------------------------------------------------------------------


def sim_norm_123(x):
    """ The norm of (x1, x2, x3). """

    return [np.linalg.norm([x[f"x{k}"] for k in [1, 2, 3]])]


def sim_norm_123_shifted(x):
    """ The norm of (x1, x2, x3) shifted by one. """

    return [np.linalg.norm([x[f"x{k}"] - 1 for k in [1, 2, 3]])]


def obj_sim1(x, sim):
    """ The first simulation output. """

    return sim["sim1"]


def obj_sim2(x, sim):
    """ The second simulation output. """

    return sim["sim2"]


def test_iterate_two_simulation_problem():
    """ Check the loop on a two-simulation, two-objective problem.

    Each objective is exactly one simulation output, so every point on the
    returned Pareto front must reproduce its simulation value exactly.

    """

    moop = MOOP(LocalSurrogate_PS, hyperparams=seeded(opt_budget=100))
    for i in range(3):
        moop.addDesign({'name': f"x{i+1}", 'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(sim_dict(1, sim_norm_123, 20),
                       sim_dict(1, sim_norm_123_shifted, 20))
    moop.addObjective({'obj_func': obj_sim1}, {'obj_func': obj_sim2})
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    run_loop(moop)

    soln = moop.getPF()
    assert (soln.size > 0)
    for si in soln:
        assert (np.abs(sim_norm_123(si) - si['f1']) < 1.0e-8)
        assert (np.linalg.norm(sim_norm_123_shifted(si) - si['f2']) < 1.0e-8)


# ---------------------------------------------------------------------------
# Problem 2: user-supplied gradients, four continuous variables
# ---------------------------------------------------------------------------


def sim_identity_4(x):
    """ Return the design variables unchanged. """

    return [x[i] for i in x]


def _target_obj(sim, index):
    """ The squared distance from sim1 to 0.1 * e_index. """

    return np.linalg.norm(sim["sim1"] - 0.1 * np.eye(4)[index, :]) ** 2.0


def _target_grad(x, sim, index):
    """ The gradient of _target_obj, written in ParMOO's dict form. """

    x_out = {key: 0.0 for key in x}
    s_out = sim.copy()
    s_out["sim1"] = s_out["sim1"] * 2.0
    s_out["sim1"] = s_out["sim1"].at[index].set(s_out["sim1"][index] - 0.2)
    return x_out, s_out


def f_target0(x, sim):
    return _target_obj(sim, 0)


def g_target0(x, sim):
    return _target_grad(x, sim, 0)


def f_target1(x, sim):
    return _target_obj(sim, 1)


def g_target1(x, sim):
    return _target_grad(x, sim, 1)


def f_target2(x, sim):
    return _target_obj(sim, 2)


def g_target2(x, sim):
    return _target_grad(x, sim, 2)


def test_iterate_with_user_gradients():
    """ Check the loop on a gradient-based problem with a known optimum.

    The three objectives pull the simulation output toward 0.1 * e_1, e_2, and
    e_3, so every nondominated design must lie near the origin.

    """

    moop = MOOP(LocalSurrogate_BFGS, hyperparams=seeded(opt_budget=100))
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0, 'des_tol': 0.1})
    moop.addSimulation(sim_dict(4, sim_identity_4, 500))
    moop.addObjective({'obj_func': f_target0, 'obj_grad': g_target0},
                      {'obj_func': f_target1, 'obj_grad': g_target1},
                      {'obj_func': f_target2, 'obj_grad': g_target2})
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})

    # Seed the surrogates with the origin-adjacent points and the diagonal
    seed = []
    for i in range(1, 5):
        xi = {f"x{j}": 0.0 for j in range(1, 5)}
        xi[f"x{i}"] = 0.1
        seed.append((xi, "sim1"))
    seed.append(({f"x{j}": 0.1 for j in range(1, 5)}, "sim1"))
    run_loop(moop, seed_batch=seed)

    soln = moop.getPF()
    assert (soln.size > 0)
    for xsi in soln:
        xi = {f"x{j}": xsi[f"x{j}"] for j in range(1, 5)}
        si = {"sim1": np.asarray(sim_identity_4(xi))}
        expected = np.array([f_target0(xi, si), f_target1(xi, si),
                             f_target2(xi, si)]).flatten()
        got = [xsi["f1"], xsi["f2"], xsi["f3"]]
        assert (np.linalg.norm(expected - got) < 1.0e-8)
        # Every nondominated point stays near the origin
        for j in xi:
            assert (xi[j] <= 0.2)


# ---------------------------------------------------------------------------
# Problem 3: an integer-valued categorical variable
# ---------------------------------------------------------------------------


def sim_offset_by_x5(x):
    """ Shift the first four variables by the distance of x5 from one. """

    return abs(x["x5"] - 1) + np.array([x[f"x{i+1}"] for i in range(4)])


def _unit_obj(sim, index):
    """ The squared distance from sim1 to e_index. """

    return np.linalg.norm(sim["sim1"] - np.eye(4)[index, :]) ** 2


def _unit_grad(sim, index):
    """ The gradient of _unit_obj, written in ParMOO's dict form. """

    ds = 2.0 * np.asarray(sim["sim1"])
    ds[index] = ds[index] - 2.0
    return {f"x{i+1}": 0 for i in range(5)}, {"sim1": ds}


def f_unit0(x, sim):
    return _unit_obj(sim, 0)


def g_unit0(x, sim):
    return _unit_grad(sim, 0)


def f_unit1(x, sim):
    return _unit_obj(sim, 1)


def g_unit1(x, sim):
    return _unit_grad(sim, 1)


def f_unit2(x, sim):
    return _unit_obj(sim, 2)


def g_unit2(x, sim):
    return _unit_grad(sim, 2)


def test_iterate_with_integer_categorical():
    """ Check the loop when the design space mixes continuous and categorical.

    The simulation is minimized when x5 == 1, which is the middle of three
    integer categories, so the loop must discover that level.

    """

    moop = MOOP(LocalSurrogate_BFGS, hyperparams=seeded())
    for i in range(4):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'des_type': "categorical", 'levels': 3})
    moop.addSimulation(sim_dict(4, sim_offset_by_x5, 500))
    moop.addObjective({'obj_func': f_unit0, 'obj_grad': g_unit0},
                      {'obj_func': f_unit1, 'obj_grad': g_unit1},
                      {'obj_func': f_unit2, 'obj_grad': g_unit2})
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})

    # Seed with each unit vector, the all-ones point, and one at level 2
    seed = []
    for i in range(1, 6):
        xi = {f"x{j}": 0 for j in range(1, 6)}
        xi[f"x{i}"] = 1
        seed.append((xi, "sim1"))
    seed.append(({f"x{j}": 1 for j in range(1, 6)}, "sim1"))
    seed.append(({"x1": 1, "x2": 1, "x3": 1, "x4": 1, "x5": 2}, "sim1"))
    run_loop(moop, seed_batch=seed)

    soln = moop.getPF()
    assert (np.size(soln) > 0)
    for xsi in soln:
        xi = {f"x{j}": xsi[f"x{j}"] for j in range(1, 6)}
        si = {"sim1": sim_offset_by_x5(xi)}
        expected = np.array([f_unit0(xi, si), f_unit1(xi, si),
                             f_unit2(xi, si)]).flatten()
        got = [xsi["f1"], xsi["f2"], xsi["f3"]]
        assert (np.linalg.norm(expected - got) < 1.0e-8)
        # The loop found the minimizing category
        assert (abs(xi["x4"]) <= 0.1 and abs(xi["x5"] - 1) <= 0.1)


# ---------------------------------------------------------------------------
# Problem 4: a string-valued categorical variable
# ---------------------------------------------------------------------------


def sim_quadratic_with_level(x):
    """ A quadratic in x0 and x1, offset by the numeric value of the level. """

    return [(x["x0"] - 1.0) ** 2 + x["x1"] ** 2 + float(x["x2"])]


def f_from_sim(x, sim):
    """ The simulation output. """

    return sim["sim1"]


def g_from_sim(x, sim):
    """ The gradient of f_from_sim. """

    return {"x0": 0, "x1": 0}, {"sim1": 1}


def f_mirrored(x, sim):
    """ The mirrored quadratic, computed algebraically from the design. """

    return x["x0"] ** 2 + (x["x1"] - 1.0) ** 2 + float(x["x2"])


def g_mirrored(x, sim):
    """ The gradient of f_mirrored. """

    return ({"x0": 2.0 * x["x0"], "x1": 2.0 * x["x1"] - 2.0}, {"sim1": 0})


def test_iterate_with_string_categorical():
    """ Check the loop when a categorical variable's levels are strings.

    Both objectives increase with float(x2), so the loop must settle on the
    "0" level.  This also covers a level list of strings rather than a count.

    """

    moop = MOOP(LocalSurrogate_BFGS, hyperparams=seeded())
    moop.addDesign({'name': "x0", 'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'name': "x2", 'des_type': "categorical",
                    'levels': ["0", "1"]})
    moop.addSimulation(sim_dict(1, sim_quadratic_with_level, 100))
    moop.addObjective({'obj_func': f_from_sim, 'obj_grad': g_from_sim},
                      {'obj_func': f_mirrored, 'obj_grad': g_mirrored})
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    run_loop(moop)

    soln = moop.getPF()
    assert (soln.size > 0)
    for i, xi in enumerate(soln):
        sim = {"sim1": sim_quadratic_with_level(xi)[0]}
        assert (f_from_sim(soln[i], sim) - soln['f1'][i] < 1.0e-8)
        assert (f_mirrored(soln[i], sim) - soln['f2'][i] < 1.0e-8)
        # The cheaper level wins
        assert (xi["x2"] == "0")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
