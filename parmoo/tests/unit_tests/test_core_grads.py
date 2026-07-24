""" Unit tests for the MOOP objective/constraint/penalty backward passes.

ParMOO does not differentiate user code with jax.  Users supply obj_grad and
con_grad implementations, and MOOP._link() binds them as the backward passes
of a jax.custom_vjp.  These tests assemble the full Jacobian from those
backward passes and compare it against hand-computed values, then confirm that
jax propagates the same result through the linked forward function.

All three Jacobians are checked over the same progression of problems, so the
progression is expressed once as a parametrized list of scenarios.

"""

import numpy as np
import pytest
from jax import jacrev
from jax import numpy as jnp

from parmoo import MOOP
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import GlobalSurrogate_PS
from parmoo.tests.unit_tests.helpers import (
    sim_dict,
    sim_norm,
    sim_shifted_norms,
)

DESIGN_NAMES = ["x1", "x2", "x3"]
TRAINING_POINT = {"x1": 1, "x2": 1, "x3": 1}
# The design space of the rescaled MOOP, which is twice as wide in every
# coordinate, so its latent Jacobian must be half as large.
SCALED_BOUNDS = [(-1.0, 1.0), (0.0, 2.0), (-0.5, 1.5)]


# ---------------------------------------------------------------------------
# Differentiable objectives and constraints, with hand-written gradients
# ---------------------------------------------------------------------------


def f1(x, s):
    """ The squared 2-norm of the design point. """

    return np.sum([x[i] * x[i] for i in DESIGN_NAMES])


def df1(x, s):
    """ The gradient of f1: 2x, with no simulation dependence. """

    return ({"x1": 2 * x["x1"], "x2": 2 * x["x2"], "x3": 2 * x["x3"]},
            {"sim1": 0, "sim2": jnp.zeros(2)})


def f2(x, s):
    """ The squared distance of the simulation outputs from 0.5. """

    return np.sum([jnp.dot(s[i] - 0.5, s[i] - 0.5) for i in ["sim1", "sim2"]])


def df2(x, s):
    """ The gradient of f2: 2s - 1, with no design dependence. """

    return ({"x1": 0, "x2": 0, "x3": 0},
            {"sim1": 2 * s["sim1"] - 1, "sim2": 2 * s["sim2"] - jnp.ones(2)})


def c1(x, s):
    """ A constraint on the first design variable. """

    return x["x1"] - 0.25


def dc1(x, s):
    """ The gradient of c1: e_1. """

    return {"x1": 1, "x2": 0, "x3": 0}, {"sim1": 0, "sim2": jnp.zeros(2)}


def c2(x, s):
    """ A constraint on the first simulation output. """

    return s["sim1"] - 0.25


def dc2(x, s):
    """ The gradient of c2: the first simulation direction. """

    return {"x1": 0, "x2": 0, "x3": 0}, {"sim1": 1, "sim2": jnp.zeros(2)}


OBJ1 = [(f1, df1)]
OBJ12 = [(f1, df1), (f2, df2)]
CON1 = [(c1, dc1)]
CON12 = [(c1, dc1), (c2, dc2)]


# ---------------------------------------------------------------------------
# Jacobian assembly from the fwd/bwd pair
# ---------------------------------------------------------------------------


def jacobian(moop, x, kind):
    """ Assemble a full Jacobian from a fwd/bwd pass pair.

    Args:
        moop (MOOP): A compiled MOOP with fitted surrogates.

        x (ndarray): The latent-space point to differentiate at.

        kind (str): One of "obj", "con", or "pen".

    Returns:
        jax.numpy.ndarray: The Jacobian, with one row per output.

    """

    fwd, bwd, rows = {
        "obj": (moop._obj_fwd, moop._obj_bwd, moop.o),
        "con": (moop._con_fwd, moop._con_bwd, moop.p),
        "pen": (moop._pen_fwd, moop._pen_bwd, moop.o),
    }[kind]
    sx = moop._evaluate_surrogates(x)
    dsdx = jacrev(moop._evaluate_surrogates)(x)
    _, res = fwd(x, sx)
    out = jnp.zeros((rows, moop.n_latent))
    for i, ei in enumerate(jnp.eye(rows)):
        ddx, dds = bwd(res, ei)
        out = out.at[i].set(ddx + jnp.dot(dds, dsdx))
    return out


def grad_moop(objectives, constraints=(), with_sims=True, bounds=None):
    """ Build and compile a differentiable MOOP with fitted surrogates.

    Args:
        objectives (list of tuple): (obj_func, obj_grad) pairs.

        constraints (list of tuple): (con_func, con_grad) pairs.

        with_sims (bool): Whether to attach the two test simulations and fit
            their surrogates to a single training point.

        bounds (list of tuple, optional): Per-variable (lb, ub) pairs.
            Defaults to the unit cube.

    Returns:
        MOOP: The compiled MOOP.

    """

    moop = MOOP(GlobalSurrogate_PS)
    for lb, ub in (bounds if bounds is not None else [(0.0, 1.0)] * 3):
        moop.addDesign({'lb': lb, 'ub': ub})
    if with_sims:
        moop.addSimulation(sim_dict(1, sim_norm),
                           sim_dict(2, sim_shifted_norms))
    for func, grad in objectives:
        moop.addObjective({'obj_func': func, 'obj_grad': grad})
    for func, grad in constraints:
        moop.addConstraint({'con_func': func, 'con_grad': grad})
    moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    if with_sims:
        for name in ["sim1", "sim2"]:
            moop.evaluateSimulation(TRAINING_POINT, name)
        moop._fit_surrogates()
        moop._set_surrogate_tr(np.zeros(3), np.ones(3) * np.inf)
    return moop


# ---------------------------------------------------------------------------
# Penalty Jacobian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objs, cons, with_sims, at_one", [
    # Without a constraint the penalty is just the objective gradient, 2x
    (OBJ1, (), False, [[2.0, 2.0, 2.0]]),
    (OBJ1, (), True, [[2.0, 2.0, 2.0]]),
    # c1 = x1 - 0.25 is violated at x = 1, adding e_1 to every row
    (OBJ1, CON1, False, [[3.0, 2.0, 2.0]]),
    (OBJ1, CON1, True, [[3.0, 2.0, 2.0]]),
    # f2 has no design dependence, so its row is the constraint term alone
    (OBJ12, CON12, True, [[3.0, 2.0, 2.0], [1.0, 0.0, 0.0]]),
])
def test_penalty_jacobian(objs, cons, with_sims, at_one):
    """ Check the penalty Jacobian against hand-computed values.

    The penalty is the objective plus the violation of each constraint, so a
    violated constraint adds its gradient to every objective's row.

    """

    moop = grad_moop(objs, cons, with_sims)
    expected = np.asarray(at_one)
    assert (jacobian(moop, np.zeros(3), "pen").shape == expected.shape)
    # At the origin no constraint is violated and f1's gradient vanishes
    assert (np.all(np.abs(jacobian(moop, np.zeros(3), "pen")) < 1.0e-8))
    assert (np.all(np.abs(jacobian(moop, np.ones(3), "pen") - expected)
                   < 1.0e-8))


# ---------------------------------------------------------------------------
# Objective Jacobian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("objs, cons, with_sims, at_one", [
    (OBJ1, (), False, [[2.0, 2.0, 2.0]]),
    # Adding a constraint must leave the objective Jacobian untouched
    (OBJ1, CON1, False, [[2.0, 2.0, 2.0]]),
    (OBJ1, (), True, [[2.0, 2.0, 2.0]]),
    (OBJ1, CON1, True, [[2.0, 2.0, 2.0]]),
    # f2 depends only on the simulations, whose surrogate is flat here
    (OBJ12, CON12, True, [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]),
])
def test_objective_jacobian(objs, cons, with_sims, at_one):
    """ Check the objective Jacobian against hand-computed values. """

    moop = grad_moop(objs, cons, with_sims)
    expected = np.asarray(at_one)
    assert (jacobian(moop, np.zeros(3), "obj").shape == expected.shape)
    assert (np.all(np.abs(jacobian(moop, np.zeros(3), "obj")) < 1.0e-8))
    assert (np.all(np.abs(jacobian(moop, np.ones(3), "obj") - expected)
                   < 1.0e-8))


# ---------------------------------------------------------------------------
# Constraint Jacobian
# ---------------------------------------------------------------------------


def test_constraint_jacobian_is_empty_when_unconstrained():
    """ Check that an unconstrained MOOP has an empty constraint Jacobian. """

    moop = grad_moop(OBJ1, (), with_sims=False)
    assert (jacobian(moop, np.zeros(3), "con").size == 0)


@pytest.mark.parametrize("objs, cons, with_sims, expected", [
    (OBJ1, CON1, False, [[1.0, 0.0, 0.0]]),
    (OBJ1, CON1, True, [[1.0, 0.0, 0.0]]),
    # c2 depends only on sim1, whose surrogate is flat at the training point
    (OBJ12, CON12, True, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
])
def test_constraint_jacobian(objs, cons, with_sims, expected):
    """ Check the constraint Jacobian against hand-computed values.

    Both constraints are affine, so the Jacobian is the same everywhere.

    """

    moop = grad_moop(objs, cons, with_sims)
    expected = np.asarray(expected)
    assert (jacobian(moop, np.zeros(3), "con").shape == expected.shape)
    for xi in [np.zeros(3), np.ones(3)]:
        assert (np.all(np.abs(jacobian(moop, xi, "con") - expected) < 1.0e-8))


# ---------------------------------------------------------------------------
# Design-space rescaling and jax propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["obj", "con", "pen"])
def test_jacobian_scales_with_design_space(kind):
    """ Check that the latent Jacobian rescales with the design bounds.

    The latent space is always the unit cube, so doubling the width of every
    design variable must halve every latent partial derivative.

    """

    unit = grad_moop(OBJ12, CON12)
    scaled = grad_moop(OBJ12, CON12, bounds=SCALED_BOUNDS)
    x = unit._embed(TRAINING_POINT)
    xx = scaled._embed(TRAINING_POINT)
    assert (np.linalg.norm(jacobian(unit, x, kind) -
                           jacobian(scaled, xx, kind) * 2) < 1.0e-8)


@pytest.mark.parametrize("kind, link_index", [("obj", 0), ("con", 1),
                                              ("pen", 2)])
def test_jax_propagates_the_backward_pass(kind, link_index):
    """ Check that jax.jacrev through the linked function matches the bwd pass.

    _link() returns the forward functions that MOOP registers with
    jax.custom_vjp.  Differentiating those with jacrev must reproduce the
    Jacobian assembled directly from the backward passes.

    """

    moop = grad_moop(OBJ12, CON12)
    linked = moop._link()[link_index]

    def evaluate(x):
        return linked(x, moop._evaluate_surrogates(x))

    linked_jac = jacrev(evaluate)
    for xi in np.random.sample((5, 3)):
        assert (np.all(np.abs(jacobian(moop, xi, kind) - linked_jac(xi))
                       < 1.0e-8))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
