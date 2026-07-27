""" Unit tests for the parmoo.optimizers plugins.

Note on structure, because it differs between the two kinds of validation
these classes perform:

 * setObjective(), setConstraints(), setPenalty(), setTrFunc(), and
   addAcquisition() are validated by the SurrogateOptimizer ABC and are
   therefore *inherited* by every optimizer.  One test against one concrete
   subclass covers all of them.
 * solve() is abstract, so its input validation is implemented -- and
   copy-pasted -- separately in pattern_search.py, lbfgsb.py, and
   random_search.py.  Each copy is a distinct set of coverage lines, so that
   contract is parametrized over every optimizer class.

Every optimizer is then checked numerically on the same three-variable,
biobjective problem

    F(x) = (-x_1 + x_2 + x_3, x_1 - x_2 + x_3),   x in [0, 1]^3

subject to 0.1 <= x_3 <= 0.6, using the weights [1, 0], [0, 1], and
[0.5, 0.5].  So (1, 0, 0.1) minimizes F^T [1, 0], (0, 1, 0.1) minimizes
F^T [0, 1], and the minimizer of F^T [0.5, 0.5] satisfies x_3 = 0.1.

"""

import numpy as np
import pytest
from jax import numpy as jnp

from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import (
    GlobalSurrogate_BFGS,
    GlobalSurrogate_PS,
    GlobalSurrogate_RS,
    LocalSurrogate_BFGS,
    LocalSurrogate_PS,
)

# Problem dimensions shared by every test in this file
N = 3
N_OBJ = 2
LB = np.zeros(N)
UB = np.ones(N)

ALL_OPTIMIZERS = [
    LocalSurrogate_PS,
    GlobalSurrogate_PS,
    LocalSurrogate_BFGS,
    GlobalSurrogate_BFGS,
    GlobalSurrogate_RS,
]


def make_problem(sim_dim=2, penalty_coef=5.0, jittable=True):
    """ Build the biobjective test problem described in the module docstring.

    Args:
        sim_dim (int): The length of the simulation output vector that the
            surrogate stand-ins return.

        penalty_coef (float): The coefficient on the constraint violation in
            the penalty function.  Zero gives an unpenalized problem.

        jittable (bool): When False, build the objective out of numpy and the
            Python builtin max() so that jax cannot trace it.  The optimizers
            wrap their jit attempt in a try/except and silently fall back to
            the uncompiled function, and that fallback needs exercising too.

    Returns:
        tuple: The objective, penalty, simulation, simulation-uncertainty,
        and constraint functions, in that order.

    """

    if jittable:
        def f(z, sz):
            return jnp.asarray([-z[0] + z[1] + z[2], z[0] - z[1] + z[2]])

        def L(z, sz):
            return f(z, sz) + penalty_coef * (jnp.maximum(0.1 - z[2], 0) +
                                              jnp.maximum(z[2] - 0.6, 0))
    else:
        def f(z, sz):
            return np.asarray([-z[0] + z[1] + z[2], z[0] - z[1] + z[2]])

        def L(z, sz):
            return f(z, sz) + penalty_coef * (max(0.1 - z[2], 0) +
                                              max(z[2] - 0.6, 0))

    def S(z):
        return jnp.ones(sim_dim)

    def SD(z):
        return jnp.zeros(sim_dim)

    def C(z, sz):
        return jnp.asarray([0.1 - z[2], z[2] - 0.6])

    return f, L, S, SD, C


def make_acquisitions(penalty_func):
    """ Build 3 acquisitions targeting the 2 pure and 1 tradeoff solutions. """

    weights = [np.eye(N_OBJ)[0], np.eye(N_OBJ)[1], np.ones(N_OBJ) * 0.5]
    acqus = []
    for wi in weights:
        acqu = UniformWeights(N_OBJ, LB, UB, {})
        acqu.setTarget({}, penalty_func)
        acqu.weights[:] = wi
        acqus.append(acqu)
    return acqus


def make_optimizer(cls, hyperparams=None, sim_dim=2, penalty_coef=5.0,
                   jittable=True):
    """ Build a fully configured optimizer ready to call solve(). """

    f, L, S, SD, C = make_problem(sim_dim, penalty_coef, jittable)
    opt = cls(N_OBJ, LB, UB, hyperparams if hyperparams is not None else {})
    opt.setObjective(f)
    opt.setConstraints(C)
    opt.setSimulation(S, SD)
    opt.setPenalty(L)
    opt.addAcquisition(*make_acquisitions(L))
    opt.setTrFunc(lambda x, r: 100.0)
    return opt, f, L


def expected_solutions():
    """ Return the two known pure minimizers of the test problem. """

    x1_soln = np.eye(N)[0]
    x1_soln[N - 1] = 0.1
    x2_soln = np.eye(N)[1]
    x2_soln[N - 1] = 0.1
    return x1_soln, x2_soln


def test_optimizer_callback_validation():
    """ Check the callback setters inherited from the SurrogateOptimizer ABC.

    setObjective, setConstraints, setPenalty, setTrFunc, and addAcquisition are
    all implemented once in the ABC, so exercising them against a single
    concrete subclass covers every optimizer.

    """

    opt = LocalSurrogate_PS(N_OBJ, LB, UB, {})
    for setter in [opt.setObjective, opt.setConstraints, opt.setPenalty]:
        # Must be callable
        with pytest.raises(TypeError):
            setter(5)
        # Must accept exactly two inputs
        with pytest.raises(ValueError):
            setter(lambda z1, z2, z3: np.zeros(1))
    # setTrFunc must be callable and accept exactly two inputs
    with pytest.raises(TypeError):
        opt.setTrFunc(5)
    with pytest.raises(ValueError):
        opt.setTrFunc(lambda z1: 0.0)
    # addAcquisition only accepts AcquisitionFunction instances
    with pytest.raises(TypeError):
        opt.addAcquisition(5)


@pytest.mark.parametrize("cls", ALL_OPTIMIZERS)
def test_optimizer_bad_opt_budget(cls):
    """ Check the 'opt_budget' hyperparameter contract on every optimizer. """

    with pytest.raises(TypeError):
        cls(N_OBJ, LB, UB, {'opt_budget': 2.0})
    with pytest.raises(ValueError):
        cls(N_OBJ, LB, UB, {'opt_budget': 0})
    # A legal budget is accepted
    cls(N_OBJ, LB, UB, {'opt_budget': 100})


@pytest.mark.parametrize("cls", ALL_OPTIMIZERS)
def test_optimizer_solve_bad_shape(cls):
    """ Check solve()'s input shape contract on every optimizer.

    solve() is abstract, so this validation is re-implemented in each of
    pattern_search.py, lbfgsb.py, and random_search.py.

    """

    opt, _, _ = make_optimizer(cls)
    # The columns of x must match n
    with pytest.raises(ValueError):
        opt.solve(np.zeros((3, N - 1)))
    # The rows of x must match the number of acquisitions
    with pytest.raises(ValueError):
        opt.solve(np.zeros((4, N)))


@pytest.mark.parametrize("cls", [GlobalSurrogate_BFGS, LocalSurrogate_BFGS])
def test_bfgs_solve_infeasible_start(cls):
    """ Check that the BFGS optimizers reject infeasible starting points. """

    opt, _, _ = make_optimizer(cls, sim_dim=1, penalty_coef=2.0)
    with pytest.raises(ValueError):
        opt.solve(-np.ones((3, N)))


def test_GlobalSurrogate_PS_bad_gps_budget():
    """ Check the GlobalSurrogate_PS-specific 'gps_budget' contract.

    gps_budget must be an int, must be positive, and may not exceed the
    overall opt_budget.

    """

    with pytest.raises(TypeError):
        GlobalSurrogate_PS(N_OBJ, LB, UB,
                           {'opt_budget': 500, 'gps_budget': 2.0})
    with pytest.raises(ValueError):
        GlobalSurrogate_PS(N_OBJ, LB, UB, {'opt_budget': 500, 'gps_budget': 0})
    with pytest.raises(ValueError):
        GlobalSurrogate_PS(N_OBJ, LB, UB,
                           {'opt_budget': 500, 'gps_budget': 1000})
    # A legal pair is accepted
    GlobalSurrogate_PS(N_OBJ, LB, UB, {'opt_budget': 200, 'gps_budget': 100})


@pytest.mark.parametrize("cls, hyperparams, sim_dim, penalty_coef, eps", [
    (GlobalSurrogate_PS, {}, 2, 5.0, 0.1),
    (GlobalSurrogate_BFGS, {}, 1, 2.0, 0.01),
    (GlobalSurrogate_RS, {'opt_budget': 10010}, 2, 0.0, 0.25),
])
def test_global_optimizer_finds_solutions(cls, hyperparams, sim_dim,
                                          penalty_coef, eps):
    """ Check that each global optimizer solves the problem in one call.

    The global optimizers search the whole box, so a single solve() from the
    centroid must land within eps of each known minimizer.

    """

    opt, _, _ = make_optimizer(cls, hyperparams, sim_dim, penalty_coef)
    x1_soln, x2_soln = expected_solutions()
    x = np.ones((3, N)) * 0.5
    (x1, x2, x3) = opt.solve(x)
    assert (np.linalg.norm(x1 - x1_soln) < eps)
    assert (np.linalg.norm(x2 - x2_soln) < eps)
    assert (np.abs(x3[N - 1] - 0.1) < eps)


@pytest.mark.parametrize("cls, sim_dim, penalty_coef, iters, jittable", [
    # The pattern search runs against a deliberately non-jittable objective,
    # so that its silent fall-back to the uncompiled function is exercised.
    (LocalSurrogate_PS, 2, 5.0, 10, False),
    (LocalSurrogate_BFGS, 1, 2.0, 6, True),
])
def test_local_optimizer_finds_solutions(cls, sim_dim, penalty_coef, iters,
                                         jittable):
    """ Check that each trust-region optimizer converges over several restarts.

    The local optimizers only search within a trust region, so solve() is
    called repeatedly, feeding each iterate back through returnResults() to
    advance the trust-region center.

    """

    opt, f, L = make_optimizer(cls, {}, sim_dim, penalty_coef, jittable)
    x1_soln, x2_soln = expected_solutions()
    _, _, _, _, C = make_problem(sim_dim, penalty_coef, jittable)
    x = np.ones((3, N)) * 0.5
    for i in range(iters):
        # Re-set the callbacks so the trust region is re-linked each restart
        opt.setObjective(f)
        opt.setConstraints(C)
        opt.setPenalty(L)
        x[0], x[1], x[2] = opt.solve(x)
        for j in range(3):
            opt.returnResults(x[j], np.ones(N_OBJ) * -10,
                              np.zeros(1), np.zeros(1))
    eps = 0.01
    assert (np.linalg.norm(x[0] - x1_soln) < eps)
    assert (np.linalg.norm(x[1] - x2_soln) < eps)
    assert (np.abs(x[2, N - 1] - 0.1) < eps)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
