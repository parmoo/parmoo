""" Unit tests for the parmoo.acquisitions plugins.

Note on structure: setTarget()'s input validation is copy-pasted verbatim in
all three acquisition modules -- six times in total -- so each copy is a
distinct set of coverage lines.  The shared contract is therefore
parametrized over every acquisition class rather than deleted.  Only the
assertions that genuinely differ between classes get their own test.

"""

import numpy as np
import pytest
from jax import jacrev

from parmoo.acquisitions import (
    EI_RandomConstraint,
    FixedAugChebyshev,
    FixedWeights,
    RandomConstraint,
    UniformAugChebyshev,
    UniformWeights,
)
from parmoo.utilities.moop_utils import updatePF

# Problem dimensions used by the shared tests
N_OBJ = 3
N = 4

# Every acquisition class, for the copy-pasted setTarget() contract
ALL_ACQUISITIONS = [
    UniformWeights,
    FixedWeights,
    UniformAugChebyshev,
    FixedAugChebyshev,
    RandomConstraint,
    EI_RandomConstraint,
]

# The acquisitions that expose a convex weight vector
WEIGHTED_ACQUISITIONS = [
    UniformWeights,
    FixedWeights,
    UniformAugChebyshev,
    FixedAugChebyshev,
]

# The acquisitions that accept a user-supplied 'weights' hyperparameter
FIXED_ACQUISITIONS = [FixedWeights, FixedAugChebyshev]


def penalty(x, sx=0):
    """ A well-formed penalty function accepted by every acquisition. """

    return np.zeros(N_OBJ)


def make_data(n, o, npts=10):
    """ Build a feasible evaluation database with npts points in n dims. """

    x_vals = np.random.random_sample((npts, n))

    def obj_f(x):
        return np.asarray([np.dot(x[:o] - np.eye(o)[i],
                                  x[:o] - np.eye(o)[i]) for i in range(o)])

    return {'x_vals': x_vals,
            'f_vals': np.asarray([obj_f(x) for x in x_vals]),
            'c_vals': np.zeros((npts, 1))}


@pytest.mark.parametrize("cls", ALL_ACQUISITIONS)
def test_acquisition_bounds(cls):
    """ Check that every acquisition stores the bounds it was given. """

    acqu = cls(N_OBJ, np.zeros(N), np.ones(N), {})
    assert (np.all(acqu.lb[:] == 0.0) and np.all(acqu.ub[:] == 1.0))


@pytest.mark.parametrize("cls", ALL_ACQUISITIONS)
def test_acquisition_setTarget_bad_data(cls):
    """ Check the setTarget() data argument contract, for every acquisition.

    This block of checks is duplicated in all three acquisition modules, so
    it is exercised once per class.

    """

    acqu = cls(N_OBJ, np.zeros(N), np.ones(N), {})
    # data must be a dict.  Matched on the message, because a non-dict
    # would also trip a bare TypeError further into setTarget().
    with pytest.raises(TypeError, match="data must be a dict"):
        acqu.setTarget(5, penalty)
    # x_vals and f_vals must appear together
    with pytest.raises(AttributeError):
        acqu.setTarget({'x_vals': []}, penalty)
    # x_vals and f_vals must have equal length
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones(1), 'f_vals': np.ones(2)}, penalty)
    # the rows of x_vals must have length n
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, 1)),
                        'f_vals': np.ones((1, N_OBJ))}, penalty)
    # the rows of f_vals must have length o
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, N)),
                        'f_vals': np.ones((1, 1))}, penalty)


@pytest.mark.parametrize("cls", ALL_ACQUISITIONS)
def test_acquisition_setTarget_bad_penalty(cls):
    """ Check the setTarget() penalty_func contract, for every acquisition.

    penalty_func must be callable and accept either one (x) or two (x, sx)
    inputs.

    """

    acqu = cls(N_OBJ, np.zeros(N), np.ones(N), {})
    with pytest.raises(TypeError):
        acqu.setTarget({}, 4)
    with pytest.raises(ValueError):
        acqu.setTarget({}, lambda x, y, z: np.zeros(N_OBJ))


@pytest.mark.parametrize("cls", ALL_ACQUISITIONS)
def test_acquisition_setTarget_returns_feasible_start(cls):
    """ Check that setTarget() returns an in-bounds start for every data shape.

    Each acquisition reaches its "no data" branch two different ways -- an
    empty dict, and an x_vals/f_vals pair that is present but None -- and its
    "real data" branch two more, with one point and with a full database.

    """

    acqu = cls(N_OBJ, np.zeros(N), np.ones(N), {})
    one_point = {'x_vals': np.zeros((1, N)),
                 'f_vals': np.zeros((1, N_OBJ)),
                 'c_vals': np.zeros((1, 1))}
    for data in [{},
                 {'x_vals': None, 'f_vals': None},
                 one_point,
                 make_data(N, N_OBJ)]:
        x0 = acqu.setTarget(data, penalty)
        assert (np.all(x0[:] <= acqu.ub) and np.all(x0[:] >= acqu.lb))


@pytest.mark.parametrize("cls", FIXED_ACQUISITIONS)
def test_acquisition_bad_weights(cls):
    """ Check the 'weights' hyperparameter contract on the fixed acquisitions.

    """

    with pytest.raises(TypeError):
        cls(N_OBJ, np.zeros(N), np.ones(N), {'weights': 5.0})
    with pytest.raises(ValueError):
        cls(N_OBJ, np.zeros(N), np.ones(N), {'weights': np.ones(2)})
    # A correctly sized weight vector is accepted
    cls(N_OBJ, np.zeros(N), np.ones(N), {'weights': np.ones(N_OBJ)})


@pytest.mark.parametrize("cls", WEIGHTED_ACQUISITIONS)
def test_acquisition_weights_are_convex(cls):
    """ Check that the sampled weights are nonnegative and sum to one.

    Three independent instances are drawn because the uniform variants sample
    their weights randomly, and one of the three reaches setTarget()'s
    None-valued data branch.

    """

    for data in [{}, {}, {'x_vals': None, 'f_vals': None}]:
        acqu = cls(N_OBJ, np.zeros(N), np.ones(N), {})
        acqu.setTarget(data, penalty)
        assert (all(acqu.weights[:] >= 0.0))
        assert (abs(sum(acqu.weights[:]) - 1.0) < 1.0e-8)


@pytest.mark.parametrize("cls, tol", [(UniformWeights, 1.0e-8),
                                      (FixedWeights, 1.0e-8),
                                      (UniformAugChebyshev, 1.1e-1),
                                      (FixedAugChebyshev, 1.1e-1)])
def test_acquisition_scalarize_sums_to_one(cls, tol):
    """ Check that scalarizing the unit objective vectors recovers the weights.

    Summing the scalarization over the standard basis of objective space must
    recover the sum of the weights, which is one.

    """

    acqu = cls(N_OBJ, np.zeros(N), np.ones(N), {})
    acqu.setTarget({}, penalty)
    total = sum([acqu.scalarize(np.eye(N_OBJ)[i], np.ones(2),
                                np.ones(2), np.ones(2)) for i in range(N_OBJ)])
    assert (np.abs(total - 1.0) < tol)


def test_UniformWeights_scalarize_gradient():
    """ Check that the UniformWeights scalarization gradient is its weights.

    """

    for data in [{}, {}, {'x_vals': None, 'f_vals': None}]:
        acqu = UniformWeights(N_OBJ, np.zeros(N), np.ones(N), {})
        acqu.setTarget(data, penalty)
        df = jacrev(acqu.scalarize)(np.eye(N_OBJ)[0], np.ones(N),
                                    np.ones(1), np.ones(1))
        assert (np.linalg.norm(df - acqu.weights) < 1.0e-4)


def test_FixedWeights_scalarize_gradient():
    """ Check that the FixedWeights scalarization gradient sums to one. """

    acqu = FixedWeights(N_OBJ, np.zeros(N), np.ones(N), {})
    acqu.setTarget({}, penalty)
    da = jacrev(acqu.scalarize)(np.eye(N_OBJ)[0], np.zeros(N),
                                np.zeros(1), np.zeros(1))
    assert (np.abs(np.sum(da) - 1.0) < 1.0e-4)


def test_UniformAugChebyshev_manifold():
    """ Check the Chebyshev manifold selector against its scalarization.

    getManifold() must flag the objective attaining the weighted max, and the
    scalarized value must (up to the augmentation term) equal that weighted
    max.

    """

    for data in [{}, {}, {'x_vals': None, 'f_vals': None}]:
        acqu = UniformAugChebyshev(N_OBJ, np.zeros(N), np.ones(N), {})
        acqu.setTarget(data, penalty)
        f_vals = np.random.sample(N_OBJ)
        maxind = np.argmax(acqu.weights * f_vals)
        assert (acqu.getManifold(f_vals)[maxind] == 1)
        assert (abs(acqu.scalarize(f_vals, np.ones(2), np.ones(2), np.ones(2))
                    - acqu.weights[maxind] * f_vals[maxind]) < 3.0e-2)


def test_UniformAugChebyshev_scalarize_gradient():
    """ Check that the UniformAugChebyshev gradient peaks at the max weight.

    """

    for data in [{}, {}, {'x_vals': None, 'f_vals': None}]:
        acqu = UniformAugChebyshev(N_OBJ, np.zeros(N), np.ones(N), {})
        acqu.setTarget(data, penalty)
        df = jacrev(acqu.scalarize)(np.ones(N_OBJ), np.ones(N),
                                    np.ones(1), np.ones(1))
        assert (np.abs(np.max(df) - np.max(acqu.weights) - 1.0e-2) < 1.0e-2)


def test_FixedAugChebyshev_scalarize_gradient():
    """ Check the FixedAugChebyshev scalarization gradient. """

    acqu = FixedAugChebyshev(N_OBJ, np.zeros(N), np.ones(N), {})
    acqu.setTarget({}, penalty)
    da = jacrev(acqu.scalarize)(np.eye(N_OBJ)[0], np.zeros(N),
                                np.zeros(1), np.zeros(1))
    assert (np.abs(np.sum(da) - acqu.weights[0]) < 1.1e-1)


@pytest.mark.parametrize("cls", [RandomConstraint, EI_RandomConstraint])
def test_epsilon_constraint_targets_a_database_point(cls):
    """ Check that the epsilon-constraint acquisitions target a real point.

    Given a nonempty database, setTarget() must return one of the points in
    that database, and given no database it must return an interior point.

    """

    acqu = cls(N_OBJ, np.zeros(N_OBJ), np.ones(N_OBJ), {})
    assert (np.all(acqu.setTarget({}, penalty) < 1.0))
    assert (np.all(acqu.setTarget({}, penalty) > 0.0))
    # An infeasible single point still yields an in-bounds start
    assert (np.all(acqu.setTarget({'x_vals': np.zeros((1, N_OBJ)),
                                   'f_vals': np.zeros((1, N_OBJ)),
                                   'c_vals': np.zeros((1, 1))},
                                  lambda x, sx=0:
                                  np.ones(N_OBJ) * (0.01 - sum(x)))
                   < 1.0001))
    data = make_data(N_OBJ, N_OBJ)
    assert (acqu.setTarget(data, penalty) in data['x_vals'])


def test_RandomConstraint_scalarize():
    """ Check the RandomConstraint scalarization against the Pareto front.

    On the Pareto front the scalarized value must not exceed the sum of the
    objectives, unless the point violates the randomly drawn upper bound.

    """

    data = make_data(N_OBJ, N_OBJ)
    acqu = RandomConstraint(N_OBJ, np.zeros(N_OBJ), np.ones(N_OBJ), {})
    acqu.setTarget({'x_vals': None, 'f_vals': None}, penalty)
    acqu.setTarget(data, penalty)
    pf = updatePF(data, {})
    zeros = np.zeros(N_OBJ)
    # Check that the scalar value is either less than the sum of fi or bad
    for fi in pf['f_vals']:
        assert (acqu.scalarize(fi, zeros, zeros, zeros)
                <= np.sum(fi) + 1.0e-4 or np.any(fi > acqu.f_ub - 1.0e-4))
    # Check that the scalar grad is either less than sum of weights or bad
    grad = jacrev(acqu.scalarize, argnums=0)
    for fi in pf['f_vals']:
        assert (np.all(grad(fi, zeros, np.zeros(1), np.zeros(1))
                       <= np.sum(acqu.weights) + 1.0e-4)
                or np.any(fi > acqu.f_ub - 1.0e-4))


@pytest.mark.parametrize("sd_dim", [1, 2, 3])
def test_EI_RandomConstraint_scalarize(sd_dim):
    """ Check the EI_RandomConstraint scalarization against the Pareto front.

    The expected-improvement scalarization consumes a standard-deviation
    vector, so it is checked at several simulation dimensions.

    """

    data = make_data(N_OBJ, N_OBJ)
    acqu = EI_RandomConstraint(N_OBJ, np.zeros(N_OBJ), np.ones(N_OBJ), {})
    acqu.setTarget({'x_vals': None, 'f_vals': None}, penalty)
    acqu.setTarget(data, penalty)
    pf = updatePF(data, {})
    for fi in pf['f_vals']:
        assert (acqu.scalarize(fi, np.zeros(N_OBJ), np.zeros(sd_dim),
                               np.ones(sd_dim))
                <= np.sum(fi) + 1.0e-4 or np.any(fi > acqu.f_ub - 1.0e-4))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
