""" Unit tests for the parmoo.surrogates plugins.

Note on structure:

 * fit(), update(), and setTrustRegion() each validate their inputs with a
   block that is copy-pasted verbatim between gaussian_proc.py and
   polynomial.py, so each copy is a distinct set of coverage lines and the
   shared contract is parametrized over both classes rather than deleted.
 * improve() is implemented once in the SurrogateFunction ABC and is not
   overridden by either class, so both classes share its logic.  It is still
   parametrized over both, because the branch improve() takes depends on the
   design tolerances and fitted data of the specific model.
 * The Gaussian RBF is exercised under both a global trust region (radius
   inf) and a local one, since those follow different code paths through
   evaluate() and stdDev().

"""

import os

import numpy as np
import pytest
from jax import jacrev

from parmoo.surrogates import GaussRBF, Linear

# Both concrete surrogates, for the copy-pasted validation contracts
SURROGATES = [GaussRBF, Linear]

# Per-class fitting setup: the Linear model needs only n+1 points and is only
# accurate inside a trust region, while the RBF is fit globally.
NPTS = {GaussRBF: 10, Linear: 3}
TR_RADIUS = {GaussRBF: 0.5, Linear: 0.5}

SAVE_FILE = "parmoo.surrogate"


def random_data(npts, n=3, m=2):
    """ Generate npts random training pairs in n dims with m outputs. """

    return (np.random.random_sample((npts, n)),
            np.random.random_sample((npts, m)))


def fitted_pair(cls, hyperparams=None):
    """ Fit the same data two ways: fit-then-update, and fit all at once.

    Returns:
        tuple: (incremental model, batch model, all x values, all y values).

    """

    if hyperparams is None:
        hyperparams = {}
    npts = NPTS[cls]
    x1, y1 = random_data(npts)
    x2, y2 = random_data(npts)
    x_full = np.concatenate((x1, x2), axis=0)
    y_full = np.concatenate((y1, y2), axis=0)

    incremental = cls(2, np.zeros(3), np.ones(3), hyperparams)
    incremental.fit(x1, y1)
    incremental.update(x2, y2)
    incremental.update(np.zeros((0, 3)), np.zeros((0, 2)))  # no-op update
    batch = cls(2, np.zeros(3), np.ones(3), hyperparams)
    batch.fit(x_full, y_full)

    radius = np.ones(3) * TR_RADIUS[cls]
    incremental.setTrustRegion(0.5 * np.ones(3), radius)
    batch.setTrustRegion(0.5 * np.ones(3), radius)
    return incremental, batch, x_full, y_full


def quadratic_model(cls, hyperparams=None):
    """ Fit cls to y = x.x on the unit simplex vertices plus the centroid. """

    x_vals = np.append(np.eye(3), [[0.5, 0.5, 0.5]], axis=0)
    y_vals = np.asarray([[np.dot(xi, xi)] for xi in x_vals])
    model = cls(1, np.zeros(3), np.ones(3),
                hyperparams if hyperparams is not None else {})
    model.fit(x_vals, y_vals)
    return model, x_vals, y_vals


# ---------------------------------------------------------------------------
# Shared input validation, duplicated in gaussian_proc.py and polynomial.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_bad_des_tols(cls):
    """ Check the 'des_tols' hyperparameter contract on both surrogates. """

    # des_tols must be strictly positive
    with pytest.raises(ValueError):
        cls(2, np.zeros(3), np.ones(3), {'des_tols': np.zeros(3)})
    # des_tols must have length n
    with pytest.raises(ValueError):
        cls(2, np.zeros(3), np.ones(3), {'des_tols': np.zeros(2)})
    # des_tols must be an array, not a scalar
    with pytest.raises(TypeError):
        cls(2, np.zeros(3), np.ones(3), {'des_tols': 0.1})


@pytest.mark.parametrize("cls", SURROGATES)
@pytest.mark.parametrize("method", ["fit", "update"])
def test_surrogate_bad_training_data(cls, method):
    """ Check the fit()/update() training data contract on both surrogates. """

    model = cls(2, np.zeros(3), np.ones(3), {})
    call = getattr(model, method)
    x, y = random_data(NPTS[cls])
    # x and f must both be numpy arrays
    with pytest.raises(TypeError):
        call(0, y)
    with pytest.raises(TypeError):
        call(x, 0)
    # each row of f must have length m
    with pytest.raises(ValueError):
        call(np.zeros((10, 3)), np.zeros((10, 3)))
    # x and f must have equal length
    with pytest.raises(ValueError):
        call(np.zeros((10, 3)), np.zeros((9, 2)))


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_fit_no_data(cls):
    """ Check that fit() rejects an empty training set on both surrogates.

    Unlike update(), which accepts a no-op empty batch, fit() requires data.

    """

    model = cls(2, np.zeros(3), np.ones(3), {})
    with pytest.raises(ValueError):
        model.fit(np.zeros((0, 3)), np.zeros((0, 2)))


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_setTrustRegion_bad_center(cls):
    """ Check the setTrustRegion() center contract on both surrogates. """

    model = fitted_pair(cls)[0]
    # center must be a numpy array
    with pytest.raises(TypeError):
        model.setTrustRegion(5, np.ones(3))
    # center must have length n
    with pytest.raises(ValueError):
        model.setTrustRegion(np.zeros(5), np.ones(3))
    # center must be feasible
    with pytest.raises(ValueError):
        model.setTrustRegion(-np.ones(3), np.ones(3))


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_setTrustRegion_bad_radius(cls):
    """ Check the setTrustRegion() radius contract on both surrogates. """

    model = fitted_pair(cls)[0]
    center = np.ones(3) * 0.5
    # radius must be an array or float
    with pytest.raises(TypeError):
        model.setTrustRegion(center, 5)
    # radius must have length n
    with pytest.raises(ValueError):
        model.setTrustRegion(center, np.ones(5))
    # radius must be positive
    with pytest.raises(ValueError):
        model.setTrustRegion(center, -np.ones(3))


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_improve_bad_x(cls):
    """ Check the improve() contract, which SurrogateFunction implements. """

    model = fitted_pair(cls)[0]
    # x must be a numpy array-like object
    with pytest.raises(TypeError):
        model.improve(5, False)
    # x must have length n.  Matched on the message, because a wrongly
    # sized x would also raise a bare ValueError from the broadcast in
    # the feasibility check just below the guard.
    with pytest.raises(ValueError, match="x must have length n"):
        model.improve(np.zeros(2), False)
    # x must be feasible
    with pytest.raises(ValueError):
        model.improve(-np.ones(3), False)


# ---------------------------------------------------------------------------
# Shared behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_fit_then_update_matches_fit_once(cls):
    """ Check that incremental updates produce the same model as one fit.

    Fitting half the data and updating with the rest must be numerically
    indistinguishable from fitting everything at once -- in value, in
    uncertainty, and in gradient.

    """

    incremental, batch, x_full, _ = fitted_pair(cls)
    x = np.random.random_sample(3)
    assert (all(incremental.evaluate(x) == batch.evaluate(x)))
    for xi in x_full:
        assert (np.linalg.norm(jacrev(incremental.evaluate)(xi) -
                               jacrev(batch.evaluate)(xi)) < 1.0e-8)


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_interpolates_in_trust_region(cls):
    """ Check that both surrogates interpolate their data within a small TR.

    """

    incremental, batch, x_full, y_full = fitted_pair(cls)
    for xi, yi in zip(x_full, y_full):
        radius = np.ones(3) * 0.1
        incremental.setTrustRegion(xi, radius)
        batch.setTrustRegion(xi, radius)
        assert (np.linalg.norm(incremental.evaluate(xi) - yi) < 1.0e-8)
        assert (np.linalg.norm(batch.evaluate(xi) - yi) < 1.0e-8)


@pytest.mark.parametrize("cls", SURROGATES)
@pytest.mark.parametrize("global_improv", [False, True])
def test_surrogate_improve_stays_feasible(cls, global_improv):
    """ Check that improve() only ever suggests points inside the bounds. """

    model, x_vals, _ = quadratic_model(cls, {'tail_order': 0})
    model.setTrustRegion(x_vals[-1], np.ones(3) * 0.5)
    for i in range(4):
        x_improv = model.improve(np.zeros(3), global_improv)
        assert (np.all(x_improv[0] <= np.ones(3)) and
                np.all(x_improv[0] >= np.zeros(3)))


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_improve_escapes_bunched_points(cls):
    """ Check that improve() moves away from a tight cluster of points.

    When the n+1 nearest neighbors are all within the design tolerance,
    improve() must fall back to sampling away from the cluster rather than
    returning a duplicate.

    """

    model, x_vals, _ = quadratic_model(cls, {'tail_order': 0})
    center = np.asarray([0.5, 0.5, 0.5])
    x_new = np.ones((3, 3)) * 0.5
    f_new = np.zeros((3, 1))
    for i in range(3):
        x_new[i, i] = 0.50000001
        f_new[i, 0] = np.dot(x_new[i, :], x_new[i, :])
    model.update(x_new, f_new)
    model.setTrustRegion(center, np.ones(3) * 1.0e-4)
    x_improv = model.improve(center, False)
    assert (np.all(x_improv[0] <= np.ones(3)) and
            np.all(x_improv[0] >= np.zeros(3)))
    assert (np.linalg.norm(x_improv[0] - center) > 1.0e-8)


@pytest.mark.parametrize("cls", SURROGATES)
@pytest.mark.parametrize("global_improv", [False, True])
def test_surrogate_improve_escapes_design_tolerance(cls, global_improv):
    """ Check that improve() can escape a very large design tolerance.

    With a 1D design space of width 1 and a design tolerance of 0.3, the only
    points distinguishable from the two training points lie in the tails.

    """

    model = cls(1, np.zeros(1), np.ones(1), {'des_tols': 0.3 * np.ones(1)})
    x_vals = np.asarray([[0.4], [0.6]])
    model.fit(x_vals, np.asarray([[0.4], [0.6]]))
    model.setTrustRegion(np.ones(1) * 0.5, np.ones(1) * 0.5)
    for i in range(5):
        x_improv = model.improve(x_vals[0], global_improv)
        assert (x_improv[0][0] < 0.1 or x_improv[0][0] > 0.9)


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_fits_degenerate_plane_data(cls):
    """ Check that both surrogates tolerate data confined to a line.

    All four training points vary only along x1 and share the same output, so
    the fit is rank-deficient in the remaining two dimensions.

    """

    x_vals = np.zeros((4, 3))
    x_vals[1:, 0] = [0.1, 0.2, 0.3]
    y_vals = np.ones((4, 2))
    model = cls(2, np.zeros(3), np.ones(3), {})
    model.fit(x_vals, y_vals)
    model.update(x_vals, y_vals)
    model.setTrustRegion(x_vals[-1], np.ones(3) * 0.2)
    assert (np.linalg.norm(model.evaluate(x_vals[-1]) - y_vals[-1]) < 1.0e-8)


@pytest.mark.parametrize("cls", SURROGATES)
def test_surrogate_save_load_roundtrip(cls):
    """ Check that a saved model reloads into an identical model. """

    source = fitted_pair(cls)[0]
    target = cls(2, np.zeros(3), np.ones(3), {})
    try:
        source.save(SAVE_FILE)
        target.load(SAVE_FILE)
    finally:
        if os.path.exists(SAVE_FILE):
            os.remove(SAVE_FILE)
    radius = np.ones(3) * TR_RADIUS[cls]
    source.setTrustRegion(0.5 * np.ones(3), radius)
    target.setTrustRegion(0.5 * np.ones(3), radius)
    x = np.random.random_sample(3)
    assert (np.all(source.evaluate(x) == target.evaluate(x)))


# ---------------------------------------------------------------------------
# GaussRBF specifics: the nugget, and the standard deviation model
# ---------------------------------------------------------------------------


def test_GaussRBF_bad_nugget():
    """ Check the GaussRBF-specific 'nugget' hyperparameter contract. """

    with pytest.raises(TypeError):
        GaussRBF(2, np.zeros(3), np.ones(3), {'nugget': []})
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3), {'nugget': -1.0})
    # A legal nugget, combined with legal des_tols, is accepted
    GaussRBF(2, np.zeros(3), np.ones(3),
             {'nugget': 0.1, 'des_tols': np.ones(3) * 0.1})


@pytest.mark.parametrize("hyperparams", [{'nugget': 0.0001}, {}])
def test_GaussRBF_fits_redundant_data(hyperparams):
    """ Check that GaussRBF fits a duplicated design point.

    A repeated point makes the interpolation matrix singular, which is handled
    either by an explicit nugget or by the adaptive nugget when none is given.

    """

    x_vals, y_vals = random_data(10)
    x_vals = np.append(x_vals, np.asarray([x_vals[0, :]]), axis=0)
    y_vals = np.append(y_vals, np.asarray([y_vals[0, :]]), axis=0)
    x2, y2 = random_data(10)
    rbf = GaussRBF(2, np.zeros(3), np.ones(3), hyperparams)
    rbf.fit(x_vals, y_vals)
    rbf.update(x2, y2)
    rbf.setTrustRegion(x_vals[0], np.ones(3) * 0.1)
    assert (np.all(np.isfinite(rbf.evaluate(x_vals[0]))))
    assert (np.all(np.isfinite(jacrev(rbf.evaluate)(x_vals[0]))))


@pytest.mark.parametrize("tail_order", [0, 1])
def test_GaussRBF_tail_order(tail_order):
    """ Check that GaussRBF interpolates with a constant and a linear tail.

    The 'tail_order' hyperparameter selects the polynomial tail that is fit
    and subtracted before the radial basis solve: order 0 removes the mean,
    and order 1 additionally removes a least-squares linear trend.

    """

    incremental, batch, x_full, y_full = fitted_pair(
        GaussRBF, {'tail_order': tail_order})
    for xi, yi in zip(x_full, y_full):
        radius = np.ones(3) * 0.1
        incremental.setTrustRegion(xi, radius)
        batch.setTrustRegion(xi, radius)
        assert (np.linalg.norm(incremental.evaluate(xi) - yi) < 1.0e-8)
        assert (np.linalg.norm(batch.evaluate(xi) - yi) < 1.0e-8)


@pytest.mark.parametrize("tail_order, center, radius, expected", [
    (None, np.zeros(3), np.ones(3) * np.inf, -0.03661401),
    (0, np.asarray([0.5, 0.5, 0.5]), np.ones(3) * 0.25, -0.08798618),
])
def test_GaussRBF_known_gradient(tail_order, center, radius, expected):
    """ Check the RBF gradient against a hand-computed value.

    Fit y = x.x on the unit simplex vertices plus the centroid, then check the
    gradient at the centroid.  The expected value differs between a global
    trust region and a local one, because the RBF is rescaled to the region.

    """

    hyperparams = {} if tail_order is None else {'tail_order': tail_order}
    model, x_vals, _ = quadratic_model(GaussRBF, hyperparams)
    model.setTrustRegion(center, radius)
    grad = jacrev(model.evaluate)(x_vals[-1])
    assert (np.linalg.norm(grad - expected * np.ones(3)) < 1.0e-4)


def test_GaussRBF_interpolates_with_global_trust_region():
    """ Check that GaussRBF interpolates exactly under an unbounded TR.

    With an infinite trust-region radius the RBF is a global interpolant, so
    it must reproduce every training value and report near-zero uncertainty
    there.

    """

    incremental, batch, x_full, y_full = fitted_pair(GaussRBF)
    unbounded = np.ones(3) * np.inf
    incremental.setTrustRegion(np.zeros(3), unbounded)
    batch.setTrustRegion(np.zeros(3), unbounded)
    x = np.random.random_sample(3)
    assert (all(incremental.stdDev(x) == batch.stdDev(x)))
    for xi, yi in zip(x_full, y_full):
        assert (np.linalg.norm(incremental.evaluate(xi) - yi) < 1.0e-8)
        assert (np.linalg.norm(batch.evaluate(xi) - yi) < 1.0e-8)
        assert (np.max(incremental.stdDev(xi)) < 1.0e-4)
        assert (np.max(batch.stdDev(xi)) < 1.0e-4)
    for xi in x_full:
        assert (np.linalg.norm(jacrev(incremental.stdDev)(xi) -
                               jacrev(batch.stdDev)(xi)) < 1.0e-4)


@pytest.mark.parametrize("center, radius", [
    (np.zeros(3), np.ones(3) * np.inf),
    (np.asarray([0.5, 0.5, 0.5]), np.ones(3) * 0.25),
])
def test_GaussRBF_stddev_is_nonneg_and_varies(center, radius):
    """ Check that the RBF uncertainty is nonnegative with a nonzero gradient.

    """

    model, _, _ = quadratic_model(GaussRBF, {'tail_order': 0})
    model.setTrustRegion(center, radius)
    xi = np.random.random_sample(3)
    assert (np.all(model.stdDev(xi) >= 0))
    assert (np.any(jacrev(model.stdDev)(xi) != 0))


@pytest.mark.parametrize("center, radius, sd_floor", [
    (np.zeros(1), np.ones(1) * np.inf, 1.0e-2),
    (np.array([0.5]), np.ones(1) * 0.25, 5.0e-3),
])
def test_GaussRBF_stddev_peaks_between_two_points(center, radius, sd_floor):
    """ Check the 1D uncertainty model against its analytic shape.

    Interpolating the two endpoints of [0, 1], the RBF must be exact at the
    midpoint while its uncertainty is maximized there -- so stdDev increases
    across the first half of the interval and decreases across the second.

    """

    model = GaussRBF(1, np.zeros(1), np.ones(1), {'tail_order': 0})
    model.fit(np.array([[0.0], [1.0]]), np.array([[0.0], [1.0]]))
    model.setTrustRegion(center, radius)
    mid = np.array([0.5])
    assert (np.linalg.norm(model.evaluate(mid) - 0.5) < 1.0e-8)
    assert (np.linalg.norm(model.stdDev(mid)) > sd_floor)
    assert (np.linalg.norm(jacrev(model.evaluate)(mid)) > 1.0)
    # The uncertainty is stationary at the midpoint by symmetry
    assert (np.linalg.norm(jacrev(model.stdDev)(mid)) < 1.0e-4)
    # Sweep the interval: stdDev rises to the midpoint, then falls
    model.setTrustRegion(np.array([0.5]), np.ones(1) * np.inf)
    xx = np.linspace(0, 1).reshape((50, 1))
    maxind = 0
    for i, xi in enumerate(xx):
        if np.all(model.stdDev(xi) > model.stdDev(xx[maxind])):
            maxind = i
        if i < 25:
            assert (np.all(jacrev(model.stdDev)(xi) >= 0))
        else:
            assert (np.all(jacrev(model.stdDev)(xi) <= 0))
    assert (maxind in [24, 25])


# ---------------------------------------------------------------------------
# Linear specifics
# ---------------------------------------------------------------------------


def test_Linear_is_exact_on_constant_data():
    """ Check that the linear model reproduces a constant exactly.

    A linear model fit to two equal values must return that value with zero
    gradient.

    """

    model = Linear(1, np.zeros(1), np.ones(1), {})
    model.fit(np.array([[0], [1]]), np.array([[1], [1]]))
    model.setTrustRegion(np.array([0.5]), np.ones(1) * 0.1)
    mid = np.array([0.5])
    assert (np.linalg.norm(model.evaluate(mid) - 1.0) < 1.0e-8)
    assert (np.linalg.norm(jacrev(model.evaluate)(mid)) < 1.0e-8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
