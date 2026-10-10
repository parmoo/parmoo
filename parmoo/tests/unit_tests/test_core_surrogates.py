""" Unit tests for MOOP_base surrogate fitting, updating, and evaluation. """

import numpy as np
import pytest

from parmoo.tests.unit_tests.helpers import (
    TRAINING_POINTS,
    obj_sim1,
    obj_sim2_sum,
    obj_x1,
    train_surrogates,
    two_sim_moop,
)

OBJECTIVES = (obj_x1, obj_sim1, obj_sim2_sum)

# The exact simulation outputs at each of TRAINING_POINTS: the first
# simulation is the norm of the design point, and the second is the norms of
# the point shifted by 1.0 and by 0.5.
EXPECTED_SIM_VALUES = [
    (np.zeros(3), np.array([0.0, np.sqrt(3), np.sqrt(0.75)])),
    (np.ones(3) / 2, np.array([np.sqrt(0.75), np.sqrt(0.75), 0.0])),
    (np.eye(3)[0], np.array([1.0, np.sqrt(2), np.sqrt(0.75)])),
    (np.eye(3)[1], np.array([1.0, np.sqrt(2), np.sqrt(0.75)])),
    (np.eye(3)[2], np.array([1.0, np.sqrt(2), np.sqrt(0.75)])),
    (np.ones(3), np.array([np.sqrt(3), 0.0, np.sqrt(0.75)])),
]


@pytest.fixture
def trained_moop():
    """ A compiled MOOP whose surrogates are fit to all TRAINING_POINTS. """

    moop = two_sim_moop(objectives=OBJECTIVES)
    train_surrogates(moop)
    return moop


def test_fit_then_update_matches_fit_once():
    """ Check that fitting incrementally matches fitting all at once.

    One MOOP is fit to all six training points at once.  A second is fit to
    the first three and then updated with the remaining three.  Their
    surrogates must agree everywhere.

    """

    batch = two_sim_moop(objectives=OBJECTIVES)
    train_surrogates(batch)

    incremental = two_sim_moop(objectives=OBJECTIVES)
    train_surrogates(incremental, points=TRAINING_POINTS[:3])
    train_surrogates(incremental, points=TRAINING_POINTS[3:], update=True)

    for xi in np.random.sample((5, 3)):
        assert (np.linalg.norm(batch._evaluate_surrogates(xi) -
                               incremental._evaluate_surrogates(xi)) < 1.0e-8)


def test_evaluate_surrogates_interpolates(trained_moop):
    """ Check that the fitted surrogates reproduce the training data. """

    for xi, si in EXPECTED_SIM_VALUES:
        assert (np.linalg.norm(trained_moop._evaluate_surrogates(xi) - si)
                < 1.0e-7)


def test_surrogate_uncertainty_vanishes_at_training_points(trained_moop):
    """ Check that the surrogate uncertainty is ~0 where data was observed. """

    for xi, _ in EXPECTED_SIM_VALUES:
        assert (np.linalg.norm(trained_moop._surrogate_uncertainty(xi))
                < 1.0e-3)


def test_surrogate_uncertainty_is_positive_away_from_data(trained_moop):
    """ Check that the surrogate reports uncertainty away from its data. """

    xi = np.ones(3) * 0.75
    assert (np.linalg.norm(trained_moop._surrogate_uncertainty(xi)) > 1.0e-4)


def test_evaluate_surrogates_is_invariant_to_design_scaling():
    """ Check that rescaling the design space does not change the surrogate.

    The latent space is always the unit cube, so a MOOP declared over
    [-1, 1] x [0, 2] x [-0.5, 1.5] must produce the same surrogate values as
    one declared over the unit cube, once each is given the same feature-space
    design points.

    """

    unit = two_sim_moop(objectives=OBJECTIVES)
    train_surrogates(unit)
    scaled = two_sim_moop(objectives=OBJECTIVES,
                          bounds=[(-1.0, 1.0), (0.0, 2.0), (-0.5, 1.5)])
    train_surrogates(scaled, tr_center=np.zeros(3), tr_radius=np.inf)

    for point in [{'x1': 0, 'x2': 0, 'x3': 0}, {'x1': 1, 'x2': 1, 'x3': 1}]:
        assert (np.linalg.norm(
            unit._evaluate_surrogates(unit._embed(point)) -
            scaled._evaluate_surrogates(scaled._embed(point))) < 1.0e-8)


def test_evaluateSimulation_rejects_unknown_simulation():
    """ Check that evaluateSimulation() rejects an out-of-range index. """

    moop = two_sim_moop(objectives=OBJECTIVES)
    with pytest.raises(ValueError):
        moop.evaluateSimulation(np.zeros(3), -1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
