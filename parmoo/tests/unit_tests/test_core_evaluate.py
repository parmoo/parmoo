""" Unit tests for MOOP_base objective, constraint, and penalty evaluation. """

import numpy as np
import pytest

from parmoo.tests.unit_tests.helpers import (
    con_sim1_first,
    con_sim2_pair_sum,
    con_x1,
    con_x1_offset,
    obj_sim1_first,
    obj_sim2_pair_sum,
    obj_x1,
    train_surrogates,
    two_sim_moop,
)

# Objectives (and, separately, constraints) built from the same three
# expressions: the first design variable, the first simulation output, and the
# sum of the second simulation's two outputs.
OBJECTIVES = (obj_x1, obj_sim1_first, obj_sim2_pair_sum)
CONSTRAINTS = (con_x1, con_sim1_first, con_sim2_pair_sum)

# The exact value of those three expressions at each training point.  Used for
# both the objective and the constraint checks, since the expressions match.
EXPECTED_VALUES = [
    (np.zeros(3), np.array([0.0, 0.0, np.sqrt(3) + np.sqrt(0.75)])),
    (np.ones(3) * 0.5, np.array([0.5, np.sqrt(0.75), np.sqrt(0.75)])),
    (np.eye(3)[0], np.array([1.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
    (np.eye(3)[1], np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
    (np.eye(3)[2], np.array([0.0, 1.0, np.sqrt(2) + np.sqrt(0.75)])),
    (np.ones(3), np.array([1.0, np.sqrt(3), np.sqrt(0.75)])),
]


def test_evaluate_objectives():
    """ Check that _evaluate_objectives() computes the objectives exactly.

    Each objective reads either a design variable or a simulation output, so
    with interpolating surrogates the results must be exact.

    """

    moop = two_sim_moop(objectives=OBJECTIVES)
    train_surrogates(moop)
    for xi, fi in EXPECTED_VALUES:
        sxi = moop._evaluate_surrogates(xi)
        assert (np.linalg.norm(moop._evaluate_objectives(xi, sxi) - fi)
                < 1.0e-8)


def test_evaluate_constraints_with_no_constraints():
    """ Check that an unconstrained MOOP evaluates to a single zero.

    ParMOO always returns a constraint vector, so with nothing to enforce it
    must be a length 1 array of zeros.

    """

    moop = two_sim_moop(objectives=(obj_x1,))
    assert (np.all(moop._evaluate_constraints(np.zeros(3), np.zeros(3))
                   == np.zeros(1)))


def test_evaluate_constraints():
    """ Check that _evaluate_constraints() computes the constraints exactly.

    The MOOP is compiled once without constraints and then re-compiled after
    they are added, which is legal while the database is still empty.

    """

    moop = two_sim_moop(objectives=(obj_x1,))
    for con in CONSTRAINTS:
        moop.addConstraint({'constraint': con})
    moop.compile()
    train_surrogates(moop, tr_center=np.zeros(3), tr_radius=np.inf)
    for xi, ci in EXPECTED_VALUES:
        sxi = moop._evaluate_surrogates(xi)
        assert (np.linalg.norm(moop._evaluate_constraints(xi, sxi) - ci)
                < 1.0e-8)


def test_evaluate_penalty():
    """ Check that _evaluate_penalty() adds the constraint violation.

    The single constraint is x1 - 0.5, so it is violated by 0.5 exactly at the
    two training points where x1 == 1, and satisfied everywhere else.  The
    penalty must equal the objectives plus that violation.

    """

    moop = two_sim_moop(objectives=OBJECTIVES, constraints=(con_x1_offset,))
    train_surrogates(moop)
    violated = {tuple(np.eye(3)[0]), tuple(np.ones(3))}
    for xi, fi in EXPECTED_VALUES:
        expected = fi + (0.5 if tuple(xi) in violated else 0.0)
        sxi = moop._evaluate_surrogates(xi)
        assert (np.linalg.norm(moop._evaluate_penalty(xi, sxi) - expected)
                < 1.0e-8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
