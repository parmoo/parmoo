""" Unit tests for MOOP simulation/objective database operations. """

import numpy as np
import pytest

from parmoo import MOOP
from parmoo.acquisitions import UniformWeights
from parmoo.optimizers import LocalSurrogate_PS
from parmoo.tests.unit_tests.helpers import (
    con_sim1,
    con_sim2_sum,
    con_x1,
    obj_sim1,
    obj_sim2_first,
    sim_dict,
    sim_sq_norm,
    sim_sq_norms_shifted,
    two_sim_moop,
)

# The five design points seeded into the objective database: the origin and the
# four unit vectors.  The origin is dominated by all four, so the Pareto front
# holds four of the five.
CORNERS = [{"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
           {"x1": 1.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
           {"x1": 0.0, "x2": 1.0, "x3": 0.0, "x4": 0.0},
           {"x1": 0.0, "x2": 0.0, "x3": 1.0, "x4": 0.0},
           {"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 1.0}]


def _distance_to_corner(x, index):
    """ The distance from x to the unit vector at position index, over 4
    variables.

    """

    others = [i for i in [1, 2, 3, 4] if i != index]
    return np.sqrt(sum([x[f"x{i}"] ** 2 for i in others])
                   + (x[f"x{index}"] - 1) ** 2)


def f1(x, s):
    return _distance_to_corner(x, 1)


def f2(x, s):
    return _distance_to_corner(x, 2)


def f3(x, s):
    return _distance_to_corner(x, 3)


def c1(x, s):
    """ A constraint satisfied only at the origin. """

    return -sum([x[f"x{i}"] for i in [1, 2, 3, 4]])


def four_var_moop(with_constraint=False):
    """ Build a compiled 4-variable, 3-objective, simulation-free MOOP. """

    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'name': f"x{i + 1}", 'lb': 0.0, 'ub': 1.0})
    for func in [f1, f2, f3]:
        moop.addObjective({'obj_func': func})
    if with_constraint:
        moop.addConstraint({'constraint': c1})
    for i in range(3):
        moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    return moop


def seed_corners(moop):
    """ Write the five corner points straight into the objective database.

    Going through the database directly, rather than through the solve loop,
    is what makes the resulting Pareto front exactly predictable.

    """

    sx = np.zeros(0)
    for point in CORNERS:
        data = dict(point)
        data["f1"], data["f2"], data["f3"] = (f1(data, sx), f2(data, sx),
                                              f3(data, sx))
        data["c1"] = c1(data, sx)
        moop.database.updateObjDb(data, data, data)


# ---------------------------------------------------------------------------
# Simulation database
# ---------------------------------------------------------------------------


@pytest.fixture
def named_sim_moop():
    """ A compiled MOOP whose two simulations have explicit names. """

    return two_sim_moop(objectives=(obj_sim1,), sim_names=("g1", "g2"))


def test_checkSimDb_rejects_unknown_simulation(named_sim_moop):
    """ Check that checkSimDb() rejects a simulation not in the schema. """

    with pytest.raises(ValueError):
        named_sim_moop.checkSimDb({"x1": 0, "x2": 0, "x3": 0}, "hello world")


def test_updateSimDb_rejects_unknown_simulation(named_sim_moop):
    """ Check that updateSimDb() rejects an out-of-range simulation index. """

    sx = np.zeros(1, dtype=named_sim_moop.getSimulationType())[0]
    with pytest.raises(ValueError):
        named_sim_moop.updateSimDb({"x1": 0, "x2": 0, "x3": 0}, sx, -1)


def test_evaluateSimulation_rejects_unknown_simulation(named_sim_moop):
    """ Check that evaluateSimulation() rejects an unknown simulation name. """

    with pytest.raises(ValueError):
        named_sim_moop.evaluateSimulation({"x1": 0, "x2": 0, "x3": 0}, "g6")


def test_checkSimDb_tracks_each_simulation_separately(named_sim_moop):
    """ Check that a point evaluated for one simulation is absent from another.

    The simulation databases are independent, so evaluating a point for "g1"
    must not make it appear complete for "g2".

    """

    x = {"x1": 0, "x2": 0, "x3": 0}
    y = {"x1": 1, "x2": 1, "x3": 1}
    named_sim_moop.evaluateSimulation(x, "g1")
    named_sim_moop.evaluateSimulation(y, "g1")
    named_sim_moop.evaluateSimulation(x, "g2")
    assert (named_sim_moop.checkSimDb(x, "g1") is not None)
    assert (named_sim_moop.checkSimDb(y, "g1") is not None)
    assert (named_sim_moop.checkSimDb(x, "g2") is not None)
    # y was never evaluated for g2
    assert (named_sim_moop.checkSimDb(y, "g2") is None)


def test_getSimulationData_shapes():
    """ Check that getSimulationData() reports one row per evaluation.

    The two simulations have one and two outputs respectively, so their result
    arrays must be shaped (n,) and (n, 2).

    """

    moop = MOOP(LocalSurrogate_PS, hyperparams={})
    for i in range(4):
        moop.addDesign({'name': f"x{i + 1}", 'lb': 0.0, 'ub': 1.0})
    moop.addSimulation(sim_dict(1, sim_sq_norm, "Bobo1"),
                       sim_dict(2, sim_sq_norms_shifted, "Bobo2"))
    moop.addObjective({'obj_func': obj_sim2_first})
    moop.addAcquisition({'acquisition': UniformWeights})
    moop.compile()
    # Before any evaluation both databases are empty
    soln = moop.getSimulationData()
    assert (soln['Bobo1']['out'].size == 0)
    assert (soln['Bobo2']['out'].size == 0)
    # Evaluate the origin and the four unit vectors for both simulations
    for point in CORNERS:
        for name in ["Bobo1", "Bobo2"]:
            moop.evaluateSimulation(point, name)
    soln = moop.getSimulationData()
    assert (soln['Bobo1']['out'].shape == (5,))
    assert (soln['Bobo2']['out'].shape == (5, 2))


# ---------------------------------------------------------------------------
# Objective database
# ---------------------------------------------------------------------------


def test_getObjectiveData_returns_every_point():
    """ Check that getObjectiveData() returns the whole database.

    Unlike getPF(), it does not filter for nondominated points.

    """

    moop = four_var_moop(with_constraint=True)
    seed_corners(moop)
    assert (moop.getObjectiveData().shape[0] == 5)


def test_getPF_returns_only_nondominated_points():
    """ Check that getPF() filters the database down to the Pareto front.

    Of the five seeded points the origin is dominated by all four unit
    vectors, so exactly four survive.

    """

    moop = four_var_moop(with_constraint=True)
    seed_corners(moop)
    soln = moop.getPF()
    assert (soln.shape[0] == 4)
    for name in ["f1", "f2", "f3"]:
        assert (soln[name].size == 4)


DEDUP_OBJECTIVES = (obj_sim2_first, obj_sim1)


def test_addObjData_deduplicates():
    """ Check that addObjData() drops a point already in the database.

    The origin is added twice but must be stored once, leaving two distinct
    points alongside the all-ones corner.

    """

    moop = two_sim_moop(objectives=DEDUP_OBJECTIVES)
    x0 = moop._extract(np.zeros(moop.n_latent))
    s0 = moop._unpack_sim(np.zeros(3))
    moop.addObjData(x0, s0)
    moop.addObjData(x0, s0)
    moop.addObjData(moop._extract(np.ones(moop.n_latent)),
                    moop._unpack_sim(np.ones(3)))
    assert (len(moop.getObjectiveData()) == 2)


def test_addObjData_deduplicates_with_constraints():
    """ Check that deduplication still holds once constraints are recorded.

    Adding constraint values widens each database row, but duplicate detection
    is on the design point, so the repeated origin is still stored once.

    """

    moop = two_sim_moop(objectives=DEDUP_OBJECTIVES,
                        constraints=(con_x1, con_sim1, con_sim2_sum))
    s0 = moop._unpack_sim(np.zeros(3))
    x0 = moop._extract(np.zeros(moop.n_latent))
    moop.addObjData(x0, s0)
    moop.addObjData(x0, s0)
    moop.addObjData(moop._extract(np.eye(moop.n_latent)[2]), s0)
    moop.addObjData(moop._extract(np.ones(moop.n_latent)),
                    moop._unpack_sim(np.ones(3)))
    assert (len(moop.getObjectiveData()) == 3)


def test_addObjData_with_categorical_variable():
    """ Check that addObjData() handles a mixed continuous/categorical design.

    The categorical variable expands to two latent coordinates, so the latent
    point handed to _extract() is wider than the feature space.

    """

    moop = two_sim_moop(objectives=DEDUP_OBJECTIVES,
                        constraints=(con_x1, con_sim1, con_sim2_sum),
                        compile_moop=False)
    moop.addDesign({'des_type': "categorical", 'levels': ["L1", "L2", "L3"]})
    moop.compile()
    moop.addObjData(moop._extract(np.ones(moop.n_latent)),
                    moop._unpack_sim(np.ones(3)))
    assert (len(moop.getObjectiveData()) == 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
