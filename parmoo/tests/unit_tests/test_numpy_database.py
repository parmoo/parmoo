from parmoo.databases import NumpyDatabase
from parmoo.tests.unit_tests.setup import makeNumpyDatabase


def test_NumpyDatabase_construct_no_run():
    """ Create a new NumpyDatabase and check that it looks correct on init.

    Also checks that no add/get methods can be called on an uninitialized
    database.

    """

    import pytest

    # Create a truly empty database and check it is empty
    db = NumpyDatabase({})
    assert db.isEmpty()

    # Try to add a data point to the database before starting
    with pytest.raises(RuntimeError):
        db.updateSimDb(
            {"x1": 0.0, "x2": 0, "x3": "0"},
            [1],
            "s1"
        )
    with pytest.raises(RuntimeError):
        db.updateObjDb(
            {"x1": 0.0, "x2": 0, "x3": "0"},
            {"f1": 0.0, "f2": 0.0},
            {"c1": 0.0, "c2": 0.0}
        )

    # Try to check data before starting
    with pytest.raises(RuntimeError):
        db.checkSimDb({"x1": 0.0, "x2": 0, "x3": "0"}, "s1")
    with pytest.raises(RuntimeError):
        db.checkObjDb({"x1": 0.0, "x2": 0, "x3": "0"})

    # Try to get data before starting
    with pytest.raises(RuntimeError):
        for xi, si in db.browseCompleteSimulations():
            continue
    with pytest.raises(RuntimeError):
        db.getSimulationData()
    with pytest.raises(RuntimeError):
        db.getNewSimulationData()
    with pytest.raises(RuntimeError):
        db.getObjectiveData()
    with pytest.raises(RuntimeError):
        db.getPF()


def test_NumpyDatabase_add_get_types():
    """ Check the NumpyDatabase handles adding and getting types from schema.

    Initialize a NumpyDatabase class and add design variables, simulations,
    objectives, and constraints to the schema.  After each, check the schema
    using the get*Type() methods and ensure the dtype is as expected.

    """

    import numpy as np
    import pytest

    db = NumpyDatabase({})

    expected_des_type = []
    assert db.getDesignType() is None

    db.addDesign("x1", "f8", 0.01)
    expected_des_type.append(("x1", "f8"))
    assert db.getDesignType() == np.dtype(expected_des_type)

    db.addDesign("x2", "i4", 0)
    expected_des_type.append(("x2", "i4"))
    assert db.getDesignType() == np.dtype(expected_des_type)

    db.addDesign("x3", "U25", 0)
    expected_des_type.append(("x3", "U25"))
    assert db.getDesignType() == np.dtype(expected_des_type)

    expected_sim_type = []
    assert db.getSimulationType() is None

    db.addSimulation("s1", 1)
    expected_sim_type.append(("s1", "f8"))
    assert db.getSimulationType() == np.dtype(expected_sim_type)

    db.addSimulation("s2", 4)
    expected_sim_type.append(("s2", "f8", 4))
    assert db.getSimulationType() == np.dtype(expected_sim_type)

    expected_obj_type = []
    assert db.getObjectiveType() is None

    db.addObjective("f1")
    expected_obj_type.append(("f1", "f8"))
    assert db.getObjectiveType() == np.dtype(expected_obj_type)

    db.addObjective("f2")
    expected_obj_type.append(("f2", "f8"))
    assert db.getObjectiveType() == np.dtype(expected_obj_type)

    expected_con_type = []
    assert db.getConstraintType() is None

    db.addConstraint("c1")
    expected_con_type.append(("c1", "f8"))
    assert db.getConstraintType() == np.dtype(expected_con_type)

    db.addConstraint("c2")
    expected_con_type.append(("c2", "f8"))
    assert db.getConstraintType() == np.dtype(expected_con_type)

    with pytest.raises(TypeError):
        db.addDesign(1, "i4", 0)
    with pytest.raises(ValueError):
        db.addDesign("x1", "i4", 0)
    with pytest.raises(ValueError):
        db.addDesign("s1", "i4", 0)
    with pytest.raises(ValueError):
        db.addDesign("f1", "i4", 0)
    with pytest.raises(ValueError):
        db.addDesign("c1", "i4", 0)
    assert db.getDesignType() == np.dtype(expected_des_type)
    with pytest.raises(TypeError):
        db.addSimulation(1, 4)
    with pytest.raises(ValueError):
        db.addSimulation("x1", 4)
    with pytest.raises(ValueError):
        db.addSimulation("s1", 4)
    with pytest.raises(ValueError):
        db.addSimulation("f1", 4)
    with pytest.raises(ValueError):
        db.addSimulation("c1", 4)
    assert db.getSimulationType() == np.dtype(expected_sim_type)
    with pytest.raises(TypeError):
        db.addObjective(1)
    with pytest.raises(ValueError):
        db.addObjective("x1")
    with pytest.raises(ValueError):
        db.addObjective("s1")
    with pytest.raises(ValueError):
        db.addObjective("f1")
    with pytest.raises(ValueError):
        db.addObjective("c1")
    assert db.getObjectiveType() == np.dtype(expected_obj_type)
    with pytest.raises(TypeError):
        db.addConstraint(1)
    with pytest.raises(ValueError):
        db.addConstraint("x1")
    with pytest.raises(ValueError):
        db.addConstraint("s1")
    with pytest.raises(ValueError):
        db.addConstraint("f1")
    with pytest.raises(ValueError):
        db.addConstraint("c1")
    assert db.getConstraintType() == np.dtype(expected_con_type)


def test_NumpyDatabase_simulation_database():
    """ Test the simulation database operations.

    Create a NumpyDatabase, start the database, check, update, browse, and get
    the simulation database contents.

    """

    import numpy as np
    import pytest

    # Create an empty database (with schema) and check it is empty
    db = makeNumpyDatabase()
    assert db.isEmpty()

    # Start the database and add check that all the lookups return empty
    db.startDatabase()
    assert db.isEmpty()
    assert db.checkSimDb({"x1": 1.0e-16, "x2": 0, "x3": "0"}, "s1") is None
    assert db.checkSimDb({"x1": 0.0, "x2": 0, "x3": "0"}, "s2") is None
    assert len([(xi, si) for xi, si in db.browseCompleteSimulations()]) == 0
    assert len(db.getSimulationData()["s1"]) == 0
    assert len(db.getSimulationData()["s2"]) == 0
    assert len(db.getNewSimulationData()["s1"]) == 0
    assert len(db.getNewSimulationData()["s2"]) == 0

    # Add an output for 1 simulation at 1 data point
    db.updateSimDb(
        {"x1": 0.0, "x2": 0, "x3": "0"},
        [1],
        "s1"
    )
    # Database is now non-empty and contains 1 data entry for s1
    assert not db.isEmpty()
    assert db.checkSimDb({"x1": 1.0e-16, "x2": 0, "x3": "0"}, "s1") == 1
    assert db.checkSimDb({"x1": 1.0, "x2": 1, "x3": "1"}, "s1") is None
    assert db.checkSimDb({"x1": 0.0, "x2": 0, "x3": "0"}, "s2") is None
    assert db.checkSimDb({"x1": 1.0, "x2": 1, "x3": "1"}, "s2") is None
    # Check the iteration over all completed sim data (should be empty)
    assert len([(xi, si) for xi, si in db.browseCompleteSimulations()]) == 0
    # Check "s1" has 1 new sim data point, "s2" has none
    new_sim_data = db.getNewSimulationData()
    assert len(new_sim_data["s1"]) == 1
    assert new_sim_data["s1"][0]["x1"] == 0.0
    assert new_sim_data["s1"][0]["x2"] == 0
    assert new_sim_data["s1"][0]["x3"] == "0"
    assert new_sim_data["s1"][0]["out"] == 1
    assert len(new_sim_data["s2"]) == 0
    # Check "s1" has 1 total sim data points, "s2" has none
    assert len(db.getSimulationData()["s1"]) == 1
    assert db.getSimulationData()["s1"][0]["x1"] == 0.0
    assert db.getSimulationData()["s1"][0]["x2"] == 0
    assert db.getSimulationData()["s1"][0]["x3"] == "0"
    assert db.getSimulationData()["s1"][0]["out"] == 1
    assert len(db.getSimulationData()["s2"]) == 0

    # Add an output for the other simulation at the same data point
    db.updateSimDb(
        {"x1": 0.0, "x2": 0, "x3": "0"},
        [1, 1, 1, 1],
        "s2"
    )
    assert not db.isEmpty()
    assert db.checkSimDb({"x1": 1.0e-16, "x2": 0, "x3": "0"}, "s1") == 1
    assert db.checkSimDb({"x1": 1.0, "x2": 1, "x3": "1"}, "s1") is None
    assert np.all(
        db.checkSimDb({"x1": 0.0, "x2": 0, "x3": "0"}, "s2") == np.ones(4)
    )
    assert db.checkSimDb({"x1": 1.0, "x2": 1, "x3": "1"}, "s2") is None
    # Check the iteration over all completed sim data (contains 1 pt)
    assert len([(xi, si) for xi, si in db.browseCompleteSimulations()]) == 1
    for xi, si in db.browseCompleteSimulations():
        assert xi["x1"] == 0.0
        assert xi["x2"] == 0
        assert xi["x3"] == "0"
        assert si["s1"] == 1
        assert np.all(si["s2"] == np.ones(4))
    # Check "s2" has 1 new sim data point, "s1" has none
    new_sim_data = db.getNewSimulationData()
    assert len(new_sim_data["s1"]) == 0
    assert len(new_sim_data["s2"]) == 1
    assert new_sim_data["s2"][0]["x1"] == 0.0
    assert new_sim_data["s2"][0]["x2"] == 0
    assert new_sim_data["s2"][0]["x3"] == "0"
    assert np.all(new_sim_data["s2"][0]["out"] == np.ones(4))
    # Check both "s1" and "s2" have 1 total sim data points (pandas format)
    assert len(db.getSimulationData(format="pandas")["s1"]) == 1
    assert db.getSimulationData(format="pandas")["s1"].iloc[0]["x1"] == 0.0
    assert db.getSimulationData(format="pandas")["s1"].iloc[0]["x2"] == 0
    assert db.getSimulationData(format="pandas")["s1"].iloc[0]["x3"] == "0"
    assert db.getSimulationData(format="pandas")["s1"].iloc[0]["out"] == 1
    assert len(db.getSimulationData(format="pandas")["s2"]) == 1
    assert db.getSimulationData(format="pandas")["s2"].iloc[0]["x1"] == 0.0
    assert db.getSimulationData(format="pandas")["s2"].iloc[0]["x2"] == 0
    assert db.getSimulationData(format="pandas")["s2"].iloc[0]["x3"] == "0"
    assert db.getSimulationData(format="pandas")["s2"].iloc[0]["out_0"] == 1.0
    assert db.getSimulationData(format="pandas")["s2"].iloc[0]["out_1"] == 1.0
    assert db.getSimulationData(format="pandas")["s2"].iloc[0]["out_2"] == 1.0
    assert db.getSimulationData(format="pandas")["s2"].iloc[0]["out_3"] == 1.0

    # Add another output for "s1" at a new data point
    db.updateSimDb(
        {"x1": 1.0, "x2": 1, "x3": "1"},
        2,
        "s1"
    )
    assert not db.isEmpty()
    assert db.checkSimDb({"x1": 1.0e-16, "x2": 0, "x3": "0"}, "s1") == 1
    assert db.checkSimDb({"x1": 1.0, "x2": 1, "x3": "1"}, "s1") == 2
    assert np.all(
        db.checkSimDb({"x1": 0.0, "x2": 0, "x3": "0"}, "s2") == np.ones(4)
    )
    assert db.checkSimDb({"x1": 1.0, "x2": 1, "x3": "1"}, "s2") is None
    # Check the iteration over all completed sim data (contains 1 pt)
    assert len([(xi, si) for xi, si in db.browseCompleteSimulations()]) == 1
    for xi, si in db.browseCompleteSimulations():
        assert xi["x1"] == 0.0
        assert xi["x2"] == 0
        assert xi["x3"] == "0"
        assert si["s1"] == 1
        assert np.all(si["s2"] == np.ones(4))
    # Check "s1" has 1 new sim data point, "s2" has none
    new_sim_data = db.getNewSimulationData()
    assert len(new_sim_data["s1"]) == 1
    assert new_sim_data["s1"][0]["x1"] == 1.0
    assert new_sim_data["s1"][0]["x2"] == 1
    assert new_sim_data["s1"][0]["x3"] == "1"
    assert new_sim_data["s1"][0]["out"] == 2
    assert len(new_sim_data["s2"]) == 0
    # Check "s1" has 2 and "s2" has 1 total sim data points
    assert len(db.getSimulationData()["s1"]) == 2
    assert db.getSimulationData()["s1"][0]["x1"] == 0.0
    assert db.getSimulationData()["s1"][0]["x2"] == 0
    assert db.getSimulationData()["s1"][0]["x3"] == "0"
    assert db.getSimulationData()["s1"][0]["out"] == 1
    assert db.getSimulationData()["s1"][1]["x1"] == 1.0
    assert db.getSimulationData()["s1"][1]["x2"] == 1
    assert db.getSimulationData()["s1"][1]["x3"] == "1"
    assert db.getSimulationData()["s1"][1]["out"] == 2
    assert len(db.getSimulationData()["s2"]) == 1
    assert db.getSimulationData()["s2"][0]["x1"] == 0.0
    assert db.getSimulationData()["s2"][0]["x2"] == 0
    assert db.getSimulationData()["s2"][0]["x3"] == "0"
    assert np.all(db.getSimulationData()["s2"][0]["out"] == np.ones(4))

    # Try to add a non-existent simulation to the database
    with pytest.raises(ValueError):
        db.updateSimDb(
            {"x1": 0.0, "x2": 0, "x3": "0"},
            [1, 1, 1, 1],
            "simulation2"
        )
    # Try to retrieve an invalid format
    with pytest.raises(ValueError):
        db.getSimulationData(format="BadFormatString")
    # Try to re-start a running database
    with pytest.raises(RuntimeError):
        db.startDatabase()


def test_NumpyDatabase_objective_database():
    """ Test the objective database operations.

    Create a NumpyDatabase, start the database, check, update, and get
    the objective database contents.

    """

    import pytest

    # Create an empty database (with schema) and check it is empty
    db = makeNumpyDatabase()
    assert db.isEmpty()

    # Start the database and add check that all the lookups return empty
    db.startDatabase()
    assert db.isEmpty()
    assert db.checkObjDb({"x1": 0.0, "x2": 0, "x3": "0"}) is None
    assert db.checkObjDb({"x1": 0.0, "x2": 0, "x3": "0"}) is None
    assert len(db.getObjectiveData()) == 0
    assert len(db.getPF()) == 0

    # Add an output at 1 data point
    db.updateObjDb(
        {"x1": 0.0, "x2": 0, "x3": "0"},
        {"f1": 0.5, "f2": 0.5},
        {"c1": 0.0, "c2": 1.0},
    )
    # Database is now non-empty and contains 1 data entry
    assert not db.isEmpty()
    assert db.checkObjDb({"x1": 1.0e-16, "x2": 0, "x3": "0"}) is not None
    assert db.checkObjDb({"x1": 1.0e-16, "x2": 0, "x3": "0"})[0]['f1'] == 0.5
    assert db.checkObjDb({"x1": 1.0e-16, "x2": 0, "x3": "0"})[0]['f2'] == 0.5
    assert db.checkObjDb({"x1": 1.0e-16, "x2": 0, "x3": "0"})[1]['c1'] == 0.0
    assert db.checkObjDb({"x1": 1.0e-16, "x2": 0, "x3": "0"})[1]['c2'] == 1.0
    assert db.checkObjDb({"x1": 1.0, "x2": 1, "x3": "1"}) is None
    # Check the objective database has 1 entry
    assert len(db.getObjectiveData()) == 1
    assert db.getObjectiveData()[0]["x1"] == 0.0
    assert db.getObjectiveData()[0]["x2"] == 0
    assert db.getObjectiveData()[0]["x3"] == "0"
    assert db.getObjectiveData()[0]["f1"] == 0.5
    assert db.getObjectiveData()[0]["f2"] == 0.5
    assert db.getObjectiveData()[0]["c1"] == 0.0
    assert db.getObjectiveData()[0]["c2"] == 1.0
    # Check the PF is empty (the above point is infeasible)
    assert len(db.getPF()) == 0

    # Add an output at another data point
    db.updateObjDb(
        {"x1": 1.0, "x2": 1, "x3": "1"},
        {"f1": 1.0, "f2": 1.0},
        {"c1": 0.0, "c2": 0.0},
    )
    # Database is now non-empty and contains another data entry
    assert not db.isEmpty()
    assert db.checkObjDb({"x1": 1.0, "x2": 1, "x3": "1"}) is not None
    assert db.checkObjDb({"x1": 1.0, "x2": 1, "x3": "1"})[0]['f1'] == 1.0
    assert db.checkObjDb({"x1": 1.0, "x2": 1, "x3": "1"})[0]['f2'] == 1.0
    assert db.checkObjDb({"x1": 1.0, "x2": 1, "x3": "1"})[1]['c1'] == 0.0
    assert db.checkObjDb({"x1": 1.0, "x2": 1, "x3": "1"})[1]['c2'] == 0.0
    # Check the objective database has 2 entries (test pandas format)
    assert len(db.getObjectiveData(format="pandas")) == 2
    assert db.getObjectiveData(format="pandas").iloc[0]["x1"] == 0.0
    assert db.getObjectiveData(format="pandas").iloc[0]["x2"] == 0
    assert db.getObjectiveData(format="pandas").iloc[0]["x3"] == "0"
    assert db.getObjectiveData(format="pandas").iloc[0]["f1"] == 0.5
    assert db.getObjectiveData(format="pandas").iloc[0]["f2"] == 0.5
    assert db.getObjectiveData(format="pandas").iloc[0]["c1"] == 0.0
    assert db.getObjectiveData(format="pandas").iloc[0]["c2"] == 1.0
    assert db.getObjectiveData(format="pandas").iloc[1]["x1"] == 1.0
    assert db.getObjectiveData(format="pandas").iloc[1]["x2"] == 1
    assert db.getObjectiveData(format="pandas").iloc[1]["x3"] == "1"
    assert db.getObjectiveData(format="pandas").iloc[1]["f1"] == 1.0
    assert db.getObjectiveData(format="pandas").iloc[1]["f2"] == 1.0
    assert db.getObjectiveData(format="pandas").iloc[1]["c1"] == 0.0
    assert db.getObjectiveData(format="pandas").iloc[1]["c2"] == 0.0
    # Check the PF contains just the above point (test pandas format)
    assert len(db.getPF(format="pandas")) == 1
    assert db.getPF(format="pandas").iloc[0]["x1"] == 1.0
    assert db.getPF(format="pandas").iloc[0]["x2"] == 1
    assert db.getPF(format="pandas").iloc[0]["x3"] == "1"
    assert db.getPF(format="pandas").iloc[0]["f1"] == 1.0
    assert db.getPF(format="pandas").iloc[0]["f2"] == 1.0
    assert db.getPF(format="pandas").iloc[0]["c1"] == 0.0
    assert db.getPF(format="pandas").iloc[0]["c2"] == 0.0

    # Add 2 more Pareto optimal data points
    db.updateObjDb(
        {"x1": 0.0, "x2": 1, "x3": "1"},
        {"f1": 0.0, "f2": 1.0},
        {"c1": 0.0, "c2": 0.0},
    )
    db.updateObjDb(
        {"x1": 1.0, "x2": 0, "x3": "1"},
        {"f1": 1.0, "f2": 0.0},
        {"c1": 0.0, "c2": 0.0},
    )
    # Just check the summary stats
    assert not db.isEmpty()
    assert len(db.getObjectiveData()) == 4
    assert len(db.getPF()) == 2
    # Detailed check of PF
    for xi in db.getPF():
        assert xi["x1"] + xi["x2"] == 1
        assert xi["x3"] == "1"
        assert xi["f1"] + xi["f2"] == 1.0
        assert xi["c1"] == xi["c2"] == 0.0

    # Re-create another database without any constraints
    db = makeNumpyDatabase(with_constraints=False)
    db.startDatabase()
    db.updateObjDb(
        {"x1": 0.5, "x2": 1, "x3": "1"},
        {"f1": 0.5, "f2": 0.5}, {}
    )
    db.updateObjDb(
        {"x1": 1.0, "x2": 1, "x3": "0"},
        {"f1": 1.0, "f2": 1.0}, {}
    )
    db.updateObjDb(
        {"x1": 0.0, "x2": 1, "x3": "1"},
        {"f1": 0.0, "f2": 1.0}, {}
    )
    db.updateObjDb(
        {"x1": 0.0, "x2": 1, "x3": "0"},
        {"f1": 0.0, "f2": 2.0}, {}
    )
    db.updateObjDb(
        {"x1": 1.0, "x2": 0, "x3": "0"},
        {"f1": 2.0, "f2": 0.0}, {}
    )
    db.updateObjDb(
        {"x1": 1.0, "x2": 0, "x3": "1"},
        {"f1": 1.0, "f2": 0.0}, {}
    )
    # Check the summary stats
    assert not db.isEmpty()
    assert len(db.getObjectiveData()) == 6
    assert len(db.getPF()) == 3
    # Detailed check of PF
    for xi in db.getPF():
        assert xi["x1"] + xi["x2"] == 1 or xi["x1"] + xi["x2"] == 1.5
        assert xi["x3"] == "1"
        assert xi["f1"] + xi["f2"] == 1.0

    # Try to retrieve an invalid format
    with pytest.raises(ValueError):
        db.getObjectiveData(format="BadFormatString")
    with pytest.raises(ValueError):
        db.getPF(format="BadFormatString")
    # Try to re-start a running database
    with pytest.raises(RuntimeError):
        db.startDatabase()


def test_NumpyDatabase_checkpoint():
    """ Test the database checkpointing operations.

    Create a NumpyDatabase, set the checkpointing, add some data, and reload
    that data.

    """

    import numpy as np
    import os
    import pytest

    # Create a new database, start it, and add 1 data point
    db1 = makeNumpyDatabase()
    db1.startDatabase()
    db1.updateSimDb(
        {"x1": 0.0, "x2": 0, "x3": "0"},
        [1],
        "s1"
    )
    db1.updateSimDb(
        {"x1": 0.0, "x2": 0, "x3": "0"},
        [1, 1, 1, 1],
        "s2"
    )
    db1.updateObjDb(
        {"x1": 0.0, "x2": 0, "x3": "0"},
        {"f1": 0.0, "f2": 1.0},
        {"c1": 0.0, "c2": 1.0},
    )
    # Activate checkpointing and add another 1-and-a-half data points
    db1.setCheckpoint(True)
    db1.getNewSimulationData()  # Mark all the following data as "new"
    db1.updateSimDb(
        {"x1": 1.0, "x2": 1, "x3": "1"},
        [2],
        "s1"
    )
    db1.updateSimDb(
        {"x1": 1.0, "x2": 1, "x3": "1"},
        [2, 2, 2, 2],
        "s2"
    )
    db1.updateObjDb(
        {"x1": 1.0, "x2": 1, "x3": "1"},
        {"f1": 1.0, "f2": 0.0},
        {"c1": 1.0, "c2": 0.0},
    )
    db1.updateSimDb(
        {"x1": 0.0, "x2": 0, "x3": "1"},
        [1],
        "s1"
    )
    # Reload into another database
    db2 = NumpyDatabase({})
    db2.loadCheckpoint()
    # Check all the dtypes match
    assert db2.getDesignType() == db1.getDesignType()
    assert db2.getSimulationType() == db1.getSimulationType()
    assert db2.getObjectiveType() == db1.getObjectiveType()
    assert db2.getConstraintType() == db1.getConstraintType()
    # Check the simulation databases match
    for key in db2.getSimulationData():
        assert (
            len(db2.getSimulationData()[key]) ==
            len(db1.getSimulationData()[key])
        )
        assert np.all(
            db2.getSimulationData()[key] ==
            db1.getSimulationData()[key]
        )
    # Check the objective databases match
    assert len(db2.getObjectiveData()) == len(db1.getObjectiveData())
    assert np.all(db2.getObjectiveData() == db1.getObjectiveData())
    # Check new data in db1 and db2 match
    new_db1_data = db1.getNewSimulationData()
    new_db2_data = db2.getNewSimulationData()
    for key in db2.getSimulationData():
        assert len(new_db2_data[key]) == len(new_db1_data[key])
        assert np.all(new_db2_data[key]['out'] == new_db1_data[key]['out'])

    # Cleanup
    os.remove("parmoo.simdb.json")

    # Try some bad checkpointing
    with pytest.raises(TypeError):
        db2.setCheckpoint("false")
    with pytest.raises(TypeError):
        db2.setCheckpoint(True, 5)


if __name__ == "__main__":
    test_NumpyDatabase_construct_no_run()
    test_NumpyDatabase_add_get_types()
    test_NumpyDatabase_simulation_database()
    test_NumpyDatabase_objective_database()
    test_NumpyDatabase_checkpoint()
