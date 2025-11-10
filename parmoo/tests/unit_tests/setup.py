
def makeNumpyDatabase(with_constraints=True):
    """ Create a NumpyDatabase object for testing.

    Args:
        with_constraints (bool, optional): An optional variable that can be
            used to create a database with no constraints (by setting to
            False). Defaults to True.

    Returns:
        NumpyDatabase: A database with 3 design variables ("x1", "x2", "x3"), 2
        simulations ("s1", "s2"), and 2 objectives ("f1", "f2").  If
        with_constraints is set (default) then there are also 2 constraints
        ("c1", "c2"); otherwise, there are no constraints.

    """

    from parmoo.databases import NumpyDatabase

    db = NumpyDatabase({})
    db.addDesign("x1", "f8", 0.01)
    db.addDesign("x2", "i4", 0)
    db.addDesign("x3", "U25", 0)
    db.addSimulation("s1", 1)
    db.addSimulation("s2", 4)
    db.addObjective("f1")
    db.addObjective("f2")
    if with_constraints:
        db.addConstraint("c1")
        db.addConstraint("c2")
    return db
