def test_SingleSimObjective():
    """ Test the SingleSimObjective() function.

    Initialize an objective then evaluate.

    """

    from parmoo.objectives import SingleSimObjective, SingleSimGradient
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 2)]
    # Create input vectors
    x = np.zeros(1, dtype=xtype)[0]
    sx = np.zeros(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    sx['sim2'][1] = 3.0
    # Create the objective and gradient functions
    obj_func1 = SingleSimObjective(xtype, stype, 'sim1', goal='max')
    grad_func1 = SingleSimGradient(xtype, stype, 'sim1', goal='max')
    obj_func2 = SingleSimObjective(xtype, stype, ('sim2', 1), goal='min')
    grad_func2 = SingleSimGradient(xtype, stype, ('sim2', 1), goal='min')
    # Test function evaluations
    assert (obj_func1(x, sx) == -2.0)
    assert (obj_func2(x, sx) == 3.0)
    # Test dx and ds evaluation
    xkeys = ["x1", "x2", "x3"]
    dx, ds = grad_func1(x, sx)
    assert (all([dx[i] == 0.0 for i in xkeys]))
    assert (ds["sim1"] == -1.0)
    assert (np.all(ds["sim2"] == 0.0))
    dx, ds = grad_func2(x, sx)
    assert (all([dx[i] == 0.0 for i in xkeys]))
    assert (ds["sim1"] == 0.0)
    assert (np.all(ds["sim2"] == np.eye(2)[1]))


def test_SumOfSimSquaresObjective():
    """ Test the SumOfSimSquaresObjective() function.

    Initialize an objective then evaluate.

    """

    from parmoo.objectives import SumOfSimSquaresObjective, SumOfSimSquaresGradient
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 4)]
    # Create input vectors
    x = np.ones(1, dtype=xtype)[0]
    sx = np.ones(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    sx['sim2'][:] = 2.0
    # Create the objective and gradient functions
    sim_list = ['sim1', ('sim2', 0), ('sim2', 1), ('sim2', 2)]
    obj_func = SumOfSimSquaresObjective(xtype, stype, sim_list, goal="max")
    grad_func = SumOfSimSquaresGradient(xtype, stype, sim_list, goal="max")
    # Test function evaluation
    assert (np.abs(obj_func(x, sx) + 16.0) < 1.0e-8)
    # Test dx and ds evaluation
    xkeys = ["x1", "x2", "x3", "x4"]
    dx, ds = grad_func(x, sx)
    assert (all([dx[i] == 0.0 for i in xkeys]))
    assert (ds["sim1"] == -4.0)
    assert (np.all(ds["sim2"] == (np.eye(4)[3] - np.ones(4)) * 4.0))


def test_SumOfSimsObjective():
    """ Test the SumOfSimsObjective() function.

    Initialize an objective then evaluate.

    """

    from parmoo.objectives import SumOfSimsObjective, SumOfSimsGradient
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 4)]
    # Create input vectors
    x = np.ones(1, dtype=xtype)[0]
    sx = np.ones(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    sx['sim2'][:] = -2.0
    # Create the objective functions
    sim_list = ['sim1', ('sim2', 0), ('sim2', 1), ('sim2', 2)]
    obj_func1 = SumOfSimsObjective(xtype, stype, sim_list, goal="max")
    grad_func1 = SumOfSimsGradient(xtype, stype, sim_list, goal="max")
    obj_func2 = SumOfSimsObjective(xtype, stype, sim_list,
                                   goal='min', absolute=True)
    grad_func2 = SumOfSimsGradient(xtype, stype, sim_list,
                                   goal='min', absolute=True)
    # Test function evaluation
    assert (np.abs(obj_func1(x, sx) - 4.0) < 1.0e-8)
    assert (np.abs(obj_func2(x, sx) - 8.0) < 1.0e-8)
    # Test dx and ds evaluation
    xkeys = ["x1", "x2", "x3", "x4"]
    dx, ds = grad_func1(x, sx)
    assert (all([dx[i] == 0.0 for i in xkeys]))
    assert (ds["sim1"] == -1.0)
    assert (np.all(ds["sim2"] == (np.eye(4)[3] - np.ones(4))))
    dx, ds = grad_func2(x, sx)
    assert (all([dx[i] == 0.0 for i in xkeys]))
    assert (ds["sim1"] == 1.0)
    assert (np.all(ds["sim2"] == (np.eye(4)[3] - np.ones(4))))


if __name__ == "__main__":
    test_SingleSimObjective()
    test_SumOfSimSquaresObjective()
    test_SumOfSimsObjective()
