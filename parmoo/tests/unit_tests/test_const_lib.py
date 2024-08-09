def test_SingleSimBound():
    """ Test the SingleSimBound() function.

    Initialize an objective then evaluate.

    """

    from parmoo.constraints import SingleSimBound, SingleSimBoundGradient
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
    obj_func1 = SingleSimBound(xtype, stype, 'sim1', bound_type='lower')
    grad_func1 = SingleSimBoundGradient(xtype, stype, 'sim1', bound_type='lower')
    obj_func2 = SingleSimBound(xtype, stype, ('sim2', 1), bound_type='upper')
    grad_func2 = SingleSimBoundGradient(xtype, stype, ('sim2', 1), bound_type='upper')
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


def test_SumOfSimSquaresBound():
    """ Test the SumOfSimSquaresBound() function.

    Initialize an objective then evaluate.

    """

    from parmoo.constraints import SumOfSimSquaresBound, SumOfSimSquaresBoundGradient
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
    obj_func = SumOfSimSquaresBound(xtype, stype, sim_list, bound_type="lower")
    grad_func = SumOfSimSquaresBoundGradient(xtype, stype, sim_list, bound_type="lower")
    # Test function evaluation
    assert (np.abs(obj_func(x, sx) + 16.0) < 1.0e-8)
    # Test dx and ds evaluation
    xkeys = ["x1", "x2", "x3", "x4"]
    dx, ds = grad_func(x, sx)
    assert (all([dx[i] == 0.0 for i in xkeys]))
    assert (ds["sim1"] == -4.0)
    assert (np.all(ds["sim2"] == (np.eye(4)[3] - np.ones(4)) * 4.0))


def test_SumOfSimsBound():
    """ Test the SumOfSimsBound() function.

    Initialize an objective then evaluate.

    """

    from parmoo.constraints import SumOfSimsBound, SumOfSimsBoundGradient
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
    obj_func1 = SumOfSimsBound(xtype, stype, sim_list, bound_type="lower")
    grad_func1 = SumOfSimsBoundGradient(xtype, stype, sim_list, bound_type="lower")
    obj_func2 = SumOfSimsBound(xtype, stype, sim_list,
                               bound_type='upper', absolute=True)
    grad_func2 = SumOfSimsBoundGradient(xtype, stype, sim_list,
                                        bound_type='upper', absolute=True)
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
    test_SingleSimBound()
    test_SumOfSimSquaresBound()
    test_SumOfSimsBound()
