def test_dtlz1_obj():
    """ Test the dtlz1_obj() objective function.

    Initialize an objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives.dtlz import dtlz1_obj
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8")]
    # Create input vectors
    x = np.zeros(1, dtype=xtype)[0]
    sx = np.zeros(1, dtype=stype)[0]
    # Create the objective function
    obj1 = dtlz1_obj(xtype, stype, 0, num_obj=3)
    obj2 = dtlz1_obj(xtype, stype, 1, num_obj=3)
    obj3 = dtlz1_obj(xtype, stype, 2, num_obj=3)
    # Test function evaluation
    assert (np.abs(obj1(x, sx) - 0.0) < 1.0e-8)
    assert (np.abs(obj2(x, sx) - 0.0) < 1.0e-8)
    assert (np.abs(obj3(x, sx) - 0.5) < 1.0e-8)
    ## Test dx evaluation
    #df1x = np.zeros(1, dtype=xtype)[0]
    #df2x = np.zeros(1, dtype=xtype)[0]
    #df2x['x1'] = 0.5
    #df3x = np.zeros(1, dtype=xtype)[0]
    #df3x['x1'] = -0.5
    #assert (np.all([np.abs(obj1(x, sx, der=1)[name[0]] - df1x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    #assert (np.all([np.abs(obj2(x, sx, der=1)[name[0]] - df2x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    #assert (np.all([np.abs(obj3(x, sx, der=1)[name[0]] - df3x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    ## Test ds evaluation
    #dfds = np.zeros(1, dtype=stype)[0]
    #assert (np.abs(obj1(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)
    #assert (np.abs(obj2(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)
    #dfds['sim1'] = 0.5
    #assert (np.abs(obj3(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)


def test_dtlz2_obj():
    """ Test the dtlz2_obj() objective function.

    Initialize an objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives.dtlz import dtlz2_obj
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8")]
    # Create input vectors
    x = np.zeros(1, dtype=xtype)[0]
    sx = np.zeros(1, dtype=stype)[0]
    # Create the objective function
    obj1 = dtlz2_obj(xtype, stype, 0, num_obj=3)
    obj2 = dtlz2_obj(xtype, stype, 1, num_obj=3)
    obj3 = dtlz2_obj(xtype, stype, 2, num_obj=3)
    # Test function evaluation
    assert (np.abs(obj1(x, sx) - 1.0) < 1.0e-8)
    assert (np.abs(obj2(x, sx) - 0.0) < 1.0e-8)
    assert (np.abs(obj3(x, sx) - 0.0) < 1.0e-8)
    ## Test dx evaluation
    #df1x = np.zeros(1, dtype=xtype)[0]
    #df2x = np.zeros(1, dtype=xtype)[0]
    #df2x['x2'] = np.pi / 2.0
    #df3x = np.zeros(1, dtype=xtype)[0]
    #df3x['x1'] = np.pi / 2.0
    #assert (np.all([np.abs(obj1(x, sx, der=1)[name[0]] - df1x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    #assert (np.all([np.abs(obj2(x, sx, der=1)[name[0]] - df2x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    #assert (np.all([np.abs(obj3(x, sx, der=1)[name[0]] - df3x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    ## Test ds evaluation
    #dfds = np.zeros(1, dtype=stype)[0]
    #dfds['sim1'] = 1.0
    #assert (np.abs(obj1(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)
    #dfds['sim1'] = 0.0
    #assert (np.abs(obj2(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)
    #assert (np.abs(obj3(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)


def test_dtlz3_obj():
    """ Test the dtlz3_obj() objective function.

    Initialize an objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives.dtlz import dtlz3_obj
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8")]
    # Create input vectors
    x = np.zeros(1, dtype=xtype)[0]
    sx = np.zeros(1, dtype=stype)[0]
    # Create the objective function
    obj1 = dtlz3_obj(xtype, stype, 0, num_obj=3)
    obj2 = dtlz3_obj(xtype, stype, 1, num_obj=3)
    obj3 = dtlz3_obj(xtype, stype, 2, num_obj=3)
    # Test function evaluation
    assert (np.abs(obj1(x, sx) - 1.0) < 1.0e-8)
    assert (np.abs(obj2(x, sx) - 0.0) < 1.0e-8)
    assert (np.abs(obj3(x, sx) - 0.0) < 1.0e-8)
    ## Test dx evaluation
    #df1x = np.zeros(1, dtype=xtype)[0]
    #df2x = np.zeros(1, dtype=xtype)[0]
    #df2x['x2'] = np.pi / 2.0
    #df3x = np.zeros(1, dtype=xtype)[0]
    #df3x['x1'] = np.pi / 2.0
    #assert (np.all([np.abs(obj1(x, sx, der=1)[name[0]] - df1x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    #assert (np.all([np.abs(obj2(x, sx, der=1)[name[0]] - df2x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    #assert (np.all([np.abs(obj3(x, sx, der=1)[name[0]] - df3x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    ## Test ds evaluation
    #dfds = np.zeros(1, dtype=stype)[0]
    #dfds['sim1'] = 1.0
    #assert (np.abs(obj1(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)
    #dfds['sim1'] = 0.0
    #assert (np.abs(obj2(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)
    #assert (np.abs(obj3(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)


def test_dtlz4_obj():
    """ Test the dtlz4_obj() objective function.

    Initialize an objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives.dtlz import dtlz4_obj
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8")]
    # Create input vectors
    x = np.zeros(1, dtype=xtype)[0]
    sx = np.zeros(1, dtype=stype)[0]
    # Create the objective function
    obj1 = dtlz4_obj(xtype, stype, 0, num_obj=3)
    obj2 = dtlz4_obj(xtype, stype, 1, num_obj=3)
    obj3 = dtlz4_obj(xtype, stype, 2, num_obj=3)
    # Test function evaluation
    assert (np.abs(obj1(x, sx) - 1.0) < 1.0e-8)
    assert (np.abs(obj2(x, sx) - 0.0) < 1.0e-8)
    assert (np.abs(obj3(x, sx) - 0.0) < 1.0e-8)
    ## Test dx evaluation
    #df1x = np.zeros(1, dtype=xtype)[0]
    #df2x = np.zeros(1, dtype=xtype)[0]
    #df3x = np.zeros(1, dtype=xtype)[0]
    #assert (np.all([np.abs(obj1(x, sx, der=1)[name[0]] - df1x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    #assert (np.all([np.abs(obj2(x, sx, der=1)[name[0]] - df2x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    #assert (np.all([np.abs(obj3(x, sx, der=1)[name[0]] - df3x[name[0]])
    #                < 1.0e-8 for name in xtype]))
    ## Test ds evaluation
    #dfds = np.zeros(1, dtype=stype)[0]
    #dfds['sim1'] = 1.0
    #assert (np.abs(obj1(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)
    #dfds['sim1'] = 0.0
    #assert (np.abs(obj2(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)
    #assert (np.abs(obj3(x, sx, der=2)['sim1'] - dfds['sim1']) < 1.0e-8)


if __name__ == "__main__":
    test_dtlz1_obj()
    test_dtlz2_obj()
    test_dtlz3_obj()
    test_dtlz4_obj()
