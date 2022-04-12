def test_single_sim_out_unnamed():
    """ Test the single_sim_out() objective function.

    Initialize an unnamed objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives import single_sim_out
    import numpy as np

    # Create the objective function
    obj_func = single_sim_out(3, 3, 0, goal='min')
    # Test function evaluation
    assert(obj_func(np.zeros(3), np.eye(3)[0] * 1.5) == 1.5)
    # Test dx evaluation
    assert(np.all(obj_func(np.zeros(3), np.ones(3) * 4.0, der=1) ==
                  np.zeros(3)))
    # Test ds evaluation
    assert(np.all(obj_func(np.zeros(3), np.ones(3) * 3.0, der=2) ==
                  np.eye(3)[0]))


def test_single_sim_out_named1():
    """ Test the single_sim_out() objective function.

    Initialize a named objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives import single_sim_out
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 2)]
    # Create input vectors
    x = np.zeros(1, dtype=xtype)[0]
    sx = np.zeros(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    # Create the objective function
    obj_func = single_sim_out(xtype, stype, 'sim1', goal='max')
    # Test function evaluation
    assert(obj_func(x, sx) == -2.0)
    # Test dx evaluation
    assert(np.all(obj_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_out = np.zeros(1, dtype=stype)[0]
    s_out['sim1'] = -1.0
    assert(np.all(obj_func(x, sx, der=2) == s_out))


def test_single_sim_out_named2():
    """ Test the single_sim_out() objective function.

    Initialize a named objective, then evaluate the function value and
    derivative with respect to x and sx.

    Choose a simulation with multiple outputs.

    """

    from parmoo.objectives import single_sim_out
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 2)]
    # Create input vectors
    x = np.zeros(1, dtype=xtype)[0]
    sx = np.zeros(1, dtype=stype)[0]
    sx['sim2'][0] = 1.0
    sx['sim2'][1] = 2.0
    # Create the objective function
    obj_func = single_sim_out(xtype, stype, ('sim2', 1), goal='min')
    # Test function evaluation
    assert(obj_func(x, sx) == 2.0)
    # Test dx evaluation
    assert(np.all(obj_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_out = np.zeros(1, dtype=stype)[0]
    s_out['sim2'][1] = 1.0
    assert(np.all(obj_func(x, sx, der=2) == s_out))


def test_sos_sim_out_unnamed():
    """ Test the sos_sim_out() objective function.

    Initialize an unnamed objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives import sos_sim_out
    import numpy as np

    # Create the objective function
    obj_func = sos_sim_out(5, 5, [0, 1, 2, 3], goal='min')
    # Test function evaluation
    assert(np.abs(obj_func(np.zeros(5), np.ones(5) * 2.0) - 16.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(obj_func(np.zeros(5), np.ones(5) * 2.0, der=1) ==
                  np.zeros(5)))
    # Test ds evaluation
    s_out = np.ones(5) * 4.0
    s_out[4] = 0.0
    assert(np.all(np.abs(obj_func(np.zeros(5), np.ones(5) * 2.0, der=2) -
                         s_out) < 1.0e-8))


def test_sos_sim_out_named():
    """ Test the sos_sim_out() objective function.

    Initialize a named objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives import sos_sim_out
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 4)]
    # Create input vectors
    x = np.ones(1, dtype=xtype)[0]
    sx = np.ones(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    sx['sim2'][:] = 2.0
    # Create the objective function
    obj_func = sos_sim_out(xtype, stype,
                           ['sim1', ('sim2', 0), ('sim2', 1), ('sim2', 2)],
                           goal='max')
    # Test function evaluation
    assert(np.abs(obj_func(x, sx) + 16.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(obj_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_out = np.zeros(1, dtype=stype)[0]
    s_out['sim1'] = sx['sim1'] * -2.0
    s_out['sim2'][:-1] = sx['sim2'][:-1] * -2.0
    assert(np.abs(obj_func(x, sx, der=2)['sim1'] - s_out['sim1']) < 1.0e-8)
    assert(np.all(np.abs(obj_func(x, sx, der=2)['sim2'] - s_out['sim2'])
                  < 1.0e-8))


def test_sum_sim_out_unnamed():
    """ Test the sum_sim_out() objective function.

    Initialize an unnamed objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives import sum_sim_out
    import numpy as np

    # Create the objective function
    obj_func = sum_sim_out(5, 5, [0, 1, 2, 3], goal='min')
    # Test function evaluation
    assert(np.abs(obj_func(np.zeros(5), np.ones(5) * 2.0) - 8.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(obj_func(np.zeros(5), np.ones(5) * 2.0, der=1) ==
                  np.zeros(5)))
    # Test ds evaluation
    assert(np.all(np.abs(obj_func(np.zeros(5), np.ones(5) * 2.0, der=2) -
                         (np.ones(5) - np.eye(5)[4])) < 1.0e-8))


def test_sum_sim_out_named():
    """ Test the sum_sim_out() objective function.

    Initialize a named objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives import sum_sim_out
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 4)]
    # Create input vectors
    x = np.ones(1, dtype=xtype)[0]
    sx = np.ones(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    sx['sim2'][:] = 2.0
    # Create the objective function
    obj_func = sum_sim_out(xtype, stype,
                           ['sim1', ('sim2', 0), ('sim2', 1), ('sim2', 2)],
                           goal='max')
    # Test function evaluation
    assert(np.abs(obj_func(x, sx) + 8.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(obj_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_out = np.ones(1, dtype=stype)[0]
    s_out['sim1'] *= -1.0
    s_out['sim2'][:] *= -1.0
    s_out['sim2'][3] = 0.0
    assert(np.abs(obj_func(x, sx, der=2)['sim1'] - s_out['sim1']) < 1.0e-8)
    assert(np.all(np.abs(obj_func(x, sx, der=2)['sim2'] - s_out['sim2'])
                  < 1.0e-8))


def test_sum_sim_out_named_abs():
    """ Test the sum_sim_out() objective function with "absolute=True" option.

    Initialize a named objective, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.objectives import sum_sim_out
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 4)]
    # Create input vectors
    x = np.ones(1, dtype=xtype)[0]
    sx = np.ones(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    sx['sim2'][:] = -2.0
    # Create the objective function
    obj_func = sum_sim_out(xtype, stype,
                           ['sim1', ('sim2', 0), ('sim2', 1), ('sim2', 2)],
                           goal='min', absolute=True)
    # Test function evaluation
    assert(np.abs(obj_func(x, sx) - 8.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(obj_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_out = np.zeros(1, dtype=stype)[0]
    s_out['sim1'] = 1.0
    s_out['sim2'][:-1] = -1.0
    assert(np.abs(obj_func(x, sx, der=2)['sim1'] - s_out['sim1']) < 1.0e-8)
    assert(np.all(np.abs(obj_func(x, sx, der=2)['sim2'] - s_out['sim2'])
                  < 1.0e-8))


if __name__ == "__main__":
    test_single_sim_out_unnamed()
    test_single_sim_out_named1()
    test_single_sim_out_named2()
    test_sos_sim_out_unnamed()
    test_sos_sim_out_named()
    test_sum_sim_out_unnamed()
    test_sum_sim_out_named()
    test_sum_sim_out_named_abs()
