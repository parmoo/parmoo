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


if __name__ == "__main__":
    test_single_sim_out_unnamed()
    test_single_sim_out_named1()
    test_single_sim_out_named2()
