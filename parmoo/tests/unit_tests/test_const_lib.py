def test_single_sim_bound_unnamed():
    """ Test the single_sim_bound() constraint function.

    Initialize an unnamed constraint, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.constraints import single_sim_bound
    import numpy as np

    # Create the constraint function
    const_func = single_sim_bound(3, 3, 0, type='upper')
    # Test function evaluation
    assert(const_func(np.zeros(3), np.eye(3)[0] * 1.5) == 1.5)
    # Test dx evaluation
    assert(np.all(const_func(np.zeros(3), np.ones(3) * 4.0, der=1) ==
                  np.zeros(3)))
    # Test ds evaluation
    assert(np.all(const_func(np.zeros(3), np.ones(3) * 3.0, der=2) ==
                  np.eye(3)[0]))


def test_single_sim_bound_named1():
    """ Test the single_sim_bound() constraint function.

    Initialize a named constraint, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.constraints import single_sim_bound
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 2)]
    # Create input vectors
    x = np.zeros(1, dtype=xtype)[0]
    sx = np.zeros(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    # Create the constraint function
    const_func = single_sim_bound(xtype, stype, 'sim1', type='lower')
    # Test function evaluation
    assert(const_func(x, sx) == -2.0)
    # Test dx evaluation
    assert(np.all(const_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_bound = np.zeros(1, dtype=stype)[0]
    s_bound['sim1'] = -1.0
    assert(np.all(const_func(x, sx, der=2) == s_bound))


def test_single_sim_bound_named2():
    """ Test the single_sim_bound() constraint function.

    Initialize a named constraint, then evaluate the function value and
    derivative with respect to x and sx.

    Choose a simulation with multiple boundputs.

    """

    from parmoo.constraints import single_sim_bound
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 2)]
    # Create input vectors
    x = np.zeros(1, dtype=xtype)[0]
    sx = np.zeros(1, dtype=stype)[0]
    sx['sim2'][0] = 1.0
    sx['sim2'][1] = 2.0
    # Create the constraint function
    const_func = single_sim_bound(xtype, stype, ('sim2', 1), type='upper')
    # Test function evaluation
    assert(const_func(x, sx) == 2.0)
    # Test dx evaluation
    assert(np.all(const_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_bound = np.zeros(1, dtype=stype)[0]
    s_bound['sim2'][1] = 1.0
    assert(np.all(const_func(x, sx, der=2) == s_bound))


def test_sos_sim_bound_unnamed():
    """ Test the sos_sim_bound() constraint function.

    Initialize an unnamed constraint, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.constraints import sos_sim_bound
    import numpy as np

    # Create the constraint function
    const_func = sos_sim_bound(5, 5, [0, 1, 2, 3], type='upper')
    # Test function evaluation
    assert(np.abs(const_func(np.zeros(5), np.ones(5) * 2.0) - 16.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(const_func(np.zeros(5), np.ones(5) * 2.0, der=1) ==
                  np.zeros(5)))
    # Test ds evaluation
    s_bound = np.ones(5) * 4.0
    s_bound[4] = 0.0
    assert(np.all(np.abs(const_func(np.zeros(5), np.ones(5) * 2.0, der=2) -
                         s_bound) < 1.0e-8))


def test_sos_sim_bound_named():
    """ Test the sos_sim_bound() constraint function.

    Initialize a named constraint, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.constraints import sos_sim_bound
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 4)]
    # Create input vectors
    x = np.ones(1, dtype=xtype)[0]
    sx = np.ones(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    sx['sim2'][:] = 2.0
    # Create the constraint function
    const_func = sos_sim_bound(xtype, stype,
                           ['sim1', ('sim2', 0), ('sim2', 1), ('sim2', 2)],
                           type='lower')
    # Test function evaluation
    assert(np.abs(const_func(x, sx) + 16.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(const_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_bound = np.zeros(1, dtype=stype)[0]
    s_bound['sim1'] = sx['sim1'] * -2.0
    s_bound['sim2'][:-1] = sx['sim2'][:-1] * -2.0
    assert(np.abs(const_func(x, sx, der=2)['sim1'] - s_bound['sim1']) < 1.0e-8)
    assert(np.all(np.abs(const_func(x, sx, der=2)['sim2'] - s_bound['sim2'])
                  < 1.0e-8))


def test_sum_sim_bound_unnamed():
    """ Test the sum_sim_bound() constraint function.

    Initialize an unnamed constraint, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.constraints import sum_sim_bound
    import numpy as np

    # Create the constraint function
    const_func = sum_sim_bound(5, 5, [0, 1, 2, 3], type='upper')
    # Test function evaluation
    assert(np.abs(const_func(np.zeros(5), np.ones(5) * 2.0) - 8.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(const_func(np.zeros(5), np.ones(5) * 2.0, der=1) ==
                  np.zeros(5)))
    # Test ds evaluation
    assert(np.all(np.abs(const_func(np.zeros(5), np.ones(5) * 2.0, der=2) -
                         (np.ones(5) - np.eye(5)[4])) < 1.0e-8))


def test_sum_sim_bound_named():
    """ Test the sum_sim_bound() constraint function.

    Initialize a named constraint, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.constraints import sum_sim_bound
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 4)]
    # Create input vectors
    x = np.ones(1, dtype=xtype)[0]
    sx = np.ones(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    sx['sim2'][:] = 2.0
    # Create the constraint function
    const_func = sum_sim_bound(xtype, stype,
                           ['sim1', ('sim2', 0), ('sim2', 1), ('sim2', 2)],
                           type='lower')
    # Test function evaluation
    assert(np.abs(const_func(x, sx) + 8.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(const_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_bound = np.ones(1, dtype=stype)[0]
    s_bound['sim1'] *= -1.0
    s_bound['sim2'][:] *= -1.0
    s_bound['sim2'][3] = 0.0
    assert(np.abs(const_func(x, sx, der=2)['sim1'] - s_bound['sim1']) < 1.0e-8)
    assert(np.all(np.abs(const_func(x, sx, der=2)['sim2'] - s_bound['sim2'])
                  < 1.0e-8))


def test_sum_sim_bound_named_abs():
    """ Test the sum_sim_bound() constraint function with "absolute=True" option.

    Initialize a named constraint, then evaluate the function value and
    derivative with respect to x and sx.

    """

    from parmoo.constraints import sum_sim_bound
    import numpy as np

    # Create named dtypes
    xtype = [("x1", "f8"), ("x2", "f8"), ("x3", "f8"), ("x4", "f8")]
    stype = [("sim1", "f8"), ("sim2", "f8", 4)]
    # Create input vectors
    x = np.ones(1, dtype=xtype)[0]
    sx = np.ones(1, dtype=stype)[0]
    sx['sim1'] = 2.0
    sx['sim2'][:] = -2.0
    # Create the constraint function
    const_func = sum_sim_bound(xtype, stype,
                           ['sim1', ('sim2', 0), ('sim2', 1), ('sim2', 2)],
                           type='upper', absolute=True)
    # Test function evaluation
    assert(np.abs(const_func(x, sx) - 8.0) < 1.0e-8)
    # Test dx evaluation
    assert(np.all(const_func(x, sx, der=1) == np.zeros(1, dtype=xtype)[0]))
    # Test ds evaluation
    s_bound = np.zeros(1, dtype=stype)[0]
    s_bound['sim1'] = 1.0
    s_bound['sim2'][:-1] = -1.0
    assert(np.abs(const_func(x, sx, der=2)['sim1'] - s_bound['sim1']) < 1.0e-8)
    assert(np.all(np.abs(const_func(x, sx, der=2)['sim2'] - s_bound['sim2'])
                  < 1.0e-8))


if __name__ == "__main__":
    test_single_sim_bound_unnamed()
    test_single_sim_bound_named1()
    test_single_sim_bound_named2()
    test_sos_sim_bound_unnamed()
    test_sos_sim_bound_named()
    test_sum_sim_bound_unnamed()
    test_sum_sim_bound_named()
    test_sum_sim_bound_named_abs()
