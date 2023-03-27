
def test_sims_dtlz_g1():
    """ Test the g1 kernel function.

    Create an instance of g1 and check that its output is correct.

    """

    from parmoo.simulations.dtlz import g1_sim
    import numpy as np

    g1 = g1_sim(5, num_obj=3, offset=0.6)
    assert (np.abs(g1(np.ones(5) * 0.6)) < 1.0e-8)


def test_sims_dtlz_g2():
    """ Test the g2 kernel function.

    Create an instance of g2 and check that its output is correct.

    """

    from parmoo.simulations.dtlz import g2_sim
    import numpy as np

    g2 = g2_sim(5, num_obj=3, offset=0.6)
    assert (np.abs(g2(np.ones(5) * 0.6)) < 1.0e-8)


def test_sims_dtlz_g3():
    """ Test the g3 kernel function.

    Create an instance of g3 and check that its output is correct.

    """

    from parmoo.simulations.dtlz import g3_sim
    import numpy as np

    g3 = g3_sim(5, num_obj=3, offset=0.6)
    assert (np.abs(g3(np.ones(5) * 0.6)) < 1.0e-8)


def test_sims_dtlz_g4():
    """ Test the g4 kernel function.

    Create an instance of g4 and check that its output is correct.

    """

    from parmoo.simulations.dtlz import g4_sim
    import numpy as np

    g4 = g4_sim(5, num_obj=3, offset=0.6)
    assert (np.abs(g4(np.ones(5) * 0.6) - 1.0) < 1.0e-8)


def test_sims_dtlz_dtlz1():
    """ Test the dtlz1 simulation function.

    Create an instance of dtlz1_sim and check that its output is correct.

    """

    from parmoo.simulations.dtlz import dtlz1_sim
    import numpy as np

    dtlz1 = dtlz1_sim(5, num_obj=3, offset=0.6)
    assert (sum(dtlz1(np.ones(5) * 0.6)) - 0.5 < 1.0e-8)


def test_sims_dtlz_dtlz2():
    """ Test the dtlz2 simulation function.

    Create an instance of dtlz2_sim and check that its output is correct.

    """

    from parmoo.simulations.dtlz import dtlz2_sim
    import numpy as np

    dtlz2 = dtlz2_sim(5, num_obj=3, offset=0.6)
    assert (np.linalg.norm(dtlz2(np.ones(5) * 0.6)) - 1.0 < 1.0e-8)


def test_sims_dtlz_dtlz3():
    """ Test the dtlz3 simulation function.

    Create an instance of dtlz3_sim and check that its output is correct.

    """

    from parmoo.simulations.dtlz import dtlz3_sim
    import numpy as np

    dtlz3 = dtlz3_sim(5, num_obj=3, offset=0.6)
    assert (np.linalg.norm(dtlz3(np.ones(5) * 0.6)) - 1.0 < 1.0e-8)


def test_sims_dtlz_dtlz4():
    """ Test the dtlz4 simulation function.

    Create an instance of dtlz4_sim and check that its output is correct.

    """

    from parmoo.simulations.dtlz import dtlz4_sim
    import numpy as np

    dtlz4 = dtlz4_sim(5, num_obj=3, offset=0.6)
    assert (np.linalg.norm(dtlz4(np.ones(5) * 0.6)) - 1.0 < 1.0e-8)


def test_sims_dtlz_dtlz5():
    """ Test the dtlz5 simulation function.

    Create an instance of dtlz5_sim and check that its output is correct.

    """

    from parmoo.simulations.dtlz import dtlz5_sim
    import numpy as np

    dtlz5 = dtlz5_sim(5, num_obj=3, offset=0.6)
    assert (np.linalg.norm(dtlz5(np.ones(5) * 0.6)) - 1.0 < 1.0e-8)


def test_sims_dtlz_dtlz6():
    """ Test the dtlz6 simulation function.

    Create an instance of dtlz6_sim and check that its output is correct.

    """

    from parmoo.simulations.dtlz import dtlz6_sim
    import numpy as np

    dtlz6 = dtlz6_sim(5, num_obj=3, offset=0.6)
    assert (np.linalg.norm(dtlz6(np.ones(5) * 0.6)) - 1.0 < 1.0e-8)


def test_sims_dtlz_dtlz7():
    """ Test the dtlz7 simulation function.

    Create an instance of dtlz7_sim and check that its output is correct.

    """

    from parmoo.simulations.dtlz import dtlz7_sim
    import numpy as np

    dtlz7 = dtlz7_sim(5, num_obj=3, offset=0.6)
    x_in = np.ones(5) * 0.6
    x_in[:2] = 0.0
    assert (np.abs(dtlz7(x_in)[2] - 6.0) < 1.0e-8)


def test_sims_dtlz_dtlz8():
    """ Test the dtlz8 simulation function.

    Create an instance of dtlz8_sim and check that its output is correct.

    """

    from parmoo.simulations.dtlz import dtlz8_sim
    import numpy as np

    dtlz8 = dtlz8_sim(5, num_obj=3, offset=0.6)
    assert (np.all(np.abs(dtlz8(np.ones(5) * 0.6)) < 1.0e-8))


def test_sims_dtlz_dtlz9():
    """ Test the dtlz9 simulation function.

    Create an instance of dtlz9_sim and check that its output is correct.

    """

    from parmoo.simulations.dtlz import dtlz9_sim
    import numpy as np

    dtlz9 = dtlz9_sim(5, num_obj=3, offset=0.6)
    assert (np.all(np.abs(dtlz9(np.ones(5) * 0.6)) < 1.0e-8))


if __name__ == "__main__":
    test_sims_dtlz_g1()
    test_sims_dtlz_g2()
    test_sims_dtlz_g3()
    test_sims_dtlz_g4()
    test_sims_dtlz_dtlz1()
    test_sims_dtlz_dtlz2()
    test_sims_dtlz_dtlz3()
    test_sims_dtlz_dtlz4()
    test_sims_dtlz_dtlz5()
    test_sims_dtlz_dtlz6()
    test_sims_dtlz_dtlz7()
    test_sims_dtlz_dtlz8()
    test_sims_dtlz_dtlz9()
