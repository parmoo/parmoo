
def test_sims_LinearSim():
    """ Test the linear sim function.

    Create an instance of sims.LinearSim and check that its output is correct.

    """

    from parmoo.simulations.simple import LinearSim
    import numpy as np

    ls = LinearSim(5, num_obj=3, num_r=5)
    x1 = np.zeros(5)
    x2 = np.zeros(5)
    x2[:] = 1 / 3
    x3 = np.zeros(5)
    x3[:] = 2 / 3
    assert (np.all(np.abs(ls(x1)[:5]) < 1.0e-8))
    assert (np.all(np.abs(ls(x2)[:5] - 1 / 3) < 1.0e-8))
    assert (np.all(np.abs(ls(x3)[:5] - 2 / 3) < 1.0e-8))
    assert (np.all(np.abs(ls(x1)[5:10] + 1 / 3) < 1.0e-8))
    assert (np.all(np.abs(ls(x2)[5:10]) < 1.0e-8))
    assert (np.all(np.abs(ls(x3)[5:10] - 1 / 3) < 1.0e-8))
    assert (np.all(np.abs(ls(x1)[10:] + 2 / 3) < 1.0e-8))
    assert (np.all(np.abs(ls(x2)[10:] + 1 / 3) < 1.0e-8))
    assert (np.all(np.abs(ls(x3)[10:]) < 1.0e-8))


def test_sims_QuadraticSim():
    """ Test the quadratic sim function.

    Create an instance of sims.QuadraticSim and check that its output is
    correct.

    """

    from parmoo.simulations.simple import QuadraticSim
    import numpy as np

    qs = QuadraticSim(5, num_obj=3, num_r=5)
    x1 = np.zeros(5)
    x2 = np.zeros(5)
    x2[:] = 1 / 3
    x3 = np.zeros(5)
    x3[:] = 2 / 3
    assert (np.abs(qs(x1)[0]) < 1.0e-8)
    assert (np.abs(qs(x2)[0]) > 1.0e-1)
    assert (np.abs(qs(x3)[0]) > 1.0e-1)
    assert (np.abs(qs(x1)[1]) > 1.0e-1)
    assert (np.abs(qs(x2)[1]) < 1.0e-8)
    assert (np.abs(qs(x3)[1]) > 1.0e-1)
    assert (np.abs(qs(x1)[2]) > 1.0e-1)
    assert (np.abs(qs(x2)[2]) > 1.0e-1)
    assert (np.abs(qs(x3)[2]) < 1.0e-8)


if __name__ == "__main__":
    test_sims_LinearSim()
    test_sims_QuadraticSim()
