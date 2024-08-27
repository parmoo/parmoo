
def test_sims_LinearSim():
    """ Test the linear sim function.

    Create an instance of sims.LinearSim and check that its output is correct.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo.simulations.simple import LinearSim

    dtype = np.dtype([(f"x{i+1}", "f8") for i in range(5)])
    ls = LinearSim(dtype, num_obj=3, num_r=5)
    x1 = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "x5": 0}
    x2 = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3, "x4": 1 / 3, "x5": 1 / 3}
    x3 = {"x1": 2 / 3, "x2": 2 / 3, "x3": 2 / 3, "x4": 2 / 3, "x5": 2 / 3}
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

    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from parmoo.simulations.simple import QuadraticSim

    dtype = np.dtype([(f"x{i+1}", "f8") for i in range(5)])
    qs = QuadraticSim(dtype, num_obj=3, num_r=5)
    x1 = {"x1": 0, "x2": 0, "x3": 0, "x4": 0, "x5": 0}
    x2 = {"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3, "x4": 1 / 3, "x5": 1 / 3}
    x3 = {"x1": 2 / 3, "x2": 2 / 3, "x3": 2 / 3, "x4": 2 / 3, "x5": 2 / 3}
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
