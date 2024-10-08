
def test_UniformAugChebyshev():
    """ Test the UniformAugChebyshev class in acquisitions.py.

    Use the UniformAugChebyshev class to randomly sample 3 sets of convex
    weights from the unit simplex, and make sure that they are all convex.

    """

    from jax import jacrev
    import numpy as np
    from parmoo.acquisitions import UniformAugChebyshev
    import pytest

    # Initialize a good acquisition for future testing
    acqu = UniformAugChebyshev(3, np.zeros(4), np.ones(4), {})
    assert (np.all(acqu.lb[:] == 0.0) and np.all(acqu.ub[:] == 1.0))
    # Set some bad targets to check error handling
    with pytest.raises(TypeError):
        acqu.setTarget(5, lambda x: np.zeros(3))
    with pytest.raises(AttributeError):
        acqu.setTarget({'x_vals': []}, lambda x: np.zeros(3))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones(1), 'f_vals': np.ones(2)},
                       lambda x: np.zeros(3))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, 1)), 'f_vals': np.ones((1, 3))},
                       lambda x: np.zeros(3))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, 4)), 'f_vals': np.ones((1, 1))},
                       lambda x: np.zeros(3))
    with pytest.raises(TypeError):
        acqu.setTarget({}, 4)
    with pytest.raises(ValueError):
        acqu.setTarget({}, lambda x, y, z: np.zeros(3))
    # Set a good target for future usage
    x0 = acqu.setTarget({}, lambda x: np.zeros(3))
    assert (abs(sum(acqu.weights) - 1.0) < 1.0e-8)
    assert (np.all(x0[:] <= acqu.ub) and np.all(x0[:] >= acqu.lb))
    # Generate 3 random weight vector
    acqu1 = UniformAugChebyshev(3, np.zeros(4), np.ones(4), {})
    acqu1.setTarget({}, lambda x: np.zeros(3))
    acqu2 = UniformAugChebyshev(3, np.zeros(4), np.ones(4), {})
    acqu2.setTarget({}, lambda x: np.zeros(3))
    acqu3 = UniformAugChebyshev(3, np.zeros(4), np.ones(4), {})
    acqu3.setTarget({'x_vals': None, 'f_vals': None},
                    lambda x: np.zeros(3))
    # Check that the weights are all greater than 0 and sum to 1
    assert (all(acqu1.weights[:] >= 0.0))
    assert (all(acqu2.weights[:] >= 0.0))
    assert (all(acqu3.weights[:] >= 0.0))
    assert (abs(sum(acqu1.weights[:]) - 1.0) < 1.0e-8)
    assert (abs(sum(acqu2.weights[:]) - 1.0) < 1.0e-8)
    assert (abs(sum(acqu3.weights[:]) - 1.0) < 1.0e-8)
    # Check the scalarization and manifold checker appears to work correctly
    f_vals = np.random.sample(3)
    maxind = np.argmax(acqu1.weights * f_vals)
    assert (acqu1.getManifold(f_vals)[maxind] == 1)
    assert (abs(acqu1.scalarize(f_vals, np.ones(2), np.ones(2), np.ones(2)) -
                acqu1.weights[maxind] * f_vals[maxind]) < 3.0e-2)
    f_vals = np.random.sample(3)
    maxind = np.argmax(acqu2.weights * f_vals)
    assert (acqu2.getManifold(f_vals)[maxind] == 1)
    assert (abs(acqu2.scalarize(f_vals, np.ones(2), np.ones(2), np.ones(2)) -
                acqu2.weights[maxind] * f_vals[maxind]) < 3.0e-2)
    f_vals = np.random.sample(3)
    maxind = np.argmax(acqu3.weights * f_vals)
    assert (acqu3.getManifold(f_vals)[maxind] == 1)
    assert (abs(acqu3.scalarize(f_vals, np.ones(2), np.ones(2), np.ones(2)) -
                acqu3.weights[maxind] * f_vals[maxind]) < 3.0e-2)
    # Check the gradient scalarization appears to work correctly
    df1 = jacrev(acqu1.scalarize)(np.ones(3), np.ones(4),
                                  np.ones(1), np.ones(1))
    df2 = jacrev(acqu2.scalarize)(np.ones(3), np.ones(4),
                                  np.ones(1), np.ones(1))
    df3 = jacrev(acqu3.scalarize)(np.ones(3), np.ones(4),
                                  np.ones(1), np.ones(1))
    assert (np.abs(np.max(df1) - np.max(acqu1.weights) - 1.0e-2) < 1.0e-2)
    assert (np.abs(np.max(df2) - np.max(acqu2.weights) - 1.0e-2) < 1.0e-2)
    assert (np.abs(np.max(df3) - np.max(acqu3.weights) - 1.0e-2) < 1.0e-2)


def test_FixedAugChebyshev():
    """ Test the FixedAugChebyshev class in acquisitions.py.

    Use the FixedAugChebyshev class to try 2 sets of fixed convex weights
    from the unit simplex, and make sure that they are all convex.

    """

    from jax import jacrev
    import numpy as np
    from parmoo.acquisitions import FixedAugChebyshev
    import pytest

    # Try some bad initializations to test error handling
    with pytest.raises(TypeError):
        FixedAugChebyshev(3, np.zeros(4), np.ones(4), {'weights': 5.0})
    with pytest.raises(ValueError):
        FixedAugChebyshev(3, np.zeros(4), np.ones(4), {'weights': np.ones(2)})
    # Initialize a good acquisition for future testing
    acqu = FixedAugChebyshev(3, np.zeros(4), np.ones(4),
                             {'weights': np.ones((3))})
    assert (np.all(acqu.lb[:] == 0.0) and np.all(acqu.ub[:] == 1.0))
    acqu = FixedAugChebyshev(3, np.zeros(4), np.ones(4), {})
    assert (np.all(acqu.lb[:] == 0.0) and np.all(acqu.ub[:] == 1.0))
    assert (np.sum(acqu.weights[:]) - 1.0 < 1.0e-8)
    # Set some bad targets to check error handling
    with pytest.raises(TypeError):
        acqu.setTarget(5, lambda x: np.zeros(3))
    with pytest.raises(AttributeError):
        acqu.setTarget({'x_vals': []}, lambda x: np.zeros(3))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones(1), 'f_vals': np.ones(2)},
                       lambda x: np.zeros(3))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, 1)), 'f_vals': np.ones((1, 3))},
                       lambda x: np.zeros(3))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, 4)), 'f_vals': np.ones((1, 1))},
                       lambda x: np.zeros(3))
    with pytest.raises(TypeError):
        acqu.setTarget({}, 4)
    with pytest.raises(ValueError):
        acqu.setTarget({}, lambda x, y, z: np.zeros(3))
    # Set some good targets
    x0 = acqu.setTarget({'x_vals': None, 'f_vals': None},
                        lambda x: np.zeros(3))
    assert (np.all(x0[:] <= acqu.ub) and np.all(x0[:] >= acqu.lb))
    x0 = acqu.setTarget({'x_vals': np.zeros((1, 4)),
                         'f_vals': np.zeros((1, 3)),
                         'c_vals': np.zeros((1, 1))},
                        lambda x: np.zeros(3))
    assert (np.all(x0[:] <= acqu.ub) and np.all(x0[:] >= acqu.lb))
    x0 = acqu.setTarget({}, lambda x: np.zeros(3))
    assert (np.all(x0[:] <= acqu.ub) and np.all(x0[:] >= acqu.lb))
    # Use the scalarization function to check the weights sum to 1
    assert (np.abs(acqu.scalarize(np.eye(3)[0], np.ones(2),
                                  np.ones(2), np.ones(2))
                   + acqu.scalarize(np.eye(3)[1], np.ones(2),
                                    np.ones(2), np.ones(2))
                   + acqu.scalarize(np.eye(3)[2], np.ones(2),
                                    np.ones(2), np.ones(2)) - 1.0)
            < 1.1e-1)
    # Check the scalarization gradient
    da = jacrev(acqu.scalarize)(np.eye(3)[0], np.zeros(4),
                                np.zeros(1), np.zeros(1))
    assert (np.abs(np.sum(da) - acqu.weights[0]) < 1.1e-1)


if __name__ == "__main__":
    test_UniformAugChebyshev()
    test_FixedAugChebyshev()
