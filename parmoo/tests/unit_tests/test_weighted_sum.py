
def test_UniformWeights():
    """ Test the UniformWeights class in acquisitions.py.

    Use the UniformWeights class to randomly sample 3 sets of convex weights
    from the unit simplex, and make sure that they are all convex.

    """

    from jax import jacrev
    import numpy as np
    from parmoo.acquisitions import UniformWeights
    import pytest

    # Initilaize a good acquisition for future testing
    acqu = UniformWeights(3, np.zeros(4), np.ones(4), {})
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
    assert (abs(sum(acqu.weights) - 1.0) < 0.00000001)
    assert (np.all(x0[:] <= acqu.ub) and np.all(x0[:] >= acqu.lb))
    # Generate 3 random weight vector
    acqu1 = UniformWeights(3, np.zeros(4), np.ones(4), {})
    acqu1.setTarget({}, lambda x: np.zeros(3))
    acqu2 = UniformWeights(3, np.zeros(4), np.ones(4), {})
    acqu2.setTarget({}, lambda x: np.zeros(3))
    acqu3 = UniformWeights(3, np.zeros(4), np.ones(4), {})
    acqu3.setTarget({'x_vals': None, 'f_vals': None},
                    lambda x: np.zeros(3))
    # Check that the weights are all greater than 0
    assert (all(acqu1.weights[:] >= 0.0))
    assert (all(acqu2.weights[:] >= 0.0))
    assert (all(acqu3.weights[:] >= 0.0))
    # Use the scalarization function to check that the weights sum to 1
    assert (np.abs(acqu1.scalarize(np.eye(3)[0], np.ones(2),
                                   np.ones(2), np.ones(2))
                   + acqu1.scalarize(np.eye(3)[1], np.ones(2),
                                     np.ones(2), np.ones(2))
                   + acqu1.scalarize(np.eye(3)[2], np.ones(2),
                                     np.ones(2), np.ones(2)) - 1.0)
            < 0.00000001)
    assert (np.abs(acqu2.scalarize(np.eye(3)[0], np.ones(2),
                                   np.ones(2), np.ones(2))
                   + acqu2.scalarize(np.eye(3)[1], np.ones(2),
                                     np.ones(2), np.ones(2))
                   + acqu2.scalarize(np.eye(3)[2], np.ones(2),
                                     np.ones(2), np.ones(2)) - 1.0)
            < 0.00000001)
    assert (np.abs(acqu3.scalarize(np.eye(3)[0], np.ones(2),
                                   np.ones(2), np.ones(2))
                   + acqu3.scalarize(np.eye(3)[1], np.ones(2),
                                     np.ones(2), np.ones(2))
                   + acqu3.scalarize(np.eye(3)[2], np.ones(2),
                                     np.ones(2), np.ones(2)) - 1.0)
            < 0.00000001)
    # Check the gradient scalarization appears to work correctly
    df1 = jacrev(acqu1.scalarize)(np.eye(3)[0], np.ones(4),
                                  np.ones(1), np.ones(1))
    df2 = jacrev(acqu2.scalarize)(np.eye(3)[0], np.ones(4),
                                  np.ones(1), np.ones(1))
    df3 = jacrev(acqu3.scalarize)(np.eye(3)[0], np.ones(4),
                                  np.ones(1), np.ones(1))
    assert (np.linalg.norm(df1 - acqu1.weights) < 1.0e-4)
    assert (np.linalg.norm(df2 - acqu2.weights) < 1.0e-4)
    assert (np.linalg.norm(df3 - acqu3.weights) < 1.0e-4)


def test_FixedWeights():
    """ Test the FixedWeights class in acquisitions.py.

    Use the FixedWeights class to try 2 sets of fixed convex weights
    from the unit simplex, and make sure that they are all convex.

    """

    from jax import jacrev
    import numpy as np
    from parmoo.acquisitions import FixedWeights
    import pytest

    # Try some bad initializations to test error handling
    with pytest.raises(TypeError):
        FixedWeights(3, np.zeros(4), np.ones(4), {'weights': 5.0})
    with pytest.raises(ValueError):
        FixedWeights(3, np.zeros(4), np.ones(4), {'weights': np.ones(2)})
    # Initilaize a good acquisition for future testing
    acqu = FixedWeights(3, np.zeros(4), np.ones(4), {'weights': np.ones((3))})
    assert (np.all(acqu.lb[:] == 0.0) and np.all(acqu.ub[:] == 1.0))
    acqu = FixedWeights(3, np.zeros(4), np.ones(4), {})
    assert (np.all(acqu.lb[:] == 0.0) and np.all(acqu.ub[:] == 1.0))
    assert (np.sum(acqu.weights[:]) - 1.0 < 0.00000001)
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
    # Use the scalarization function to check that the weights sum to 1
    assert (np.abs(acqu.scalarize(np.eye(3)[0], np.ones(2),
                                  np.ones(2), np.ones(2))
                   + acqu.scalarize(np.eye(3)[1], np.ones(2),
                                    np.ones(2), np.ones(2))
                   + acqu.scalarize(np.eye(3)[2], np.ones(2),
                                    np.ones(2), np.ones(2)) - 1.0)
            < 0.00000001)
    # Check the scalarization gradient
    da = jacrev(acqu.scalarize)(np.eye(3)[0], np.zeros(4),
                                np.zeros(1), np.zeros(1))
    assert (np.abs(np.sum(da) - 1.0) < 1.0e-4)


if __name__ == "__main__":
    test_UniformWeights()
    test_FixedWeights()
