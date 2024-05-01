
def test_GaussRBF():
    """ Test the GaussRBF class in surrogates.py.

    Generate random x and y values for a 3D input space and 2D output space,
    then use the GaussRBF class to fit Gaussian RBFs to the data.

    Check that the GaussianRBFs interpolate the original data.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    from jax import jacrev
    import numpy as np
    import os
    from parmoo.surrogates import GaussRBF
    import pytest

    # Try some bad initializations to test error handling
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3), {'nugget': []})
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3), {'nugget': -1.0})
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3),
                 {'nugget': 0.1, 'des_tols': np.zeros(3)})
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3),
                 {'nugget': 0.1, 'des_tols': np.zeros(2)})
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3),
                 {'nugget': 0.1, 'des_tols': 0.1})
    # Create 2 identical RBFs
    rbf1 = GaussRBF(2, np.zeros(3), np.ones(3), {})
    rbf2 = GaussRBF(2, np.zeros(3), np.ones(3), {})
    # Generate some random data with 3 design variables and 2 outputs
    x_vals1 = np.random.random_sample((10, 3))
    y_vals1 = np.random.random_sample((10, 2))
    x_vals2 = np.random.random_sample((10, 3))
    y_vals2 = np.random.random_sample((10, 2))
    x_vals_full = np.concatenate((x_vals1, x_vals2), axis=0)
    y_vals_full = np.concatenate((y_vals1, y_vals2), axis=0)
    # Try to fit/update with illegal data to test error handling
    with pytest.raises(TypeError):
        rbf1.fit(0, y_vals1)
    with pytest.raises(TypeError):
        rbf1.fit(x_vals1, 0)
    with pytest.raises(ValueError):
        rbf1.fit(np.zeros((0, 3)), np.zeros((0, 2)))
    with pytest.raises(ValueError):
        rbf1.fit(np.zeros((10, 3)), np.zeros((10, 3)))
    with pytest.raises(ValueError):
        rbf1.fit(np.zeros((10, 3)), np.zeros((9, 2)))
    with pytest.raises(TypeError):
        rbf1.update(0, y_vals1)
    with pytest.raises(TypeError):
        rbf1.update(x_vals1, 0)
    with pytest.raises(ValueError):
        rbf1.update(np.zeros((10, 3)), np.zeros((10, 3)))
    with pytest.raises(ValueError):
        rbf1.update(np.zeros((10, 3)), np.zeros((9, 2)))
    # Fit both RBFs, with and without using the update method
    rbf1.fit(x_vals1, y_vals1)
    rbf1.update(x_vals2, y_vals2)
    rbf1.update(np.zeros((0, 3)), np.zeros((0, 2)))    # Update with no data
    rbf1.setTrustRegion(np.zeros(3), np.ones(3) * np.infty)
    rbf2.fit(x_vals_full, y_vals_full)
    rbf2.setTrustRegion(np.zeros(3), np.ones(3) * np.infty)
    # Try a bad improvement step to test error handling
    with pytest.raises(TypeError):
        rbf1.improve(5, False)
    with pytest.raises(ValueError):
        rbf1.improve(np.zeros(2), False)
    with pytest.raises(ValueError):
        rbf1.improve(-np.ones(3), False)
    # Check that the RBFs match on a random evaluation point
    x = np.random.random_sample((3))
    assert (all(rbf1.evaluate(x) == rbf2.evaluate(x)))
    assert (all(rbf1.stdDev(x) == rbf2.stdDev(x)))
    # Check that the RBFs interpolate, up to 8 decimal digits of precision
    for i in range(x_vals_full.shape[0]):
        assert (np.linalg.norm(rbf1.evaluate(x_vals_full[i]) - y_vals_full[i])
                < 1.0e-8)
        assert (np.linalg.norm(rbf2.evaluate(x_vals_full[i]) - y_vals_full[i])
                < 1.0e-8)
        assert (np.max(rbf1.stdDev(x_vals_full[i])) < 1.0e-4)
        assert (np.max(rbf2.stdDev(x_vals_full[i])) < 1.0e-4)
    # Check that the RBFs compute the same grad, up to 8 digits of precision
    for i in range(x_vals_full.shape[0]):
        assert (np.linalg.norm(jacrev(rbf1.evaluate)(x_vals_full[i]) -
                               jacrev(rbf2.evaluate)(x_vals_full[i]))
                < 1.0e-8)
    for i in range(x_vals_full.shape[0]):
        assert (np.linalg.norm(jacrev(rbf1.stdDev)(x_vals_full[i]) -
                               jacrev(rbf2.stdDev)(x_vals_full[i])) < 1.0e-4)
    # Check that the RBF gradient evaluates correctly on a known dataset
    x_vals3 = np.eye(3)
    x_vals3 = np.append(x_vals3, [[0.5, 0.5, 0.5]], axis=0)
    y_vals3 = np.asarray([[np.dot(xi, xi)] for xi in x_vals3])
    rbf3 = GaussRBF(1, np.zeros(3), np.ones(3), {})
    rbf3.fit(x_vals3, y_vals3)
    rbf3.setTrustRegion(np.zeros(3), np.ones(3) * np.infty)
    y_grad_vals3 = -0.03661401 * np.ones((1, 3))
    assert (np.linalg.norm(jacrev(rbf3.evaluate)(x_vals3[-1])
                           - y_grad_vals3[-1]) < 1.0e-4)
    # Check standard deviation calculations
    xi = np.random.random_sample(3)
    assert (np.all(rbf3.stdDev(xi) >= 0))
    assert (np.any(jacrev(rbf3.stdDev)(xi) != 0))
    # Check that the RBF generates feasible local improvement points
    for i in range(4):
        x_improv = rbf3.improve(np.zeros(3), False)
        assert (np.all(x_improv[0] <= np.ones(3)) and
                np.all(x_improv[0] >= np.zeros(3)))
    # Check that the RBF generates feasible global improvement points
    x_improv = rbf3.improve(x_vals3[-1], True)
    assert (np.all(x_improv[0] <= np.ones(3)) and
            np.all(x_improv[0] >= np.zeros(3)))
    # Check that RBF generates good improvements when points are bunched
    x_new = np.ones((3, 3))
    x_new = x_new * 0.5
    f_new = np.zeros((3, 1))
    for i in range(3):
        x_new[i, i] = 0.50000001
        f_new[i, 0] = np.dot(x_new[i, :], x_new[i, :])
    rbf3.update(x_new, f_new)
    x_improv = rbf3.improve(np.asarray([0.5, 0.5, 0.5]), False)
    assert (np.all(x_improv[0] <= np.ones(3)) and
            np.all(x_improv[0] >= np.zeros(3)))
    assert (np.all(np.linalg.norm(x_improv[0] - np.asarray([0.5, 0.5, 0.5]))
            > 1.0e-8))
    # Now fit redundant data points using a nugget
    x_vals1 = np.append(x_vals1, np.asarray([x_vals1[0, :]]), axis=0)
    y_vals1 = np.append(y_vals1, np.asarray([y_vals1[0, :]]), axis=0)
    rbf4 = GaussRBF(2, np.zeros(3), np.ones(3), {'nugget': 0.0001})
    rbf4.fit(x_vals1, y_vals1)
    rbf4.update(x_vals2, y_vals2)
    rbf4.setTrustRegion(np.zeros(3), np.ones(3) * np.infty)
    # Now create a really tiny design space with a large tolerance
    rbf5 = GaussRBF(1, np.zeros(1), np.ones(1),
                    {'des_tols': 0.3 * np.ones(1)})
    xdat5 = np.asarray([[0.4], [0.6]])
    ydat5 = np.asarray([[0.4], [0.6]])
    rbf5.fit(xdat5, ydat5)
    rbf5.setTrustRegion(np.zeros(1), np.ones(1) * np.infty)
    # Test that improve() is able to find points outside the design tolerance
    for i in range(5):
        x_improv = rbf5.improve(xdat5[0], False)
        assert (x_improv[0][0] < 0.1 or x_improv[0][0] > 0.9)
    for i in range(5):
        x_improv = rbf5.improve(xdat5[0], True)
        assert (x_improv[0][0] < 0.1 or x_improv[0][0] > 0.9)
    # Now fit redundant data points without a nugget
    rbf6 = GaussRBF(2, np.zeros(3), np.ones(3), {})
    rbf6.fit(x_vals1, y_vals1)
    rbf6.update(x_vals2, y_vals2)
    rbf6.setTrustRegion(np.zeros(3), np.ones(3) * np.infty)
    # Now fit datapoints in a plane
    x_vals3 = np.zeros((4, 3))
    x_vals3[1, 0] = 0.1
    x_vals3[2, 0] = 0.2
    x_vals3[3, 0] = 0.3
    y_vals3 = np.ones((4, 2))
    rbf7 = GaussRBF(2, np.zeros(3), np.ones(3), {})
    rbf7.fit(x_vals3, y_vals3)
    rbf7.update(x_vals3, y_vals3)
    rbf7.setTrustRegion(np.zeros(3), np.ones(3) * np.infty)
    # Test save and load
    rbf6.save("parmoo.surrogate")
    rbf7.load("parmoo.surrogate")
    os.remove("parmoo.surrogate")
    rbf7.setTrustRegion(np.zeros(3), np.ones(3) * np.infty)
    xx = np.random.random_sample(3)
    assert (np.all(rbf6.evaluate(xx) == rbf7.evaluate(xx)))
    # Generate a simple 1D RBF and check its stdDev is accurate
    x_vals4 = np.array([[0.0], [1.0]])
    y_vals4 = np.array([[0.0], [1.0]])
    rbf8 = GaussRBF(1, np.zeros(1), np.ones(1), {'tail_order': 0})
    rbf8.fit(x_vals4, y_vals4)
    rbf8.setTrustRegion(np.zeros(1), np.ones(1) * np.infty)
    assert (np.linalg.norm(rbf8.evaluate(np.array([0.5])) - 0.5) < 1.0e-8)
    assert (np.linalg.norm(rbf8.stdDev(np.array([0.5]))) > 1.0e-2)
    assert (np.linalg.norm(jacrev(rbf8.evaluate)(np.array([0.5]))) > 1)
    assert (np.linalg.norm(jacrev(rbf8.stdDev)(np.array([0.5]))) < 1.0e-4)
    xx = np.linspace(0, 1).reshape((50, 1))
    maxind = 0
    for i, xi in enumerate(xx):
        if np.all(rbf8.stdDev(xi) > rbf8.stdDev(xx[maxind])):
            maxind = i
        if i < 25:
            assert (np.all(jacrev(rbf8.stdDev)(xi) >= 0))
        else:
            assert (np.all(jacrev(rbf8.stdDev)(xi) <= 0))
    assert (maxind in [24, 25])
    return


def test_LocalGaussRBF():
    """ Test the LocalGaussRBF class in surrogates.py.

    Generate random x and y values for a 3D input space and 2D output space,
    then use the LocalGaussRBF class to fit Gaussian RBFs to the data.

    Check that the local Gaussian RBFs interpolate the original data.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    from jax import jacrev
    import numpy as np
    import os
    from parmoo.surrogates import GaussRBF
    import pytest

    # Try some bad initializations to test error handling
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3), {'nugget': []})
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3), {'nugget': -1.0})
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3),
                 {'nugget': 0.1, 'des_tols': np.zeros(3)})
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3),
                 {'nugget': 0.1, 'des_tols': np.zeros(2)})
    with pytest.raises(ValueError):
        GaussRBF(2, np.zeros(3), np.ones(3), {'nugget': 0.1, 'des_tols': 0.1})
    # Create 2 identical RBFs
    rbf1 = GaussRBF(2, np.zeros(3), np.ones(3), {'tail_order': 1})
    rbf2 = GaussRBF(2, np.zeros(3), np.ones(3), {'tail_order': 1})
    # Generate some random data with 3 design variables and 2 outputs
    x_vals1 = np.random.random_sample((10, 3))
    y_vals1 = np.random.random_sample((10, 2))
    x_vals2 = np.random.random_sample((10, 3))
    y_vals2 = np.random.random_sample((10, 2))
    x_vals_full = np.concatenate((x_vals1, x_vals2), axis=0)
    y_vals_full = np.concatenate((y_vals1, y_vals2), axis=0)
    # Try to fit/update with illegal data to test error handling
    with pytest.raises(TypeError):
        rbf1.fit(0, y_vals1)
    with pytest.raises(TypeError):
        rbf1.fit(x_vals1, 0)
    with pytest.raises(ValueError):
        rbf1.fit(np.zeros((0, 3)), np.zeros((0, 2)))
    with pytest.raises(ValueError):
        rbf1.fit(np.zeros((10, 3)), np.zeros((10, 3)))
    with pytest.raises(ValueError):
        rbf1.fit(np.zeros((10, 3)), np.zeros((9, 2)))
    with pytest.raises(TypeError):
        rbf1.update(0, y_vals1)
    with pytest.raises(TypeError):
        rbf1.update(x_vals1, 0)
    with pytest.raises(ValueError):
        rbf1.update(np.zeros((10, 3)), np.zeros((10, 3)))
    with pytest.raises(ValueError):
        rbf1.update(np.zeros((10, 3)), np.zeros((9, 2)))
    # Fit both RBFs, with and without using the update method
    rbf1.fit(x_vals1, y_vals1)
    rbf1.update(x_vals2, y_vals2)
    rbf1.update(np.zeros((0, 3)), np.zeros((0, 2)))    # Update with no data
    rbf1.setTrustRegion(0.5 * np.ones(3), np.ones(3) * 0.5)
    rbf2.fit(x_vals_full, y_vals_full)
    rbf2.setTrustRegion(0.5 * np.ones(3), np.ones(3) * 0.5)
    # Try to set the center with illegal values to test error handling
    with pytest.raises(TypeError):
        rbf1.setTrustRegion(5, np.ones(3))
    with pytest.raises(ValueError):
        rbf1.setTrustRegion(np.zeros(5), np.ones(3))
    with pytest.raises(ValueError):
        rbf1.setTrustRegion(-np.ones(3), np.ones(3))
    with pytest.raises(TypeError):
        rbf1.setTrustRegion(np.ones(3) * 0.5, 5)
    with pytest.raises(ValueError):
        rbf1.setTrustRegion(np.ones(3) * 0.5, np.ones(5))
    with pytest.raises(ValueError):
        rbf1.setTrustRegion(np.ones(3) * 0.5, -np.ones(3))
    # Try a bad improvement step to test error handling
    with pytest.raises(TypeError):
        rbf1.improve(5, False)
    with pytest.raises(ValueError):
        rbf1.improve(np.zeros(2), False)
    with pytest.raises(ValueError):
        rbf1.improve(-np.ones(3), False)
    # Check that the RBFs match on a random evaluation point
    x = np.random.random_sample((3))
    assert (all(rbf1.evaluate(x) == rbf2.evaluate(x)))
    assert (all(rbf1.stdDev(x) == rbf2.stdDev(x)))
    # Check that the RBFs interpolate, up to 8 decimal digits of precision
    for i in range(x_vals_full.shape[0]):
        rbf1.setTrustRegion(x_vals_full[i], np.ones(3) * 0.1)
        rbf2.setTrustRegion(x_vals_full[i], np.ones(3) * 0.1)
        assert (np.linalg.norm(rbf1.evaluate(x_vals_full[i])-y_vals_full[i])
                < 1.0e-8)
        assert (np.linalg.norm(rbf2.evaluate(x_vals_full[i])-y_vals_full[i])
                < 1.0e-8)
        assert (np.linalg.norm(rbf1.stdDev(x_vals_full[i]) < 1.0e-4))
        assert (np.linalg.norm(rbf2.stdDev(x_vals_full[i]) < 1.0e-4))
    # Check that the RBFs compute the same grad, up to 8 digits of precision
    rbf1.setTrustRegion(0.5 * np.ones(3), np.ones(3) * 0.1)
    rbf2.setTrustRegion(0.5 * np.ones(3), np.ones(3) * 0.1)
    for i in range(x_vals_full.shape[0]):
        assert (np.linalg.norm(jacrev(rbf1.evaluate)(x_vals_full[i]) -
                               jacrev(rbf2.evaluate)(x_vals_full[i])) < 1.0e-8)
        assert (np.linalg.norm(jacrev(rbf1.stdDev)(x_vals_full[i]) -
                               jacrev(rbf2.stdDev)(x_vals_full[i])) < 1.0e-4)
    # Check that the RBF gradient evaluates correctly on a known dataset
    x_vals3 = np.eye(3)
    x_vals3 = np.append(x_vals3, [[0.5, 0.5, 0.5]], axis=0)
    y_vals3 = np.asarray([[np.dot(xi, xi)] for xi in x_vals3])
    rbf3 = GaussRBF(1, np.zeros(3), np.ones(3), {'tail_order': 0})
    rbf3.fit(x_vals3, y_vals3)
    rbf3.setTrustRegion(x_vals3[-1], np.ones(3) * 0.25)
    y_grad_vals3 = -0.08798618 * np.ones((1, 3))
    assert (np.linalg.norm(jacrev(rbf3.evaluate)(x_vals3[-1])
                           - y_grad_vals3[-1]) < 1.0e-4)
    assert (np.linalg.norm(jacrev(rbf3.stdDev)(x_vals3[-1]) >= 0))
    # Check standard deviation calculations
    xi = np.random.random_sample(3)
    assert (np.all(rbf3.stdDev(xi) >= 0))
    assert (np.any(jacrev(rbf3.stdDev)(xi) != 0))
    # Check that the RBF generates feasible local improvement points
    for i in range(4):
        x_improv = rbf3.improve(np.zeros(3), False)
        assert (np.all(x_improv[0] <= np.ones(3)) and
                np.all(x_improv[0] >= np.zeros(3)))
    # Check that the RBF generates feasible global improvement points
    x_improv = rbf3.improve(x_vals3[-1], True)
    assert (np.all(x_improv[0] <= np.ones(3)) and
            np.all(x_improv[0] >= np.zeros(3)))
    # Check that RBF generates good improvements when points are bunched
    x_new = np.ones((3, 3))
    x_new = x_new * 0.5
    f_new = np.zeros((3, 1))
    for i in range(3):
        x_new[i, i] = 0.50000001
        f_new[i, 0] = np.dot(x_new[i, :], x_new[i, :])
    rbf3.update(x_new, f_new)
    rbf3.setTrustRegion(0.5 * np.ones(3), 1.0e-4)
    x_improv = rbf3.improve(np.asarray([0.5, 0.5, 0.5]), False)
    assert (np.all(x_improv[0] <= np.ones(3)) and
            np.all(x_improv[0] >= np.zeros(3)))
    assert (np.all(np.linalg.norm(x_improv[0] - np.asarray([0.5, 0.5, 0.5]))
            > 1.0e-8))
    # Now fit redundant data points using a nugget
    x_vals1 = np.append(x_vals1, np.asarray([x_vals1[0, :]]), axis=0)
    y_vals1 = np.append(y_vals1, np.asarray([y_vals1[0, :]]), axis=0)
    rbf4 = GaussRBF(2, np.zeros(3), np.ones(3), {'nugget': 0.0001})
    rbf4.fit(x_vals1, y_vals1)
    rbf4.setTrustRegion(x_vals1[0], np.ones(3) * 0.1)
    rbf4.evaluate(x_vals1[0])
    rbf4.update(x_vals2, y_vals2)
    rbf4.setTrustRegion(x_vals1[0], np.ones(3) * 0.1)
    jacrev(rbf4.evaluate)(x_vals1[0])
    # Now create a really tiny design space with a large tolerance
    rbf5 = GaussRBF(1, np.zeros(1), np.ones(1), {'des_tols': 0.3 * np.ones(1)})
    xdat5 = np.asarray([[0.4], [0.6]])
    ydat5 = np.asarray([[0.4], [0.6]])
    rbf5.fit(xdat5, ydat5)
    rbf5.setTrustRegion(np.ones(1) * 0.5, np.ones(1) * 1.0e-1)
    # Test that improve() is able to find points outside the design tolerance
    for i in range(5):
        x_improv = rbf5.improve(xdat5[0], False)
        assert (x_improv[0][0] < 0.1 or x_improv[0][0] > 0.9)
    for i in range(5):
        x_improv = rbf5.improve(xdat5[0], True)
        assert (x_improv[0][0] < 0.1 or x_improv[0][0] > 0.9)
    # Now fit redundant data points to test adaptive nugget
    rbf6 = GaussRBF(2, np.zeros(3), np.ones(3), {})
    rbf6.fit(x_vals1, y_vals1)
    rbf6.update(x_vals2, y_vals2)
    rbf6.setTrustRegion(x_vals1[0], np.ones(3) * 0.1)
    # Now fit datapoints in a plane
    x_vals3 = np.zeros((4, 3))
    x_vals3[1, 0] = 0.1
    x_vals3[2, 0] = 0.2
    x_vals3[3, 0] = 0.3
    y_vals3 = np.ones((4, 2))
    rbf7 = GaussRBF(2, np.zeros(3), np.ones(3), {})
    rbf7.fit(x_vals3, y_vals3)
    rbf7.update(x_vals3, y_vals3)
    # Test save and load
    rbf6.save("parmoo.surrogate")
    rbf7.load("parmoo.surrogate")
    os.remove("parmoo.surrogate")
    rbf7.setTrustRegion(np.zeros(3), np.ones(3) * np.infty)
    xx = np.random.random_sample(3)
    assert (np.all(rbf6.evaluate(xx) == rbf7.evaluate(xx)))
    # Generate a simple 1D RBF and check its stdDev is accurate
    x_vals4 = np.array([[0.0], [1.0]])
    y_vals4 = np.array([[0.0], [1.0]])
    rbf8 = GaussRBF(1, np.zeros(1), np.ones(1), {'tail_order': 0})
    rbf8.fit(x_vals4, y_vals4)
    rbf8.setTrustRegion(np.array([0.5]), np.ones(1) * 0.25)
    assert (np.linalg.norm(rbf8.evaluate(np.array([0.5])) - 0.5) < 1.0e-8)
    assert (np.linalg.norm(rbf8.stdDev(np.array([0.5]))) > 5.0e-3)
    assert (np.linalg.norm(jacrev(rbf8.evaluate)(np.array([0.5]))) > 1.0)
    assert (np.linalg.norm(jacrev(rbf8.stdDev)(np.array([0.5]))) < 1.0e-4)
    xx = np.linspace(0, 1).reshape((50, 1))
    maxind = 0
    for i, xi in enumerate(xx):
        if np.all(rbf8.stdDev(xi) > rbf8.stdDev(xx[maxind])):
            maxind = i
        if i < 25:
            assert (np.all(jacrev(rbf8.stdDev)(xi) >= 0))
        else:
            assert (np.all(jacrev(rbf8.stdDev)(xi) <= 0))
    assert (maxind in [24, 25])
    return


if __name__ == "__main__":
    test_GaussRBF()
    test_LocalGaussRBF()
