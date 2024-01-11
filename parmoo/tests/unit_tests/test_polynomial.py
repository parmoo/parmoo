
def test_Linear():
    """ Test the Linear model class in surrogates.py.

    Generate random x and y values for a 3D input space and 2D output space,
    then use the Linear class to fit linear models to the data.

    Check that the linear models interpolate the original data.

    """

    from parmoo.surrogates import Linear
    import numpy as np
    import pytest
    import os

    # Try some bad initializations to test error handling
    with pytest.raises(ValueError):
        Linear(2, np.zeros(3), np.ones(3), {'des_tols': np.zeros(3)})
    with pytest.raises(ValueError):
        Linear(2, np.zeros(3), np.ones(3), {'des_tols': np.zeros(2)})
    with pytest.raises(ValueError):
        Linear(2, np.zeros(3), np.ones(3), {'des_tols': 0.1})
    # Create 2 identical Linear models
    lsm1 = Linear(2, np.zeros(3), np.ones(3), {})
    lsm2 = Linear(2, np.zeros(3), np.ones(3), {})
    # Generate some random data with 3 design variables and 2 outputs
    x_vals1 = np.random.random_sample((3, 3))
    y_vals1 = np.random.random_sample((3, 2))
    x_vals2 = np.random.random_sample((3, 3))
    y_vals2 = np.random.random_sample((3, 2))
    x_vals_full = np.concatenate((x_vals1, x_vals2), axis=0)
    y_vals_full = np.concatenate((y_vals1, y_vals2), axis=0)
    # Try to fit/update with illegal data to test error handling
    with pytest.raises(TypeError):
        lsm1.fit(0, y_vals1)
    with pytest.raises(TypeError):
        lsm1.fit(x_vals1, 0)
    with pytest.raises(ValueError):
        lsm1.fit(np.zeros((0, 3)), np.zeros((0, 2)))
    with pytest.raises(ValueError):
        lsm1.fit(np.zeros((10, 3)), np.zeros((10, 3)))
    with pytest.raises(ValueError):
        lsm1.fit(np.zeros((10, 3)), np.zeros((9, 2)))
    with pytest.raises(TypeError):
        lsm1.update(0, y_vals1)
    with pytest.raises(TypeError):
        lsm1.update(x_vals1, 0)
    with pytest.raises(ValueError):
        lsm1.update(np.zeros((10, 3)), np.zeros((10, 3)))
    with pytest.raises(ValueError):
        lsm1.update(np.zeros((10, 3)), np.zeros((9, 2)))
    # Fit both models, with and without using the update method
    lsm1.fit(x_vals1, y_vals1)
    lsm1.update(x_vals2, y_vals2)
    lsm1.update(np.zeros((0, 3)), np.zeros((0, 2)))    # Update with no data
    lsm1.setTrustRegion(0.5 * np.ones(3), np.ones(3) * 0.5)
    lsm2.fit(x_vals_full, y_vals_full)
    lsm2.setTrustRegion(0.5 * np.ones(3), np.ones(3) * 0.5)
    # Try to set the center with illegal values to test error handling
    with pytest.raises(TypeError):
        lsm1.setTrustRegion(5, np.ones(3))
    with pytest.raises(ValueError):
        lsm1.setTrustRegion(np.zeros(5), np.ones(3))
    with pytest.raises(ValueError):
        lsm1.setTrustRegion(-np.ones(3), np.ones(3))
    # Try to set the radius with illegal values to test error handling
    with pytest.raises(TypeError):
        lsm1.setTrustRegion(np.ones(3), 5)
    with pytest.raises(ValueError):
        lsm1.setTrustRegion(np.ones(3), np.ones(5))
    with pytest.raises(ValueError):
        lsm1.setTrustRegion(np.ones(3), -np.ones(3))
    # Try a bad improvement call to test error handling
    with pytest.raises(TypeError):
        lsm1.improve(5, False)
    with pytest.raises(ValueError):
        lsm1.improve(np.zeros(2), False)
    with pytest.raises(ValueError):
        lsm1.improve(-np.ones(3), False)
    # Check that the models match on a random evaluation point
    x = np.random.random_sample((3))
    assert (all(lsm1.evaluate(x) == lsm2.evaluate(x)))
    # Check that the models interpolate, up to 8 decimal digits of precision
    for xi, yi in zip(x_vals_full, y_vals_full):
        lsm1.setTrustRegion(xi, np.ones(3) * 0.1)
        lsm2.setTrustRegion(xi, np.ones(3) * 0.1)
        assert (np.linalg.norm(lsm1.evaluate(xi) - yi) < 1.0e-8)
        assert (np.linalg.norm(lsm2.evaluate(xi) - yi) < 1.0e-8)
    # Check that the models compute the same grad, up to 8 digits of precision
    lsm1.setTrustRegion(0.5 * np.ones(3), np.ones(3) * 0.5)
    lsm2.setTrustRegion(0.5 * np.ones(3), np.ones(3) * 0.5)
    for i in range(x_vals_full.shape[0]):
        assert (np.linalg.norm(lsm1.gradient(x_vals_full[i]) -
                               lsm2.gradient(x_vals_full[i])) < 0.00000001)
    # Check that the model generates feasible local improvement points
    x_vals3 = np.eye(3)
    x_vals3 = np.append(x_vals3, [[0.5, 0.5, 0.5]], axis=0)
    y_vals3 = np.asarray([[np.dot(xi, xi)] for xi in x_vals3])
    lsm3 = Linear(1, np.zeros(3), np.ones(3), {'tail_order': 0})
    lsm3.fit(x_vals3, y_vals3)
    lsm3.setTrustRegion(x_vals3[-1], np.ones(3) * 0.5)
    for i in range(4):
        x_improv = lsm3.improve(np.zeros(3), False)
        assert (np.all(x_improv[0] <= np.ones(3)) and
                np.all(x_improv[0] >= np.zeros(3)))
    # Check that the model generates feasible global improvement points
    x_improv = lsm3.improve(x_vals3[-1], True)
    assert (np.all(x_improv[0] <= np.ones(3)) and
            np.all(x_improv[0] >= np.zeros(3)))
    # Check that model generates good improvements when points are bunched
    x_new = np.ones((3, 3))
    x_new = x_new * 0.5
    f_new = np.zeros((3, 1))
    for i in range(3):
        x_new[i, i] = 0.50000001
        f_new[i, 0] = np.dot(x_new[i, :], x_new[i, :])
    lsm3.update(x_new, f_new)
    lsm3.setTrustRegion(0.5 * np.ones(3), np.ones(3) * 1.0e-4)
    x_improv = lsm3.improve(np.asarray([0.5, 0.5, 0.5]), False)
    assert (np.all(x_improv[0] <= np.ones(3)) and
            np.all(x_improv[0] >= np.zeros(3)))
    assert (np.all(np.linalg.norm(x_improv[0] - np.asarray([0.5, 0.5, 0.5]))
            > 0.00000001))
    # Now create a really tiny design space with a large tolerance
    lsm4 = Linear(1, np.zeros(1), np.ones(1),
                         {'des_tols': 0.3 * np.ones(1)})
    xdat4 = np.asarray([[0.4], [0.6]])
    ydat4 = np.asarray([[0.4], [0.6]])
    lsm4.fit(xdat4, ydat4)
    # Test that improve() is able to find points outside the design tolerance
    for i in range(5):
        x_improv = lsm4.improve(xdat4[0], False)
        assert (x_improv[0][0] < 0.1 or x_improv[0][0] > 0.9)
    for i in range(5):
        x_improv = lsm4.improve(xdat4[0], True)
        assert (x_improv[0][0] < 0.1 or x_improv[0][0] > 0.9)
    # Now fit datapoints in a plane
    x_vals3 = np.zeros((4, 3))
    x_vals3[1, 0] = 0.1
    x_vals3[2, 0] = 0.2
    x_vals3[3, 0] = 0.3
    y_vals3 = np.ones((4, 2))
    lsm5 = Linear(2, np.zeros(3), np.ones(3), {})
    lsm5.fit(x_vals3, y_vals3)
    lsm5.update(x_vals3, y_vals3)
    lsm5.setTrustRegion(x_vals3[-1], np.ones(3) * 0.2)
    # Test save and load
    lsm5.save("parmoo.surrogate")
    lsm4.load("parmoo.surrogate")
    xx = np.random.random_sample(3)
    assert (np.all(lsm4.evaluate(xx) == lsm5.evaluate(xx)))
    os.remove("parmoo.surrogate")
    # Generate a simple 1D model and check it is accurate
    x_vals4 = np.array([[0], [1]])
    y_vals4 = np.array([[1], [1]])
    lsm6 = Linear(1, np.zeros(1), np.ones(1), {})
    lsm6.fit(x_vals4, y_vals4)
    lsm6.setTrustRegion(np.array([0.5]), np.ones(1) * 0.1)
    assert (np.linalg.norm(lsm6.evaluate(np.array([0.5])) - 1.0) < 1.0e-8)
    assert (np.linalg.norm(lsm6.gradient(np.array([0.5]))) < 1.0e-8)
    return


if __name__ == "__main__":
    test_Linear()
