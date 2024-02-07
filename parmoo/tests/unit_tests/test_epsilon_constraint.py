
def test_RandomConstraint():
    """ Test the RandomConstraint class in acquisitions.py.

    Use the RandomConstraint class to randomly select a target point and
    improvement direction, and check that exactly one point was targeted.

    """

    from parmoo.acquisitions import RandomConstraint
    from parmoo.util import updatePF
    import numpy as np
    import pytest

    # Define the objective function
    def obj_f(x):
        return np.asarray([np.dot(x - np.eye(3)[i],
                                  x - np.eye(3)[i]) for i in range(3)])

    # Generate a database
    x_vals = np.random.random_sample((10, 3))
    data = {'x_vals': x_vals,
            'f_vals': np.asarray([obj_f(x) for x in x_vals]),
            'c_vals': np.ones((10, 1))}

    # Initialize a good instance
    acqu = RandomConstraint(3, np.zeros(3), np.ones(3), {})
    # Try some bad targets to test error handling
    with pytest.raises(TypeError):
        acqu.setTarget(5, lambda x: np.zeros(1))
    with pytest.raises(AttributeError):
        acqu.setTarget({'x_vals': []}, lambda x: np.zeros(0))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones(1), 'f_vals': np.ones(2)},
                       lambda x: np.zeros(0))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, 1)), 'f_vals': np.ones((1, 3))},
                       lambda x: np.zeros(0))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, 3)), 'f_vals': np.ones((1, 1))},
                       lambda x: np.zeros(0))
    with pytest.raises(TypeError):
        acqu.setTarget(data, 5)
    with pytest.raises(ValueError):
        acqu.setTarget(data, lambda x, y, z: np.zeros(0))
    data['c_vals'] = np.zeros((10, 1))
    # Set a few good target
    assert (np.all(acqu.setTarget({}, lambda x: np.zeros(3)) < 1.0))
    assert (np.all(acqu.setTarget({}, lambda x: np.zeros(3)) > 0.0))
    assert (np.all(acqu.setTarget({'x_vals': np.zeros((1, 3)),
                                   'f_vals': np.zeros((1, 3)),
                                   'c_vals': np.zeros((1, 1))},
                                  lambda x: np.ones(3) * (0.01 - sum(x))) < 1.0))
    assert (acqu.setTarget(data, lambda x: np.zeros(3)) in data['x_vals'])
    # Generate a random scalarization target and check the scalarization
    acqu = RandomConstraint(3, np.zeros(3), np.ones(3), {})
    acqu.setTarget({'x_vals': None, 'f_vals': None},
                   lambda x: np.zeros(3))
    acqu.setTarget(data, lambda x: np.zeros(3))
    # Get a copy of the Pareto front for checking correctness
    pf = updatePF(data, {})
    # Check that the scalar value is either less than the sum of fi or bad
    for fi in pf['f_vals']:
        assert (acqu.scalarize(fi, np.zeros(3), np.zeros(3), np.zeros(3))
                <= np.sum(fi) or np.any(fi > acqu.f_ub))
    # Check that the scalar grad is either less than the sum of fi or bad
    for fi in pf['f_vals']:
        gi = np.random.random_sample((3, 3))
        assert (all(acqu.scalarizeGrad(fi, gi) <= np.sum(gi, axis=0)) or
                np.any(fi > acqu.f_ub))
    return


def test_EI_RandomConstraint():
    """ Test the EI_RandomConstraint class in acquisitions.py.

    Use the EI_RandomConstraint class to randomly select a target point and
    improvement direction, and check that exactly one point was targeted.

    """

    from parmoo.acquisitions import EI_RandomConstraint
    from parmoo.util import updatePF
    import numpy as np
    import pytest

    # Define the objective function
    def obj_f(x):
        return np.asarray([np.dot(x - np.eye(3)[i],
                                  x - np.eye(3)[i]) for i in range(3)])

    # Generate a database
    x_vals = np.random.random_sample((10, 3))
    data = {'x_vals': x_vals,
            'f_vals': np.asarray([obj_f(x) for x in x_vals]),
            'c_vals': np.ones((10, 1))}

    # Initialize a good instance
    acqu = EI_RandomConstraint(3, np.zeros(3), np.ones(3), {})
    # Try some bad targets to test error handling
    with pytest.raises(TypeError):
        acqu.setTarget(5, lambda x: np.zeros(1))
    with pytest.raises(AttributeError):
        acqu.setTarget({'x_vals': []}, lambda x: np.zeros(0))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones(1), 'f_vals': np.ones(2)},
                       lambda x: np.zeros(0))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, 1)), 'f_vals': np.ones((1, 3))},
                       lambda x: np.zeros(0))
    with pytest.raises(ValueError):
        acqu.setTarget({'x_vals': np.ones((1, 3)), 'f_vals': np.ones((1, 1))},
                       lambda x: np.zeros(0))
    with pytest.raises(TypeError):
        acqu.setTarget(data, 5)
    with pytest.raises(ValueError):
        acqu.setTarget(data, lambda x, y, z: np.zeros(0))
    data['c_vals'] = np.zeros((10, 1))
    # Set a few good target
    assert (np.all(acqu.setTarget({}, lambda x, sx=0: np.zeros(3)) < 1.0))
    assert (np.all(acqu.setTarget({}, lambda x, sx=0: np.zeros(3)) > 0.0))
    assert (np.all(acqu.setTarget({'x_vals': np.zeros((1, 3)),
                                   'f_vals': np.zeros((1, 3)),
                                   'c_vals': np.zeros((1, 1))},
                                  lambda x, sx=0: np.ones(3) * (0.01 - sum(x))) < 1.0))
    assert (acqu.setTarget(data, lambda x, sx=0: np.zeros(3))
            in data['x_vals'])
    # Generate a random 1D scalarization target and check the scalarization
    acqu = EI_RandomConstraint(3, np.zeros(3), np.ones(3), {})
    acqu.setTarget({'x_vals': None, 'f_vals': None},
                   lambda x, sx=0: np.zeros(3))
    acqu.setTarget(data, lambda x, sx=0: np.zeros(3))
    # Get a copy of the Pareto front for checking correctness
    pf = updatePF(data, {})
    # Check that the scalar value is either less than the sum of fi or bad
    for fi in pf['f_vals']:
        assert (acqu.scalarize(fi, np.zeros(3), np.zeros(1), np.ones(1))
                <= np.sum(fi) or np.any(fi > acqu.f_ub))
    # Generate a random 2D scalarization target and check the scalarization
    acqu = EI_RandomConstraint(3, np.zeros(3), np.ones(3), {})
    acqu.setTarget({'x_vals': None, 'f_vals': None},
                   lambda x, sx=0: np.zeros(3))
    acqu.setTarget(data, lambda x, sx=0: np.zeros(3))
    # Get a copy of the Pareto front for checking correctness
    pf = updatePF(data, {})
    # Check that the scalar value is either less than the sum of fi or bad
    for fi in pf['f_vals']:
        assert (acqu.scalarize(fi, np.zeros(3), np.zeros(2), np.ones(2))
                <= np.sum(fi) or np.any(fi > acqu.f_ub))
    # Generate a random 3D scalarization target and check the scalarization
    acqu = EI_RandomConstraint(3, np.zeros(3), np.ones(3), {})
    acqu.setTarget({'x_vals': None, 'f_vals': None},
                   lambda x, sx=0: np.zeros(3))
    acqu.setTarget(data, lambda x, sx=0: np.zeros(3))
    # Get a copy of the Pareto front for checking correctness
    pf = updatePF(data, {})
    # Check that the scalar value is either less than the sum of fi or bad
    for fi in pf['f_vals']:
        assert (acqu.scalarize(fi, np.zeros(3), np.zeros(3), np.ones(3))
                <= np.sum(fi) or np.any(fi > acqu.f_ub))
    # Check that the scalar grad is either less than the sum of fi or bad
    fi = np.random.random_sample(3)
    gi = np.random.random_sample((3, 3))
    with pytest.raises(NotImplementedError):
        acqu.scalarizeGrad(fi, gi)
    return


if __name__ == "__main__":
    test_RandomConstraint()
    test_EI_RandomConstraint()
