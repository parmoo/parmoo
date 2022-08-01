
def test_LBFGSB():
    """ Test the LBFGS class in optimizers.py.

    Perform a test of the LBFGSB class by minimizing the three variable,
    biobjective function

    $$
    F(x) = (-x_1 + x_2 + x_3, x_1 - x_2 + x_3)
    $$

    s.t. $x in [0, 1]^n$.

    Use the weights [1, 0], [0, 1], and [0.5, 0.5].

    Assert that the solutions are all correct. I.e., (1, 0, 0) is
    the minizer of $F^T [1, 0]$; (0, 1, 0) is the minimizer of $F^T [0, 1]$;
    and the minimizer of $F^T [0.5, 0.5]$ satisfies x_3 = 0.

    """

    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LBFGSB
    import numpy as np
    import pytest

    # Initialize the problem dimensions
    n = 3
    o = 2
    lb = np.zeros(n)
    ub = np.ones(n)

    # Create the biobjective function and its augmented Lagrangian
    def f(z):
        return np.asarray([-z[0] + z[1] + z[2], z[0] - z[1] + z[2]])

    def L(z):
        res = np.asarray([-z[0] + z[1] + z[2], z[0] - z[1] + z[2]])
        if z[2] < 0.1:
            res[:] = res[:] + 2.0 * (0.1 - z[2])
        if z[2] > 0.6:
            res[:] = res[:] + 2.0 * (z[2] - 0.6)
        return res

    def g(z):
        res = np.asarray([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])
        if z[2] < 0.1:
            res[:, 2] = res[:, 2] - 2.0
        if z[2] > 0.6:
            res[:, 2] = res[:, 2] + 2.0
        return res

    # Create 2 acquisition functions targeting 2 "pure" solutions
    acqu1 = UniformWeights(o, lb, ub, {})
    acqu1.setTarget({}, lambda x: np.zeros(2), {})
    acqu1.weights[:] = 0.0
    acqu1.weights[0] = 1.0
    acqu2 = UniformWeights(o, lb, ub, {})
    acqu2.setTarget({}, lambda x: np.zeros(2), {})
    acqu2.weights[:] = 0.0
    acqu2.weights[1] = 1.0
    # Create a third acquisition function targeting a random tradeoff solution
    acqu3 = UniformWeights(o, lb, ub, {})
    acqu3.setTarget({}, lambda x: np.zeros(2), {})
    acqu3.weights[:] = 0.5
    # Try some bad initializations to test error handling
    with pytest.raises(ValueError):
        LBFGSB(o, lb, ub, {'opt_budget': 2.0})
    with pytest.raises(ValueError):
        LBFGSB(o, lb, ub, {'opt_budget': 0})
    # Initialize the problem correctly, with and without an optional budget
    LBFGSB(o, lb, ub, {'opt_budget': 100})
    opt = LBFGSB(o, lb, ub, {})
    # Try to add some bad objectives, constraints, and acquisitions
    with pytest.raises(ValueError):
        opt.setObjective(5)
    with pytest.raises(ValueError):
        opt.setObjective(lambda z1, z2: np.zeros(1))
    with pytest.raises(ValueError):
        opt.setConstraints(5)
    with pytest.raises(ValueError):
        opt.setConstraints(lambda z1, z2: np.zeros(1))
    with pytest.raises(ValueError):
        opt.setLagrangian(5, lambda z: np.zeros(1))
    with pytest.raises(ValueError):
        opt.setLagrangian(lambda z1, z2: np.zeros(1), lambda z: np.zeros(1))
    with pytest.raises(ValueError):
        opt.setLagrangian(lambda z: np.zeros(1), 5)
    with pytest.raises(ValueError):
        opt.setLagrangian(lambda z: np.zeros(1), lambda z1, z2: np.zeros(1))
    with pytest.raises(ValueError):
        opt.addAcquisition(5)
    # Add the correct objective and constraints
    opt.setObjective(f)
    opt.setConstraints(lambda z: np.asarray([0.1 - z[2], z[2] - 0.6]))
    opt.setLagrangian(L, g)
    opt.addAcquisition(acqu1, acqu2, acqu3)
    opt.setReset(lambda x: 100.0)
    # Try to solve with invalid inputs to test error handling
    with pytest.raises(ValueError):
        opt.solve(5)
    with pytest.raises(ValueError):
        opt.solve(np.zeros((3, n-1)))
    with pytest.raises(ValueError):
        opt.solve(np.zeros((4, n)))
    with pytest.raises(ValueError):
        opt.solve(-np.ones((3, n)))
    # Solve the surrogate problem with LBFGSB, starting from the centroid
    x = np.zeros((3, n))
    x[:] = 0.5
    (x1, x2, x3) = opt.solve(x)
    # Define the solution
    x1_soln = np.eye(n)[0]
    x1_soln[n-1] = 0.1
    x2_soln = np.eye(n)[1]
    x2_soln[n-1] = 0.1
    # eps is the tolerance for rejecting a solution as incorrect
    eps = 0.01
    # Check that the computed solutions are within eps of the truth
    assert(np.linalg.norm(x1 - x1_soln) < eps)
    assert(np.linalg.norm(x2 - x2_soln) < eps)
    assert(np.abs(x3[n-1] - 0.1) < eps)
    return


def test_TR_LBFGSB():
    """ Test the TR_LBFGSB class in optimizers.py.

    Perform a test of the TR_LBFGSB class by minimizing the three variable,
    biobjective function

    $$
    F(x) = (-x_1 + x_2 + x_3, x_1 - x_2 + x_3)
    $$

    s.t. $x in [0, 1]^n$.

    Use the weights [1, 0], [0, 1], and [0.5, 0.5].

    Assert that the solutions are all correct. I.e., (1, 0, 0) is
    the minizer of $F^T [1, 0]$; (0, 1, 0) is the minimizer of $F^T [0, 1]$;
    and the minimizer of $F^T [0.5, 0.5]$ satisfies x_3 = 0.

    """

    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import TR_LBFGSB
    import numpy as np
    import pytest

    # Initialize the problem dimensions
    n = 3
    o = 2
    lb = np.zeros(n)
    ub = np.ones(n)

    # Create the biobjective function and its augmented Lagrangian
    def f(z):
        return np.asarray([-z[0] + z[1] + z[2], z[0] - z[1] + z[2]])

    def L(z):
        res = np.asarray([-z[0] + z[1] + z[2], z[0] - z[1] + z[2]])
        if z[2] < 0.1:
            res[:] = res[:] + 2.0 * (0.1 - z[2])
        if z[2] > 0.6:
            res[:] = res[:] + 2.0 * (z[2] - 0.6)
        return res

    def g(z):
        res = np.asarray([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])
        if z[2] < 0.1:
            res[:, 2] = res[:, 2] - 2.0
        if z[2] > 0.6:
            res[:, 2] = res[:, 2] + 2.0
        return res

    # Create 2 acquisition functions targeting 2 "pure" solutions
    acqu1 = UniformWeights(o, lb, ub, {})
    acqu1.setTarget({}, lambda x: np.zeros(2), {})
    acqu1.weights[:] = 0.0
    acqu1.weights[0] = 1.0
    acqu2 = UniformWeights(o, lb, ub, {})
    acqu2.setTarget({}, lambda x: np.zeros(2), {})
    acqu2.weights[:] = 0.0
    acqu2.weights[1] = 1.0
    # Create a third acquisition function targeting a random tradeoff solution
    acqu3 = UniformWeights(o, lb, ub, {})
    acqu3.setTarget({}, lambda x: np.zeros(2), {})
    acqu3.weights[:] = 0.5
    # Try some bad initializations to test error handling
    with pytest.raises(ValueError):
        TR_LBFGSB(o, lb, ub, {'opt_budget': 2.0})
    with pytest.raises(ValueError):
        TR_LBFGSB(o, lb, ub, {'opt_budget': 0})
    # Initialize the problem correctly, with and without an optional budget
    TR_LBFGSB(o, lb, ub, {'opt_budget': 100})
    opt = TR_LBFGSB(o, lb, ub, {})
    # Try to add some bad objectives, constraints, and acquisitions
    with pytest.raises(ValueError):
        opt.setObjective(5)
    with pytest.raises(ValueError):
        opt.setObjective(lambda z1, z2: np.zeros(1))
    with pytest.raises(ValueError):
        opt.setConstraints(5)
    with pytest.raises(ValueError):
        opt.setConstraints(lambda z1, z2: np.zeros(1))
    with pytest.raises(ValueError):
        opt.setLagrangian(5, lambda z: np.zeros(1))
    with pytest.raises(ValueError):
        opt.setLagrangian(lambda z1, z2: np.zeros(1), lambda z: np.zeros(1))
    with pytest.raises(ValueError):
        opt.setLagrangian(lambda z: np.zeros(1), 5)
    with pytest.raises(ValueError):
        opt.setLagrangian(lambda z: np.zeros(1), lambda z1, z2: np.zeros(1))
    with pytest.raises(ValueError):
        opt.addAcquisition(5)
    with pytest.raises(ValueError):
        opt.setReset(5)
    with pytest.raises(ValueError):
        opt.setReset(lambda z1, z2: 0.0)
    # Add the correct objective and constraints
    opt.setObjective(f)
    opt.setConstraints(lambda z: np.asarray([0.1 - z[2], z[2] - 0.6]))
    opt.setLagrangian(L, g)
    opt.addAcquisition(acqu1, acqu2, acqu3)
    opt.setReset(lambda x: 100.0)
    # Try to solve with invalid inputs to test error handling
    with pytest.raises(ValueError):
        opt.solve(5)
    with pytest.raises(ValueError):
        opt.solve(np.zeros((3, n-1)))
    with pytest.raises(ValueError):
        opt.solve(np.zeros((4, n)))
    with pytest.raises(ValueError):
        opt.solve(-np.ones((3, n)))
    # Solve the surrogate problem with LBFGSB, starting from the centroid
    x = np.zeros((3, n))
    x[:] = 0.5
    (x1, x2, x3) = opt.solve(x)
    # Define the solution
    x1_soln = np.eye(n)[0]
    x1_soln[n-1] = 0.1
    x2_soln = np.eye(n)[1]
    x2_soln[n-1] = 0.1
    # eps is the tolerance for rejecting a solution as incorrect
    eps = 0.01
    # Check that the computed solutions are within eps of the truth
    assert(np.linalg.norm(x1 - x1_soln) < eps)
    assert(np.linalg.norm(x2 - x2_soln) < eps)
    assert(np.abs(x3[n-1] - 0.1) < eps)
    return


if __name__ == "__main__":
    test_LBFGSB()
    test_TR_LBFGSB()
