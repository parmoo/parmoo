
def test_GlobalSurrogate_BFGS():
    """ Test the LBFGS class in optimizers.py.

    Perform a test of the GlobalSurrogate_BFGS class by minimizing the three
    variable, biobjective function

    $$
    F(x) = (-x_1 + x_2 + x_3, x_1 - x_2 + x_3)
    $$

    s.t. $x in [0, 1]^n$.

    Use the weights [1, 0], [0, 1], and [0.5, 0.5].

    Assert that the solutions are all correct. I.e., (1, 0, 0) is
    the minimizer of $F^T [1, 0]$; (0, 1, 0) is the minimizer of $F^T [0, 1]$;
    and the minimizer of $F^T [0.5, 0.5]$ satisfies x_3 = 0.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    from jax import numpy as jnp
    import numpy as np
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import GlobalSurrogate_BFGS
    import pytest

    # Initialize the problem dimensions
    n = 3
    o = 2
    lb = np.zeros(n)
    ub = np.ones(n)
    # Create the biobjective function and its penalty function
    def f(z, sz): return jnp.asarray([-z[0] + z[1] + z[2], z[0] - z[1] + z[2]])

    def S(z): return jnp.ones(1)

    def L(z, sz):
        return f(z, sz) + 2 * (jnp.maximum(0.1 - z[2], 0) +
                               jnp.maximum(z[2] - 0.6, 0))

    def SD(z): return jnp.zeros(1)
    # Create 3 acquisition functions targeting various pure/tradeoff solutions
    acqu1 = UniformWeights(o, lb, ub, {})
    acqu1.setTarget({}, L)
    acqu1.weights[:] = 0.0
    acqu1.weights[0] = 1.0
    acqu2 = UniformWeights(o, lb, ub, {})
    acqu2.setTarget({}, L)
    acqu2.weights[:] = 0.0
    acqu2.weights[1] = 1.0
    acqu3 = UniformWeights(o, lb, ub, {})
    acqu3.setTarget({}, L)
    acqu3.weights[:] = 0.5
    # Try some bad initializations to test error handling
    with pytest.raises(TypeError):
        GlobalSurrogate_BFGS(o, lb, ub, {'opt_budget': 2.0})
    with pytest.raises(ValueError):
        GlobalSurrogate_BFGS(o, lb, ub, {'opt_budget': 0})
    # Initialize the problem correctly, with and without an optional budget
    GlobalSurrogate_BFGS(o, lb, ub, {'opt_budget': 100})
    opt = GlobalSurrogate_BFGS(o, lb, ub, {})
    # Try to add some bad objectives, constraints, and acquisitions
    with pytest.raises(TypeError):
        opt.setObjective(5)
    with pytest.raises(ValueError):
        opt.setObjective(lambda z1, z2, z3: np.zeros(1))
    with pytest.raises(TypeError):
        opt.setConstraints(5)
    with pytest.raises(ValueError):
        opt.setConstraints(lambda z1, z2, z3: np.zeros(1))
    with pytest.raises(TypeError):
        opt.setPenalty(5)
    with pytest.raises(ValueError):
        opt.setPenalty(lambda z1, z2, z3: np.zeros(1))
    with pytest.raises(TypeError):
        opt.addAcquisition(5)
    # Try to solve with invalid inputs to test error handling
    with pytest.raises(ValueError):
        opt.solve(np.zeros((3, n-1)))
    with pytest.raises(ValueError):
        opt.solve(np.zeros((4, n)))
    with pytest.raises(ValueError):
        opt.solve(-np.ones((3, n)))
    # Add the correct objective and constraints
    opt.setObjective(f)
    opt.setConstraints(lambda z, sz: np.asarray([0.1 - z[2], z[2] - 0.6]))
    opt.setSimulation(S, SD)
    opt.setPenalty(L)
    opt.addAcquisition(acqu1, acqu2, acqu3)
    opt.setTrFunc(lambda x, r: 100.0)
    # Solve the surrogate problem with GlobalSurrogate_BFGS
    x0 = np.zeros((3, n))
    x0[:] = 0.5
    (x1, x2, x3) = opt.solve(x0)
    # Define the solution
    x1_soln = np.eye(n)[0]
    x1_soln[n-1] = 0.1
    x2_soln = np.eye(n)[1]
    x2_soln[n-1] = 0.1
    # eps is the tolerance for rejecting a solution as incorrect
    eps = 0.01
    # Check that the computed solutions are within eps of the truth
    assert (np.linalg.norm(x1 - x1_soln) < eps)
    assert (np.linalg.norm(x2 - x2_soln) < eps)
    assert (np.abs(x3[n-1] - 0.1) < eps)
    return


def test_LocalSurrogate_BFGS():
    """ Test the LocalSurrogate_BFGS class in optimizers.py.

    Perform a test of the LocalSurrogate_BFGS class by minimizing the three
    variable, biobjective function

    $$
    F(x) = (-x_1 + x_2 + x_3, x_1 - x_2 + x_3)
    $$

    s.t. $x in [0, 1]^n$.

    Use the weights [1, 0], [0, 1], and [0.5, 0.5].

    Assert that the solutions are all correct. I.e., (1, 0, 0) is
    the minimizer of $F^T [1, 0]$; (0, 1, 0) is the minimizer of $F^T [0, 1]$;
    and the minimizer of $F^T [0.5, 0.5]$ satisfies x_3 = 0.

    """

    from jax import config
    config.update("jax_enable_x64", True)
    from jax import numpy as jnp
    import numpy as np
    from parmoo.acquisitions import UniformWeights
    from parmoo.optimizers import LocalSurrogate_BFGS
    import pytest

    # Initialize the problem dimensions
    n = 3
    o = 2
    lb = np.zeros(n)
    ub = np.ones(n)
    # Create the biobjective function and its penalty function
    def f(z, sz): return jnp.asarray([-z[0] + z[1] + z[2], z[0] - z[1] + z[2]])

    def S(z): return jnp.ones(1)

    def L(z, sz):
        return f(z, sz) + 2 * (jnp.maximum(0.1 - z[2], 0) +
                               jnp.maximum(z[2] - 0.6, 0))

    def SD(z): return jnp.zeros(1)
    # Create 3 acquisition functions targeting various pure/tradeoff solutions
    acqu1 = UniformWeights(o, lb, ub, {})
    acqu1.setTarget({}, L)
    acqu1.weights[:] = 0.0
    acqu1.weights[0] = 1.0
    acqu2 = UniformWeights(o, lb, ub, {})
    acqu2.setTarget({}, L)
    acqu2.weights[:] = 0.0
    acqu2.weights[1] = 1.0
    acqu3 = UniformWeights(o, lb, ub, {})
    acqu3.setTarget({}, L)
    acqu3.weights[:] = 0.5
    # Try some bad initializations to test error handling
    with pytest.raises(TypeError):
        LocalSurrogate_BFGS(o, lb, ub, {'opt_budget': 2.0})
    with pytest.raises(ValueError):
        LocalSurrogate_BFGS(o, lb, ub, {'opt_budget': 0})
    # Initialize the problem correctly, with and without an optional budget
    LocalSurrogate_BFGS(o, lb, ub, {'opt_budget': 100})
    opt = LocalSurrogate_BFGS(o, lb, ub, {})
    # Try to add some bad objectives, constraints, and acquisitions
    with pytest.raises(TypeError):
        opt.setObjective(5)
    with pytest.raises(ValueError):
        opt.setObjective(lambda z1, z2, z3: np.zeros(1))
    with pytest.raises(TypeError):
        opt.setConstraints(5)
    with pytest.raises(ValueError):
        opt.setConstraints(lambda z1, z2, z3: np.zeros(1))
    with pytest.raises(TypeError):
        opt.setPenalty(5)
    with pytest.raises(ValueError):
        opt.setPenalty(lambda z1, z2, z3: np.zeros(1))
    with pytest.raises(TypeError):
        opt.addAcquisition(5)
    with pytest.raises(TypeError):
        opt.setTrFunc(5)
    with pytest.raises(ValueError):
        opt.setTrFunc(lambda z1: 0.0)
    # Try to solve with invalid inputs to test error handling
    with pytest.raises(ValueError):
        opt.solve(np.zeros((3, n-1)))
    with pytest.raises(ValueError):
        opt.solve(np.zeros((4, n)))
    with pytest.raises(ValueError):
        opt.solve(-np.ones((3, n)))
    # Set the acquisition function and TR reset functions
    opt.setSimulation(S, SD)
    opt.addAcquisition(acqu1, acqu2, acqu3)
    opt.setTrFunc(lambda x, r: 100.0)
    # Define the solution
    x1_soln = np.eye(n)[0]
    x1_soln[n-1] = 0.1
    x2_soln = np.eye(n)[1]
    x2_soln[n-1] = 0.1
    # Solve the surrogate problem with LBFGSB, starting from the centroid
    x0 = np.zeros((3, n))
    x0[:] = 0.5
    for i in range(6):
        # Add the correct objective and constraints
        opt.setObjective(f)
        opt.setConstraints(lambda z, sz: np.asarray([0.1 - z[2], z[2] - 0.6]))
        opt.setPenalty(L)
        (x1, x2, x3) = opt.solve(x0)
        x0[0] = x1
        x0[1] = x2
        x0[2] = x3
        for j in range(3):
            opt.returnResults(x0[j], np.ones(2) * -10,
                              np.zeros(1), np.zeros(1))
    # eps is the tolerance for rejecting a solution as incorrect
    eps = 0.01
    # Check that the computed solutions are within eps of the truth
    assert (np.linalg.norm(x0[0] - x1_soln) < eps)
    assert (np.linalg.norm(x0[1] - x2_soln) < eps)
    assert (np.abs(x0[2, n-1] - 0.1) < eps)
    return


if __name__ == "__main__":
    test_GlobalSurrogate_BFGS()
    test_LocalSurrogate_BFGS()
