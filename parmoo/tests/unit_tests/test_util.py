
def test_xerror():
    """ Check that the xerror() utility handles bad input correctly.

    Provide several bad inputs to xerror() and confirm that it raises
    the appropriate ValueErrors.

    """

    from parmoo.util import xerror
    import numpy as np
    import pytest

    # Try some bad initializations to test error handling
    with pytest.raises(TypeError):
        xerror(1.0, np.zeros(4), np.ones(4), {})
    with pytest.raises(ValueError):
        xerror(0, np.zeros(4), np.ones(4), {})
    with pytest.raises(TypeError):
        xerror(3, np.zeros(4), 1.0, {})
    with pytest.raises(TypeError):
        xerror(3, 0.0, np.ones(4), {})
    with pytest.raises(ValueError):
        xerror(3, np.zeros(3), np.ones(4), {})
    with pytest.raises(ValueError):
        xerror(3, np.ones(4), np.zeros(4), {})
    with pytest.raises(TypeError):
        xerror(3, np.zeros(4), np.ones(4), 5)
    # Perform two good initialization, to confirm that it passes
    xerror(o=3, lb=np.zeros(4), ub=np.ones(4), hyperparams={})
    xerror()


def test_check_sims():
    """ Check that the check_sims() utility handles bad input correctly.

    Provide several bad simulation dictionaries to check_sims() and
    confirm that it raises the appropriate ValueErrors.

    """

    from parmoo.util import check_sims
    from parmoo.surrogates import GaussRBF
    from parmoo.searches import LatinHypercube
    import numpy as np
    import pytest

    # Try providing invalid/incompatible simulation dictionaries
    with pytest.raises(TypeError):
        check_sims(3, 5.0)
    simdict = {}
    with pytest.raises(AttributeError):
        check_sims(3, simdict)
    simdict['m'] = 1.0
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['m'] = -1
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['m'] = 1
    with pytest.raises(AttributeError):
        check_sims(3, simdict)
    simdict['search'] = LatinHypercube(1, np.zeros(3), np.ones(3), {})
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['search'] = GaussRBF
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['search'] = LatinHypercube
    with pytest.raises(AttributeError):
        check_sims(3, simdict)
    simdict['surrogate'] = GaussRBF(1, np.zeros(1), np.ones(1), {})
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['surrogate'] = LatinHypercube
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['surrogate'] = GaussRBF
    with pytest.raises(AttributeError):
        check_sims(3, simdict)
    simdict['sim_func'] = {}
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['sim_func'] = lambda x, y, z: [np.linalg.norm(x - y - z)]
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_func'] = lambda x: [np.linalg.norm(x)]
    simdict['hyperparams'] = []
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['hyperparams'] = {}
    simdict['sim_db'] = 5
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': []}
    with pytest.raises(AttributeError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': "hel", 's_vals': "lo"}
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': np.asarray([0.0]), 's_vals': []}
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': np.asarray([[0.0, 0.0]]), 's_vals': [[0.0]]}
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': np.asarray([[0.0, 0.0, 0.0]]),
                         's_vals': [[0.0, 0.0]]}
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': np.asarray([[0.0, 0.0, 0.0]]),
                         's_vals': [[0.0], [1.0]]}
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['sim_db'] = {'x_vals': [], 's_vals': []}
    simdict['des_tol'] = 1
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    simdict['des_tol'] = 0.0
    with pytest.raises(ValueError):
        check_sims(3, simdict)
    simdict['des_tol'] = 0.00000001
    simdict['name'] = 5
    with pytest.raises(TypeError):
        check_sims(3, simdict)
    # Do one good check, and make sure nothing was raised
    simdict['name'] = "sim1"
    check_sims(3, simdict)


def test_lex_leq():
    """ Check that lex_leq() performs correct lexicographic comparisons.

    Perform several arrays lexicographically and check that the results are
    correct.

    """

    from parmoo.util import lex_leq
    import numpy as np

    # Check for a < b
    assert (lex_leq(np.zeros(3), np.ones(3)))
    assert (lex_leq(np.zeros(3), np.asarray([0.1, 0.0, 0.0])))
    assert (lex_leq(np.asarray([100.0, 0.0]), np.asarray([0.0, 0.1])))
    assert (lex_leq(np.zeros(1), np.ones(1)))
    # Check for a = b
    assert (lex_leq(np.zeros(3), np.zeros(3)))
    assert (lex_leq(np.ones(1), np.ones(1)))
    # Check for a > b
    assert (not lex_leq(np.ones(3), np.zeros(3)))
    assert (not lex_leq(np.ones(1), np.zeros(1)))
    assert (not lex_leq(np.asarray([0.1, 0.0, 0.0]), np.zeros(3)))
    assert (not lex_leq(np.asarray([0.0, 0.1]), np.asarray([1.0, 0.0])))
    # Check for mismatched dimensions
    assert (lex_leq(np.zeros(3), np.zeros(4)))
    assert (lex_leq(np.zeros(4), np.zeros(3)))
    assert (lex_leq(np.zeros(1), np.zeros(1)))


def test_updatePF():
    """ Test the updatePF function.

    Perform a test of the updatePF() function by extracting the Pareto front
    from a 3 objective problem with a 5 point database. Then perform an
    update by adding another 5 points to the database.

    Finally, add a constraint and check the updated solution.

    """

    from parmoo.util import updatePF
    import numpy as np

    # Set problem dimensions and initialize a RandomSearch object
    n = 4

    # Define the objective
    def obj(x): return np.asarray([np.linalg.norm(x - np.eye(n)[0, :]) ** 2.0,
                                   np.linalg.norm(x - np.eye(n)[1, :]) ** 2.0,
                                   np.linalg.norm(x - np.eye(n)[2, :]) ** 2.0])

    # Create a database of 10 points
    data = {'x_vals': np.zeros((10, 4)),
            'f_vals': np.zeros((10, 3)),
            'c_vals': np.zeros((10, 1))}
    data['x_vals'][0, :] = np.asarray([0.0, 0.0, 0.0, 0.0])
    data['f_vals'][0, :] = obj(np.asarray([0.0, 0.0, 0.0, 0.0]))
    data['x_vals'][1, :] = np.asarray([1.0, 0.0, 0.0, 0.1])
    data['f_vals'][1, :] = obj(np.asarray([1.0, 0.0, 0.0, 0.1]))
    data['x_vals'][2, :] = np.asarray([0.0, 1.0, 0.0, 0.0])
    data['f_vals'][2, :] = obj(np.asarray([0.0, 1.0, 0.0, 0.0]))
    data['x_vals'][3, :] = np.asarray([0.0, 0.0, 1.0, 0.0])
    data['f_vals'][3, :] = obj(np.asarray([0.0, 0.0, 1.0, 0.0]))
    data['x_vals'][4, :] = np.asarray([0.0, 0.0, 0.0, 1.0])
    data['f_vals'][4, :] = obj(np.asarray([0.0, 0.0, 0.0, 1.0]))
    data['x_vals'][5, :] = np.asarray([0.5, 0.5, 0.5, 0.5])
    data['f_vals'][5, :] = obj(np.asarray([0.5, 0.5, 0.5, 0.5]))
    data['x_vals'][6, :] = np.asarray([0.5, 0.5, 0.5, 0.0])
    data['f_vals'][6, :] = obj(np.asarray([0.5, 0.5, 0.5, 0.0]))
    data['x_vals'][7, :] = np.asarray([1.0, 0.5, 0.5, 0.0])
    data['f_vals'][7, :] = obj(np.asarray([1.0, 0.5, 0.5, 0.0]))
    data['x_vals'][8, :] = np.asarray([0.0, 0.0, 1.0, 0.0])
    data['f_vals'][8, :] = obj(np.asarray([0.0, 0.0, 1.0, 0.0]))
    data['x_vals'][9, :] = np.asarray([1.0, 0.0, 0.0, 0.0])
    data['f_vals'][9, :] = obj(np.asarray([1.0, 0.0, 0.0, 0.0]))
    # Extract the Pareto front for the first 5 points
    soln = {}
    def NoConstraints(x): return np.asarray([])
    soln = updatePF({'x_vals': data['x_vals'][:5, :],
                     'f_vals': data['f_vals'][:5, :],
                     'c_vals': data['c_vals'][:5, :]}, soln)
    assert (soln['f_vals'].shape == (4, 3))
    # Update the Pareto front with the last 5 points
    soln = updatePF({'x_vals': data['x_vals'][5:, :],
                     'f_vals': data['f_vals'][5:, :],
                     'c_vals': data['c_vals'][:5, :]}, soln)
    assert (soln['f_vals'].shape == (5, 3))
    # Add a constraint and re-filter
    def Constraints(x): return np.asarray([0.1 - obj(x)[0]])
    for i in range(10):
        data['c_vals'][i, :] = Constraints(data['x_vals'][i, :])
    # Extract the Pareto front for the first 5 points
    soln = {}
    soln = updatePF({'x_vals': data['x_vals'][:5, :],
                     'f_vals': data['f_vals'][:5, :],
                     'c_vals': data['c_vals'][:5, :]}, soln)
    assert (soln['f_vals'].shape == (3, 3))
    # Update the Pareto front with the last 5 points
    soln = updatePF({'x_vals': data['x_vals'][5:, :],
                     'f_vals': data['f_vals'][5:, :],
                     'c_vals': data['c_vals'][5:, :]}, soln)
    assert (soln['f_vals'].shape == (4, 3))


def test_to_from_array():
    """ Test the from_array function. """

    from parmoo.util import from_array, to_array
    import numpy as np
    import pytest

    # Create test inputs
    dt_bad = "hello world"
    x_bad = "hello world"
    x_unnamed = np.eye(5)[2]
    dt_named = np.dtype([("x1", "f8"), ("x2", "f8"), ("x3", "f8", 3)])
    x_named = {"x1": 0.0, "x2": 0.0, "x3": np.asarray([1.0, 0.0, 0.0])}
    # Test packing into array and unpacking from an array
    assert (np.all(x_unnamed == to_array(x_named, dt_named)))
    assert (all([all(x_named[j] == from_array(x_unnamed, dt_named)[j])
                 for j in x_named]))


if __name__ == "__main__":
    test_xerror()
    test_check_sims()
    test_lex_leq()
    test_updatePF()
    test_to_from_array()
