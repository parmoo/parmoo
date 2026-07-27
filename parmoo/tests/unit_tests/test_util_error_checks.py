""" Unit tests for parmoo.utilities.error_checks.

These four helpers are the single point of truth for the input validation that
is shared across ParMOO's plugin classes.  Because ~40 call sites in the
library delegate to them, testing them directly here is what makes it
unnecessary to re-test the same corner cases through every plugin.

"""

import numpy as np
import pytest

from parmoo.utilities.error_checks import (
    check_names,
    get_hp,
    gradient_error,
    xerror,
)


def test_xerror():
    """ Check that the xerror() utility handles bad input correctly.

    Provide several bad inputs to xerror() and confirm that it raises
    the appropriate ValueErrors.

    """

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


def test_get_hp_bad_hyperparams():
    """ Check that get_hp() rejects a hyperparams argument that is not a dict.

    """

    with pytest.raises(TypeError):
        get_hp("budget", [], int, lambda x: True, 1000)


def test_get_hp_required_key():
    """ Check that get_hp() treats a default_value of None as "required".

    A default_value of None is the contract for a required key, and must
    raise a KeyError rather than returning None.

    """

    # Matched on the message: a bare KeyError would also be raised by
    # the hyperparams[key] lookup further down, so this pins the guard.
    with pytest.raises(KeyError, match="required key"):
        get_hp("budget", {}, int, lambda x: True, None)


def test_get_hp_default_value():
    """ Check that get_hp() returns the default when the key is absent. """

    assert get_hp("budget", {}, int, lambda x: True, 1000) == 1000
    assert get_hp("budget", {'other': 1}, int, lambda x: True, 1000) == 1000


def test_get_hp_expected_type():
    """ Check that get_hp() enforces the expected type of the value. """

    with pytest.raises(TypeError, match="must be an instance of"):
        get_hp("budget", {'budget': 2.0}, int, lambda x: True, 1000)
    with pytest.raises(TypeError):
        get_hp("tols", {'tols': 0.1}, np.ndarray, lambda x: True, None)


def test_get_hp_is_legal():
    """ Check that get_hp() enforces the is_legal() predicate. """

    with pytest.raises(ValueError):
        get_hp("budget", {'budget': 0}, int, lambda x: x >= 1, 1000)
    with pytest.raises(ValueError):
        get_hp("nugget", {'nugget': -1.0}, float, lambda x: x >= 0.0, 0.0)


def test_get_hp_legal_value():
    """ Check that get_hp() returns a present, well-typed, legal value. """

    assert get_hp("budget", {'budget': 20}, int, lambda x: x >= 1, 1000) == 20
    tols = np.ones(3) * 0.1
    assert np.all(get_hp("tols", {'tols': tols}, np.ndarray,
                         lambda x: np.all(x > 0), None) == tols)


def test_check_names():
    """ Check that check_names() enforces string names and uniqueness.

    check_names() is called with all four schemas by both MOOP and
    NumpyDatabase, so a name may not collide with a design variable,
    simulation, objective, or constraint.

    """

    des_schema = [("x1", "f8")]
    sim_schema = [("s1", "f8")]
    obj_schema = [("f1", "f8")]
    con_schema = [("c1", "f8")]
    schemas = (des_schema, sim_schema, obj_schema, con_schema)

    # Every variable name must be a string
    with pytest.raises(TypeError):
        check_names(5, *schemas)
    # A name that is already in use in any schema must be rejected
    for taken in ["x1", "s1", "f1", "c1"]:
        with pytest.raises(ValueError):
            check_names(taken, *schemas)
    # An unused name passes against all four schemas
    check_names("x2", *schemas)
    # Checking against no schemas at all is legal
    check_names("x1")


def test_gradient_error():
    """ Check that gradient_error() raises, as the incomplete-gradient stub.

    gradient_error() is not a checker: MOOP_base._jit_all() assigns it as the
    backward pass whenever the set of user gradients is incomplete, so calling
    it must raise.

    """

    with pytest.raises(ValueError):
        gradient_error(None, None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
