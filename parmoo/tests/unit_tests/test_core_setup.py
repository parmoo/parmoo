""" Unit tests for the MOOP problem-definition methods.

These tests cover the per-key validation that MOOP.addDesign(),
addSimulation(), addObjective(), addConstraint(), and addAcquisition()
perform inline.  Those ~39 raises in moop.py are not delegated to a shared
helper, so they are the one family of input checks that has to be tested here
rather than through parmoo.utilities.error_checks.

The exception is name uniqueness, which every add* method delegates to
check_names().  That contract is tested directly in
test_util_error_checks.py, so it is checked once here -- across all four
methods -- rather than re-derived per method.

"""

import numpy as np
import pytest

from parmoo import MOOP
from parmoo.acquisitions import UniformWeights
from parmoo.embeddings import IdentityEmbedder
from parmoo.optimizers import LocalSurrogate_PS
from parmoo.surrogates import GaussRBF
from parmoo.tests.unit_tests.helpers import (
    obj_x1,
    sim_dict,
    sim_norm,
    sim_shifted_norms,
)


def obj_zero(x, sx):
    """ A trivial objective. """

    return 0.0


def grad_zero(x, sx):
    """ A trivial gradient, returning the two input structures unchanged. """

    return x, sx


@pytest.fixture
def empty_moop():
    """ A MOOP with no design variables yet. """

    return MOOP(LocalSurrogate_PS)


@pytest.fixture
def three_var_moop():
    """ A MOOP with 3 continuous design variables on the unit cube. """

    moop = MOOP(LocalSurrogate_PS)
    for i in range(3):
        moop.addDesign({'lb': 0.0, 'ub': 1.0})
    return moop


@pytest.fixture
def two_sim_uncompiled(three_var_moop):
    """ A MOOP with 3 variables and 2 simulations, not yet compiled. """

    three_var_moop.addSimulation(sim_dict(1, sim_norm),
                                 sim_dict(2, sim_shifted_norms))
    return three_var_moop


# ---------------------------------------------------------------------------
# MOOP.__init__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("args, kwargs", [
    ((5.0,), {}),                                   # not a class
    ((lambda w, x, y, z: 0.0,), {}),                 # not an optimizer class
    ((LocalSurrogate_PS,), {'hyperparams': []}),     # hyperparams not a dict
])
def test_MOOP_init_rejects_bad_input(args, kwargs):
    """ Check that the MOOP constructor validates its optimizer and dict. """

    with pytest.raises(TypeError):
        MOOP(*args, **kwargs)


def test_MOOP_init_starts_empty():
    """ Check that a fresh MOOP reports zero of everything. """

    moop = MOOP(LocalSurrogate_PS)
    assert (moop.m == 0 and moop.n_feature == 0 and moop.n_latent == 0 and
            moop.s == 0 and moop.o == 0 and moop.p == 0)


def test_MOOP_init_stores_hyperparams():
    """ Check that constructor hyperparameters reach the optimizer. """

    moop = MOOP(LocalSurrogate_PS, hyperparams={'test': 0})
    assert (moop.opt_hp['test'] == 0)


# ---------------------------------------------------------------------------
# MOOP.addDesign
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arg, error", [
    ([], TypeError),                          # not a dict
    ({'des_type': 1.0}, TypeError),           # des_type not a str
    ({'des_type': "hello world"}, ValueError),  # unrecognized des_type
])
def test_addDesign_rejects_bad_input(empty_moop, arg, error):
    """ Check that addDesign() validates its argument and des_type. """

    with pytest.raises(error):
        empty_moop.addDesign(arg)
    assert (empty_moop.n_latent == 0)


@pytest.mark.parametrize("settings, n_latent", [
    ({'lb': 0.0, 'ub': 1.0}, 1),
    ({'des_type': "continuous", 'lb': 0.0, 'ub': 1.0, 'des_tol': 0.01}, 1),
    ({'des_type': "integer", 'lb': 0, 'ub': 4}, 1),
    # A binary category needs a single latent coordinate
    ({'des_type': "categorical", 'levels': 2}, 1),
    # Three or more categories are one-hot encoded, minus one for the baseline
    ({'des_type': "categorical", 'levels': 3}, 2),
    ({'des_type': "categorical", 'levels': ["boy", "girl", "doggo"]}, 2),
    ({'des_type': "custom", 'lb': -100.0, 'ub': 100.0,
      'embedder': IdentityEmbedder}, 1),
    ({'des_type': "raw", 'lb': -100.0, 'ub': 100.0}, 1),
])
def test_addDesign_latent_size(empty_moop, settings, n_latent):
    """ Check how many latent coordinates each design variable type occupies.

    """

    empty_moop.addDesign(settings)
    assert (empty_moop.n_latent == n_latent)


def test_addDesign_accumulates_mixed_types(empty_moop):
    """ Check the running latent size as variables of every type are added.

    Also exercises the automatic naming of unnamed variables (x1, x2, ...),
    since a named variable is mixed in partway through.

    """

    expected = [1, 2, 3, 4, 6, 8, 9, 10]
    settings = [
        {'lb': 0.0, 'ub': 1.0},
        {'name': "x2", 'des_type': "continuous", 'lb': 0.0, 'ub': 1.0,
         'des_tol': 0.01},
        {'des_type': "integer", 'lb': 0, 'ub': 4},
        {'des_type': "categorical", 'levels': 2},
        {'des_type': "categorical", 'levels': 3},
        {'name': "x6", 'des_type': "categorical",
         'levels': ["boy", "girl", "doggo"]},
        {'des_type': "custom", 'lb': -100.0, 'ub': 100.0,
         'embedder': IdentityEmbedder},
        {'des_type': "raw", 'lb': -100.0, 'ub': 100.0},
    ]
    for si, ni in zip(settings, expected):
        empty_moop.addDesign(si)
        assert (empty_moop.n_latent == ni)


# ---------------------------------------------------------------------------
# MOOP.addSimulation
# ---------------------------------------------------------------------------


def test_addSimulation_updates_counts(three_var_moop):
    """ Check that adding simulations accumulates the output count. """

    three_var_moop.addSimulation(sim_dict(1, sim_norm))
    assert (three_var_moop.m == 1 and three_var_moop.n_latent == 3 and
            three_var_moop.s == 1 and three_var_moop.o == 0 and
            three_var_moop.p == 0)
    three_var_moop.addSimulation(sim_dict(2, sim_shifted_norms))
    assert (three_var_moop.m == 3 and three_var_moop.s == 2)


def test_addSimulation_names(three_var_moop):
    """ Check that unnamed simulations are auto-named and names are kept. """

    three_var_moop.addSimulation(sim_dict(1, sim_norm),
                                 sim_dict(2, sim_shifted_norms))
    three_var_moop.addSimulation(sim_dict(1, sim_norm, name="Bobo1"),
                                 sim_dict(2, sim_shifted_norms, name="Bobo2"))
    names = [si[0] for si in three_var_moop.sim_schema]
    assert (names == ["sim1", "sim2", "Bobo1", "Bobo2"])


# ---------------------------------------------------------------------------
# MOOP.addObjective and MOOP.addConstraint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arg, error", [
    (0, TypeError),                                 # not a dict
    ({}, AttributeError),                           # missing obj_func
    ({'obj_func': 0}, TypeError),                   # obj_func not callable
    ({'obj_func': lambda x: 0.0}, ValueError),      # wrong arity
])
def test_addObjective_rejects_bad_input(two_sim_uncompiled, arg, error):
    """ Check that addObjective() validates its dict and the callable. """

    with pytest.raises(error):
        two_sim_uncompiled.addObjective(arg)
    assert (two_sim_uncompiled.o == 0)


def test_addObjective_counts_and_names(two_sim_uncompiled):
    """ Check objective accumulation, auto-naming, and gradient acceptance. """

    two_sim_uncompiled.addObjective({'obj_func': obj_zero})
    assert (two_sim_uncompiled.o == 1)
    # Several objectives may be added in one call, with or without gradients
    two_sim_uncompiled.addObjective({'obj_func': obj_zero},
                                    {'obj_func': obj_zero,
                                     'obj_grad': grad_zero})
    assert (two_sim_uncompiled.o == 3)
    two_sim_uncompiled.addObjective({'name': "Bobo", 'obj_func': obj_zero})
    assert (two_sim_uncompiled.o == 4)
    assert (two_sim_uncompiled.obj_schema == [("f1", 'f8'), ("f2", 'f8'),
                                              ("f3", 'f8'), ("Bobo", 'f8')])


@pytest.mark.parametrize("key", ["con_func", "constraint"])
@pytest.mark.parametrize("value, error", [
    (0, TypeError),                     # not callable
    (lambda x: 0.0, ValueError),        # wrong arity
])
def test_addConstraint_rejects_bad_callable(two_sim_uncompiled, key, value,
                                            error):
    """ Check the constraint callable contract under both accepted key names.

    addConstraint() accepts the callable under either 'con_func' or the older
    'constraint' key, and validates both the same way.

    """

    with pytest.raises(error):
        two_sim_uncompiled.addConstraint({key: value})
    assert (two_sim_uncompiled.p == 0)


@pytest.mark.parametrize("arg, error", [
    (0, TypeError),             # not a dict
    ({}, AttributeError),       # missing the callable
])
def test_addConstraint_rejects_bad_input(two_sim_uncompiled, arg, error):
    """ Check that addConstraint() validates its argument dict. """

    with pytest.raises(error):
        two_sim_uncompiled.addConstraint(arg)
    assert (two_sim_uncompiled.p == 0)


def test_addConstraint_counts_and_names(two_sim_uncompiled):
    """ Check constraint accumulation, naming, and gradient acceptance.

    """

    two_sim_uncompiled.addConstraint({'constraint': obj_zero})
    assert (two_sim_uncompiled.p == 1)
    two_sim_uncompiled.addConstraint({'con_func': obj_zero},
                                     {'con_func': obj_zero,
                                      'con_grad': grad_zero})
    assert (two_sim_uncompiled.p == 3)
    two_sim_uncompiled.addConstraint({'name': "Bobo",
                                      'constraint': obj_zero})
    assert (two_sim_uncompiled.p == 4)
    assert (two_sim_uncompiled.con_schema ==
            [("c1", 'f8'), ("c2", 'f8'), ("c3", 'f8'), ("Bobo", 'f8')])


# ---------------------------------------------------------------------------
# Name uniqueness, which every add* method delegates to check_names()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method, arg", [
    ("addDesign", {'name': "dup", 'lb': 0.0, 'ub': 1.0}),
    ("addSimulation", None),        # filled in below, needs sim_dict
    ("addObjective", {'name': "dup", 'obj_func': obj_zero}),
    ("addConstraint", {'name': "dup", 'con_func': obj_zero}),
])
def test_add_rejects_duplicate_name(three_var_moop, method, arg):
    """ Check that each add* method rejects a name already in use.

    The uniqueness rule itself lives in check_names(), which is tested
    directly in test_util_error_checks.py; this confirms each method is wired
    up to it.

    """

    if arg is None:
        arg = sim_dict(1, sim_norm, name="dup")
    call = getattr(three_var_moop, method)
    call(arg)
    with pytest.raises(ValueError):
        call(arg)


# ---------------------------------------------------------------------------
# MOOP.addAcquisition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arg, error", [
    (0, TypeError),                                                # not a dict
    ({}, AttributeError),                                          # no key
    ({'acquisition': UniformWeights, 'hyperparams': 0}, TypeError),
    ({'acquisition': 0, 'hyperparams': {}}, TypeError),        # not a class
    ({'acquisition': GaussRBF, 'hyperparams': {}}, TypeError),  # wrong ABC
])
def test_addAcquisition_rejects_bad_input(three_var_moop, arg, error):
    """ Check that addAcquisition() validates its dict and the class given.

    The class is probed by instantiating it with dummy (dim, lb, ub,
    hyperparams) arguments, which is how a non-acquisition class is caught.

    """

    three_var_moop.addObjective({'obj_func': obj_zero})
    with pytest.raises(error):
        three_var_moop.addAcquisition(arg)
    assert (len(three_var_moop.acquisitions) == 0)


def test_addAcquisition_counts(three_var_moop):
    """ Check that acquisitions are only instantiated at compile time. """

    for i in range(3):
        three_var_moop.addObjective({'obj_func': obj_zero})
    three_var_moop.addAcquisition({'acquisition': UniformWeights})
    three_var_moop.addAcquisition({'acquisition': UniformWeights},
                                  {'acquisition': UniformWeights,
                                   'hyperparams': {}})
    # Nothing is built until compile()
    assert (len(three_var_moop.acquisitions) == 0)
    three_var_moop.compile()
    assert (len(three_var_moop.acquisitions) == 3)


# ---------------------------------------------------------------------------
# dtype accessors
# ---------------------------------------------------------------------------


def test_getTypes_are_None_when_empty():
    """ Check that the dtype accessors are None before anything is added.

    """

    moop = MOOP(LocalSurrogate_PS)
    assert (moop.getDesignType() is None)
    assert (moop.getSimulationType() is None)
    assert (moop.getObjectiveType() is None)
    assert (moop.getConstraintType() is None)


def test_getTypes_describe_the_schemas():
    """ Check that each dtype accessor returns a usable numpy dtype. """

    moop = MOOP(LocalSurrogate_PS)
    moop.addDesign({'name': "x1", 'lb': 0.0, 'ub': 1.0})
    moop.addDesign({'name': "x2", 'des_type': "categorical", 'levels': 3})
    moop.addSimulation(sim_dict(1, sim_norm))
    moop.addObjective({'obj_func': obj_x1})
    moop.addConstraint({'constraint': obj_x1})
    for dtype in [moop.getDesignType(), moop.getSimulationType(),
                  moop.getObjectiveType(), moop.getConstraintType()]:
        assert (np.zeros(1, dtype=dtype).size == 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
