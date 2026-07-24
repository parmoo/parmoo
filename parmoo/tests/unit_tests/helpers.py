""" Shared builders and callables for the ParMOO unit tests.

Every callable that a checkpointing test may need to save and reload lives at
module global scope here, because ``MOOP.save()`` stores simulation,
objective, and constraint functions by ``(class_name, module_name)`` reference.
Lambdas and closures cannot be checkpointed, so the named functions below are
used in place of the inline lambdas the tests previously defined.

"""

import numpy as np

from parmoo import MOOP
from parmoo.acquisitions import UniformWeights
from parmoo.databases import NumpyDatabase
from parmoo.optimizers import LocalSurrogate_PS
from parmoo.searches import LatinHypercube
from parmoo.surrogates import GaussRBF

# The design points that the surrogate-training helper evaluates: the origin,
# the centroid, the three unit vectors, and the all-ones corner.
TRAINING_POINTS = [
    {'x1': 0, 'x2': 0, 'x3': 0},
    {'x1': 0.5, 'x2': 0.5, 'x3': 0.5},
    {'x1': 1, 'x2': 0, 'x3': 0},
    {'x1': 0, 'x2': 1, 'x3': 0},
    {'x1': 0, 'x2': 0, 'x3': 1},
    {'x1': 1, 'x2': 1, 'x3': 1},
]


def seeded(**extra):
    """ Return a fresh hyperparams dict with a fixed random seed.

    This must build a new dict on every call.  MOOP.__init__ stores the
    hyperparams argument by reference and then writes into it (it replaces the
    'np_random_gen' seed with the Generator it built), so a shared dict would
    leak state from one MOOP into the next.

    Args:
        **extra: Additional hyperparameters to merge in.

    Returns:
        dict: A hyperparams dict seeded for reproducibility.

    """

    return {'np_random_gen': 0, **extra}


def makeNumpyDatabase(with_constraints=True):
    """ Create a NumpyDatabase object for testing.

    Args:
        with_constraints (bool, optional): An optional variable that can be
            used to create a database with no constraints (by setting to
            False). Defaults to True.

    Returns:
        NumpyDatabase: A database with 3 design variables ("x1", "x2", "x3"), 2
        simulations ("s1", "s2"), and 2 objectives ("f1", "f2").  If
        with_constraints is set (default) then there are also 2 constraints
        ("c1", "c2"); otherwise, there are no constraints.

    """

    db = NumpyDatabase({})
    db.addDesign("x1", "f8", 0.01)
    db.addDesign("x2", "i4", 0)
    db.addDesign("x3", "U25", 0)
    db.addSimulation("s1", 1)
    db.addSimulation("s2", 4)
    db.addObjective("f1")
    db.addObjective("f2")
    if with_constraints:
        db.addConstraint("c1")
        db.addConstraint("c2")
    return db


# ---------------------------------------------------------------------------
# Simulation functions
# ---------------------------------------------------------------------------


def sim_norm(x):
    """ A 1 output simulation returning the 2-norm of the design point. """

    return [np.linalg.norm([x[key] for key in x])]


def sim_shifted_norms(x):
    """ A 2 output simulation returning norms shifted by 1.0 and 0.5. """

    return [np.linalg.norm([x[key] - 1.0 for key in x]),
            np.linalg.norm([x[key] - 0.5 for key in x])]


def sim_identity(x):
    """ A simulation returning the design variables unchanged. """

    return [x[key] for key in x]


# ---------------------------------------------------------------------------
# Objective and constraint functions
# ---------------------------------------------------------------------------


def obj_x1(x, sx):
    """ The first design variable. """

    return x["x1"]


def obj_sim1(x, sx):
    """ The whole (length 1) output of the first simulation. """

    return sx["sim1"]


def obj_sim1_first(x, sx):
    """ The first output of the first simulation. """

    return sx["sim1"][0]


def obj_sim2_sum(x, sx):
    """ The sum of the second simulation's outputs. """

    return sum(sx["sim2"])


def obj_sim2_pair_sum(x, sx):
    """ The sum of the second simulation's two outputs, indexed explicitly. """

    return sx["sim2"][0] + sx["sim2"][1]


def con_x1(x, sx):
    """ A constraint on the first design variable. """

    return x["x1"]


def con_x1_offset(x, sx):
    """ A constraint on the first design variable, active above 0.5. """

    return x["x1"] - 0.5


def con_sim1_first(x, sx):
    """ A constraint on the first output of the first simulation. """

    return sx["sim1"][0]


def con_sim1(x, sx):
    """ A constraint on the whole (length 1) first simulation output. """

    return sx["sim1"]


def con_sim2_sum(x, sx):
    """ A constraint on the sum of the second simulation's outputs. """

    return sum(sx["sim2"])


def con_sim2_pair_sum(x, sx):
    """ A constraint on the second simulation's two outputs. """

    return sx["sim2"][0] + sx["sim2"][1]


# ---------------------------------------------------------------------------
# MOOP builders
# ---------------------------------------------------------------------------


def sim_dict(m, sim_func, name=None, hyperparams=None):
    """ Build a simulation dict for MOOP.addSimulation().

    Args:
        m (int): The number of simulation outputs.

        sim_func (callable): The simulation function.

        name (str, optional): The simulation name.  Defaults to unnamed, which
            makes ParMOO assign "sim1", "sim2", and so on.

        hyperparams (dict, optional): Hyperparameters for the search and
            surrogate.

    Returns:
        dict: A simulation dict using LatinHypercube and GaussRBF.

    """

    sim = {'m': m,
           'search': LatinHypercube,
           'sim_func': sim_func,
           'surrogate': GaussRBF,
           'hyperparams': hyperparams if hyperparams is not None else {}}
    if name is not None:
        sim['name'] = name
    return sim


def two_sim_moop(objectives=(), constraints=(), bounds=None, opt=None,
                 hyperparams=None, n_acquisitions=1, sim_names=(None, None),
                 compile_moop=True):
    """ Build the 3 design variable, 2 simulation MOOP used across the tests.

    The first simulation has one output (the norm of the design point) and the
    second has two (norms shifted by 1.0 and 0.5).

    Args:
        objectives (iterable): Objective functions of (x, sx).

        constraints (iterable): Constraint functions of (x, sx).

        bounds (list of tuple, optional): Per-variable (lb, ub) pairs.
            Defaults to the unit cube.

        opt (SurrogateOptimizer, optional): Defaults to LocalSurrogate_PS.

        hyperparams (dict, optional): MOOP hyperparameters.

        n_acquisitions (int): How many UniformWeights acquisitions to add.

        sim_names (tuple): Names for the two simulations, or None to let
            ParMOO assign them.

        compile_moop (bool): Whether to compile before returning.

    Returns:
        MOOP: The configured (and by default compiled) MOOP.

    """

    if bounds is None:
        bounds = [(0.0, 1.0)] * 3
    moop = MOOP(opt if opt is not None else LocalSurrogate_PS,
                hyperparams=hyperparams if hyperparams is not None else {})
    for lb, ub in bounds:
        moop.addDesign({'lb': lb, 'ub': ub})
    moop.addSimulation(sim_dict(1, sim_norm, sim_names[0]),
                       sim_dict(2, sim_shifted_norms, sim_names[1]))
    for obj in objectives:
        moop.addObjective({'obj_func': obj})
    for con in constraints:
        moop.addConstraint({'constraint': con})
    for i in range(n_acquisitions):
        moop.addAcquisition({'acquisition': UniformWeights})
    if compile_moop:
        moop.compile()
    return moop


def train_surrogates(moop, points=None, tr_center=None, tr_radius=None,
                     update=False):
    """ Evaluate a set of design points and fit (or update) the surrogates.

    Args:
        moop (MOOP): A compiled MOOP with two simulations.

        points (list of dict, optional): The design points to evaluate.
            Defaults to TRAINING_POINTS.

        tr_center (ndarray, optional): The trust-region center to set after
            fitting.  Defaults to the centroid of the unit cube.

        tr_radius (ndarray or float, optional): The trust-region radius.
            Defaults to 0.5 in every coordinate.

        update (bool): When True call _update_surrogates() instead of
            _fit_surrogates(), for testing incremental fitting.

    """

    sim_names = [si[0] for si in moop.sim_schema]
    for name in sim_names:
        for xi in (points if points is not None else TRAINING_POINTS):
            moop.evaluateSimulation(xi, name)
    if update:
        moop._update_surrogates()
    else:
        moop._fit_surrogates()
    moop._set_surrogate_tr(
        np.ones(3) * 0.5 if tr_center is None else tr_center,
        np.ones(3) * 0.5 if tr_radius is None else tr_radius,
    )
